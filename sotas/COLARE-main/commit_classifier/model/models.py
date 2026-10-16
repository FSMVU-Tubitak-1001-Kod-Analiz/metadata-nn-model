import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from multimodal_transformers.model.tabular_config import TabularConfig
from multimodal_transformers.model.tabular_combiner import TabularFeatCombiner
from registry import MODEL_REGISTRY


class Reducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.15,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.transformer_encoder_layer, num_layers=2
        )

    def forward(self, x, attention_mask):
        token_embeddings = self.transformer_encoder(
            x, src_key_padding_mask=attention_mask
        )

        attention_mask_rev = (~attention_mask).int()
        attention_mask_rev = attention_mask_rev.transpose(1, 0)

        input_mask_expanded = (
            attention_mask_rev.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0)
        sum_mask = input_mask_expanded.sum(0)
        # sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled = sum_embeddings / sum_mask
        return pooled


class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = None,
        dropout: int = 0.15,
        add_norm: bool = False,
    ) -> None:
        super().__init__()
        if add_norm and (in_features != out_features):
            raise ValueError

        if hidden_features is None:
            hidden_features = 4 * in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
        self.add_norm = add_norm
        if add_norm:
            self.layernorm = nn.LayerNorm(out_features)

    def _feedforward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x

    def forward(self, x):
        if self.add_norm:
            return self.layernorm(x + self._feedforward(x))
        return self._feedforward(x)


class CodeChangeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hunk_encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.hunk_compare_linear1 = nn.Linear(768 * 2, 768)
        self.hunk_compare_linear2 = nn.Linear(768, 768)
        self.hunk_compare_poller = FeedForward(768 * 3, 128, 1024)
        self.hunk_reducer = Reducer()
        self.file_reducer = Reducer()
        self.layernorm = nn.LayerNorm(768 * 3)

    def encode_hunk(self, input_ids, attention_mask):
        return self.hunk_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output

    def subtraction(self, added_code, removed_code):
        return added_code - removed_code

    def multiplication(self, added_code, removed_code):
        return added_code * removed_code

    def forward_compare_linear(self, added_code, removed_code):
        concat = torch.cat((removed_code, added_code), dim=1)
        output = self.hunk_compare_linear1(concat)
        output = F.relu(output)
        output = self.hunk_compare_linear2(output)
        return output

    def cosine_similarity(self, added_code, removed_code):
        cosine = nn.CosineSimilarity(eps=1e-6)
        return cosine(added_code, removed_code).view(-1, 1)

    def euclidean_similarity(self, added_code, removed_code):
        euclidean = nn.PairwiseDistance(p=2)
        return euclidean(added_code, removed_code).view(-1, 1)

    def compare_hunk_features(self, hunk_add_feature, hunk_del_feature):
        sub = self.subtraction(hunk_add_feature, hunk_del_feature)  # (batch, 768)
        mul = self.multiplication(hunk_add_feature, hunk_del_feature)  # (batch, 768)
        lin = self.forward_compare_linear(
            hunk_add_feature, hunk_del_feature
        )  # (batch, 768)
        out = self.layernorm(torch.cat((sub, mul, lin), dim=1))
        out = self.hunk_compare_poller(out)
        return out

    def forward_hunk(
        self,
        hunk_add_input_ids,
        hunk_add_attention_mask,
        hunk_delete_input_ids,
        hunk_delete_attention_mask,
    ):
        n_batch, n_hunk = hunk_add_input_ids.shape[0], hunk_add_input_ids.shape[1]
        hunk_features = []

        for i in range(n_hunk):
            hunk_add_input_ids_i, hunk_add_attention_mask_i = [], []
            hunk_del_input_ids_i, hunk_del_attention_mask_i = [], []
            for b in range(n_batch):
                hunk_add_input_ids_i.append(hunk_add_input_ids[b][i])
                hunk_add_attention_mask_i.append(hunk_add_attention_mask[b][i])
                hunk_del_input_ids_i.append(hunk_delete_input_ids[b][i])
                hunk_del_attention_mask_i.append(hunk_delete_attention_mask[b][i])

            hunk_add_input_ids_i = torch.stack(hunk_add_input_ids_i, dim=0)
            hunk_add_attention_mask_i = torch.stack(hunk_add_attention_mask_i, dim=0)
            hunk_del_input_ids_i = torch.stack(hunk_del_input_ids_i, dim=0)
            hunk_del_attention_mask_i = torch.stack(hunk_del_attention_mask_i, dim=0)

            hunk_add_feature_i = self.encode_hunk(
                hunk_add_input_ids_i, hunk_add_attention_mask_i
            )
            hunk_del_feature_i = self.encode_hunk(
                hunk_del_input_ids_i, hunk_del_attention_mask_i
            )
            assert hunk_add_feature_i.shape == hunk_del_feature_i.shape
            assert hunk_add_feature_i.shape == torch.Size([n_batch, 768])

            hunk_feature = self.compare_hunk_features(
                hunk_add_feature_i, hunk_del_feature_i
            )  # (batch, 128)
            hunk_features.append(hunk_feature)

        hunk_features = torch.stack(hunk_features, dim=1)  # (batch, hunk, 128)
        return hunk_features

    def forward(
        self,
        code_add_input_ids,
        code_add_attention_mask,
        code_del_input_ids,
        code_del_attention_mask,
        file_attention_mask,
        hunk_attention_mask,
    ):
        n_batch, n_file, n_hunk = (
            code_add_input_ids.shape[0],
            code_add_input_ids.shape[1],
            code_add_input_ids.shape[2],
        )

        code_features = []
        for i in range(n_file):
            file_add_input_ids_i, file_add_attention_mask_i = [], []
            file_del_input_ids_i, file_del_attention_mask_i = [], []
            for b in range(n_batch):
                file_add_input_ids_i.append(code_add_input_ids[b][i])
                file_add_attention_mask_i.append(code_add_attention_mask[b][i])
                file_del_input_ids_i.append(code_del_input_ids[b][i])
                file_del_attention_mask_i.append(code_del_attention_mask[b][i])

            file_add_input_ids_i = torch.stack(file_add_input_ids_i, dim=0)
            file_add_attention_mask_i = torch.stack(file_add_attention_mask_i, dim=0)
            file_del_input_ids_i = torch.stack(file_del_input_ids_i, dim=0)
            file_del_attention_mask_i = torch.stack(file_del_attention_mask_i, dim=0)

            code_feature = self.forward_hunk(
                file_add_input_ids_i,
                file_add_attention_mask_i,
                file_del_input_ids_i,
                file_del_attention_mask_i,
            )
            code_features.append(code_feature)

        code_features = torch.stack(code_features, dim=1)
        assert code_features.shape == torch.Size([n_batch, n_file, n_hunk, 128])

        files = None
        for f in range(n_file):
            hunks = []
            hunk_attention_masks = []
            for b in range(n_batch):
                hunk_attention_masks.append(hunk_attention_mask[b][f])
            for h in range(n_hunk):
                hunk = []
                for b in range(n_batch):
                    hunk.append(code_features[b][f][h])
                hunk = torch.stack(hunk, dim=0)
                assert hunk.shape == torch.Size([n_batch, 128])
                hunks.append(hunk)
            hunks = torch.stack(hunks, dim=0)
            hunk_attention_masks = torch.stack(hunk_attention_masks, dim=0)
            assert hunks.shape == torch.Size([n_hunk, n_batch, 128])
            assert hunk_attention_masks.shape == torch.Size([n_batch, n_hunk])
            hunks_feature = self.hunk_reducer(hunks, hunk_attention_masks)
            hunks_feature = hunks_feature.unsqueeze(dim=0)

            if files is None:
                files = hunks_feature
            else:
                files = torch.cat((files, hunks_feature), dim=0)
        assert files.shape == torch.Size([n_file, n_batch, 128]), files.shape

        commit_feature = self.file_reducer(files, file_attention_mask)
        assert commit_feature.shape == torch.Size([n_batch, 128])
        return commit_feature


@MODEL_REGISTRY.register("CCModel")
class CCModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.code_change_encoder = CodeChangeEncoder()

        tabular_config = TabularConfig(
            num_labels=3,
            combine_feat_method="gating_on_cat_and_num_feats_then_sum",
            numerical_feat_dim=29,
            numerical_bn=False,
            mlp_dropout=0.15,
        )
        tabular_config.text_feat_dim = 768
        self.feature_combiner = TabularFeatCombiner(tabular_config)
        self.text_code_combiner = FeedForward(768 + 128, 768, 2048)
        self.classifier = nn.Linear(768, 3)

    def forward(
        self,
        input_ids,
        attention_mask,
        codes_add_input_ids,
        codes_add_attention_mask,
        codes_delete_input_ids,
        codes_delete_attention_mask,
        file_attention_mask,
        hunk_attention_mask,
        numerical_features,
        **kwargs,
    ):
        msg_embeding = self.code_change_encoder.hunk_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output

        code_embeding = self.code_change_encoder(
            code_add_input_ids=codes_add_input_ids,
            code_add_attention_mask=codes_add_attention_mask,
            code_del_input_ids=codes_delete_input_ids,
            code_del_attention_mask=codes_delete_attention_mask,
            file_attention_mask=file_attention_mask,
            hunk_attention_mask=hunk_attention_mask,
        )

        combined = self.text_code_combiner(
            torch.cat((msg_embeding, code_embeding), dim=1)
        )

        commit_embeding = self.feature_combiner(
            combined,
            numerical_feats=numerical_features,
        )
        return self.classifier(commit_embeding)


@MODEL_REGISTRY.register("CodeBERTBaseline")
class CCModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.roberta = AutoModel.from_pretrained("roberta-base")
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.classifier = FeedForward(768 * 2, 3, 768 * 4)

    def forward(
        self,
        input_ids,
        attention_mask,
        diff_input_ids,
        diff_attention_mask,
        **kwargs,
    ):
        msg_embeding = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output
        diff_embedding = self.codebert(
            input_ids=diff_input_ids, attention_mask=diff_attention_mask
        ).pooler_output
        return self.classifier(torch.cat((msg_embeding, diff_embedding), dim=1))


@MODEL_REGISTRY.register("CodeFeatModel")
class CodeFeatModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.code_change_encoder = CodeChangeEncoder()
        tabular_config = TabularConfig(
            num_labels=3,
            combine_feat_method="gating_on_cat_and_num_feats_then_sum",
            numerical_feat_dim=29,
            numerical_bn=False,
            mlp_dropout=0.15,
        )
        tabular_config.text_feat_dim = 128
        self.feature_combiner = TabularFeatCombiner(tabular_config)
        self.classifier = nn.Linear(128, 3)

    def forward(
        self,
        codes_add_input_ids,
        codes_add_attention_mask,
        codes_delete_input_ids,
        codes_delete_attention_mask,
        file_attention_mask,
        hunk_attention_mask,
        numerical_features,
        **kwargs,
    ):
        code_embeding = self.code_change_encoder(
            code_add_input_ids=codes_add_input_ids,
            code_add_attention_mask=codes_add_attention_mask,
            code_del_input_ids=codes_delete_input_ids,
            code_del_attention_mask=codes_delete_attention_mask,
            file_attention_mask=file_attention_mask,
            hunk_attention_mask=hunk_attention_mask,
        )

        commit_embeding = self.feature_combiner(
            code_embeding,
            numerical_feats=numerical_features,
        )
        return self.classifier(commit_embeding)


@MODEL_REGISTRY.register("MessageCodeModel")
class MessageModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.code_change_encoder = CodeChangeEncoder()
        self.text_code_combiner = FeedForward(768 + 128, 768, 2048)
        self.classifier = nn.Linear(768, 3)

    def forward(
        self,
        input_ids,
        attention_mask,
        codes_add_input_ids,
        codes_add_attention_mask,
        codes_delete_input_ids,
        codes_delete_attention_mask,
        file_attention_mask,
        hunk_attention_mask,
        **kwargs,
    ):
        msg_embeding = self.code_change_encoder.hunk_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output
        code_embeding = self.code_change_encoder(
            code_add_input_ids=codes_add_input_ids,
            code_add_attention_mask=codes_add_attention_mask,
            code_del_input_ids=codes_delete_input_ids,
            code_del_attention_mask=codes_delete_attention_mask,
            file_attention_mask=file_attention_mask,
            hunk_attention_mask=hunk_attention_mask,
        )

        combined = self.text_code_combiner(
            torch.cat((msg_embeding, code_embeding), dim=1)
        )

        return self.classifier(combined)


@MODEL_REGISTRY.register("MessageFeatModel")
class MessageFeatModel(nn.Module):
    def __init__(self, config=None) -> None:
        super().__init__()
        self.config = config
        self.message_encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Linear(768, 3)
        tabular_config = TabularConfig(
            num_labels=3,
            combine_feat_method="gating_on_cat_and_num_feats_then_sum",
            numerical_feat_dim=29,
            numerical_bn=False,
            mlp_dropout=0.15,
        )
        tabular_config.text_feat_dim = 768
        self.feature_combiner = TabularFeatCombiner(tabular_config)

    def forward(
        self,
        input_ids,
        attention_mask,
        numerical_features,
        **kwargs,
    ):
        msg_embeding = self.message_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output
        commit_embeding = self.feature_combiner(
            msg_embeding,
            numerical_feats=numerical_features,
        )
        out = self.classifier(commit_embeding)
        return out
