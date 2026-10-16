import torch.nn as nn
from transformers import AutoConfig

from multimodal_transformers.model import AutoModelWithTabular
from multimodal_transformers.model import TabularConfig


class CommitClassifier(nn.Module):
    """
    This is only a test model build for dashboard.

    To use COLARE, you must train COLARE following 'readme.md' and replace this.
    """
    def __init__(self):
        super().__init__()
        self.bert_model = "bert-base-uncased"
        self.bert_config = AutoConfig.from_pretrained(self.bert_model)
        tabular_config = TabularConfig(
            combine_feat_method="gating_on_cat_and_num_feats_then_sum",
            num_labels=3,
            numerical_feat_dim=21,
        )
        self.bert_config.tabular_config = tabular_config
        self.model = AutoModelWithTabular.from_pretrained(
            self.bert_model, config=self.bert_config
        )

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        numerical_features=None,
        **kwargs,
    ):
        _, _, layer_out = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            numerical_feats=numerical_features,
        )
        return layer_out[1]
