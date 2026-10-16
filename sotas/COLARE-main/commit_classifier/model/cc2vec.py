import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import inspect
from torch.autograd import Variable
from registry import MODEL_REGISTRY
from transformers import AutoModel, BertLayer
from multimodal_transformers.model.tabular_config import TabularConfig
from multimodal_transformers.model.tabular_combiner import TabularFeatCombiner


# Make the the multiple attention with word vectors.
def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i]
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)


# The word RNN model for generating a sentence vector
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, batch_size, hidden_size):
        super(WordRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # Word Encoder
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.wordRNN = nn.GRU(embed_size, hidden_size, bidirectional=True)
        # Word Attention
        self.wordattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, inp, hid_state):
        emb_out = self.embed(inp)

        out_state, hid_state = self.wordRNN(emb_out, hid_state)

        word_annotation = self.wordattn(out_state)
        attn = F.softmax(self.attn_combine(word_annotation), dim=1)

        sent = attention_mul(out_state, attn)
        return sent, hid_state


# The sentence RNN model for generating a hunk vector
class SentRNN(nn.Module):
    def __init__(self, sent_size, hidden_size):
        super(SentRNN, self).__init__()
        # Sentence Encoder
        self.sent_size = sent_size
        self.sentRNN = nn.GRU(sent_size, hidden_size, bidirectional=True)

        # Sentence Attention
        self.sentattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, inp, hid_state):
        out_state, hid_state = self.sentRNN(inp, hid_state)

        sent_annotation = self.sentattn(out_state)
        attn = F.softmax(self.attn_combine(sent_annotation), dim=1)

        sent = attention_mul(out_state, attn)
        return sent, hid_state


# The hunk RNN model for generating the vector representation for the instance
class HunkRNN(nn.Module):
    def __init__(self, hunk_size, hidden_size):
        super(HunkRNN, self).__init__()
        # Sentence Encoder
        self.hunk_size = hunk_size
        self.hunkRNN = nn.GRU(hunk_size, hidden_size, bidirectional=True)

        # Sentence Attention
        self.hunkattn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, inp, hid_state):
        out_state, hid_state = self.hunkRNN(inp, hid_state)

        hunk_annotation = self.hunkattn(out_state)
        attn = F.softmax(self.attn_combine(hunk_annotation), dim=1)

        hunk = attention_mul(out_state, attn)
        return hunk, hid_state


# The HAN model
class HierachicalRNN(nn.Module):
    def __init__(self, config):
        super(HierachicalRNN, self).__init__()
        # this vocab_size was calculated on our dataset
        # and hard code here. It should be changed if another
        # dataset is used.
        self.vocab_size = 136167

        # the default setting of CC2Vec
        self.batch_size = config.batch_size
        self.embed_size = 16
        self.hidden_size = 8
        self.hidden_units = 256
        self.code_file = 2
        self.dropout = nn.Dropout(0.5)
        self.cls = 3
        self.config = config

        # Word Encoder
        self.wordRNN = WordRNN(
            self.vocab_size, self.embed_size, self.batch_size, self.hidden_size
        )
        # Sentence Encoder
        self.sentRNN = SentRNN(self.embed_size, self.hidden_size)
        # Hunk Encoder
        self.hunkRNN = HunkRNN(self.embed_size, self.hidden_size)

        # standard neural network layer
        self.standard_nn_layer = nn.Linear(
            self.embed_size * 2 * self.code_file, self.embed_size
        )

        # neural network tensor
        self.W_nn_tensor_one = nn.Linear(
            self.embed_size * self.code_file, self.embed_size * self.code_file
        )
        self.W_nn_tensor_two = nn.Linear(
            self.embed_size * self.code_file, self.embed_size * self.code_file
        )
        self.V_nn_tensor = nn.Linear(self.embed_size * 2 * self.code_file, 2)

        # Hidden layers before putting to the output layer
        self.fc1 = nn.Linear(
            2 * self.embed_size * self.code_file + self.embed_size + 4,
            2 * self.hidden_units,
        )
        self.fc2 = nn.Linear(2 * self.hidden_units, self.cls)
        self.sigmoid = nn.Sigmoid()

    def set_train(self):
        self.batch_size = self.config.batch_size

    def set_eval(self):
        self.batch_size = 1

    def forward_code(self, x, hid_state):
        hid_state_hunk, hid_state_sent, hid_state_word = hid_state
        n_batch, n_file, n_hunk, n_line, n_dim = (
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            x.shape[4],
        )

        # f: file; i: hunk; j: line; k: batch;
        files = list()
        for f in range(n_file):
            hunks = None
            for i in range(n_hunk):
                sents = None
                for j in range(n_line):
                    words = []
                    for k in range(n_batch):
                        words.append(x[k][f][i][j].unsqueeze(1))
                    words = torch.cat(words, dim=1)

                    sent, state_word = self.wordRNN(
                        words,
                        hid_state_word,
                    )
                    if sents is None:
                        sents = sent
                    else:
                        sents = torch.cat((sents, sent), 0)

                hunk, state_sent = self.sentRNN(sents, hid_state_sent)
                if hunks is None:
                    hunks = hunk
                else:
                    hunks = torch.cat((hunks, hunk), 0)
            out_hunk, state_hunk = self.hunkRNN(hunks, hid_state_hunk)
            files.append(out_hunk)
        output = torch.squeeze(torch.cat(files, dim=2))
        if len(output.shape) == 1:
            output = output.unsqueeze(0)
        return output

    def forward(
        self, added_code, removed_code, hid_state_hunk, hid_state_sent, hid_state_word
    ):
        hid_state = (hid_state_hunk, hid_state_sent, hid_state_word)

        x_added_code = self.forward_code(x=added_code, hid_state=hid_state)
        x_removed_code = self.forward_code(x=removed_code, hid_state=hid_state)

        subtract = self.subtraction(
            added_code=x_added_code, removed_code=x_removed_code
        )
        multiple = self.multiplication(
            added_code=x_added_code, removed_code=x_removed_code
        )
        cos = self.cosine_similarity(
            added_code=x_added_code, removed_code=x_removed_code
        )
        euc = self.euclidean_similarity(
            added_code=x_added_code, removed_code=x_removed_code
        )
        nn = self.standard_neural_network_layer(
            added_code=x_added_code, removed_code=x_removed_code
        )
        ntn = self.neural_network_tensor_layer(
            added_code=x_added_code, removed_code=x_removed_code
        )

        x_diff_code = torch.cat((subtract, multiple, cos, euc, nn, ntn), dim=1)

        return x_diff_code

    def forward_commit_embeds_diff(
        self, added_code, removed_code, hid_state_hunk, hid_state_sent, hid_state_word
    ):
        hid_state = (hid_state_hunk, hid_state_sent, hid_state_word)

        x_added_code = self.forward_code(x=added_code, hid_state=hid_state)
        x_removed_code = self.forward_code(x=removed_code, hid_state=hid_state)

        subtract = self.subtraction(
            added_code=x_added_code, removed_code=x_removed_code
        )
        multiple = self.multiplication(
            added_code=x_added_code, removed_code=x_removed_code
        )
        cos = self.cosine_similarity(
            added_code=x_added_code, removed_code=x_removed_code
        )
        euc = self.euclidean_similarity(
            added_code=x_added_code, removed_code=x_removed_code
        )
        nn = self.standard_neural_network_layer(
            added_code=x_added_code, removed_code=x_removed_code
        )
        ntn = self.neural_network_tensor_layer(
            added_code=x_added_code, removed_code=x_removed_code
        )

        x_diff_code = torch.cat((subtract, multiple, cos, euc, nn, ntn), dim=1)
        return x_diff_code

    def forward_commit_embeds(
        self, added_code, removed_code, hid_state_hunk, hid_state_sent, hid_state_word
    ):
        hid_state = (hid_state_hunk, hid_state_sent, hid_state_word)

        x_added_code = self.forward_code(x=added_code, hid_state=hid_state)
        x_removed_code = self.forward_code(x=removed_code, hid_state=hid_state)

        x_diff_code = torch.cat((x_added_code, x_removed_code), dim=1)
        return x_diff_code

    def subtraction(self, added_code, removed_code):
        return added_code - removed_code

    def multiplication(self, added_code, removed_code):
        return added_code * removed_code

    def cosine_similarity(self, added_code, removed_code):
        cosine = nn.CosineSimilarity(eps=1e-6)
        return cosine(added_code, removed_code).view(self.batch_size, 1)

    def euclidean_similarity(self, added_code, removed_code):
        euclidean = nn.PairwiseDistance(p=2)
        return euclidean(added_code, removed_code).view(self.batch_size, 1)

    def standard_neural_network_layer(self, added_code, removed_code):
        concat = torch.cat((removed_code, added_code), dim=1)
        output = self.standard_nn_layer(concat)
        output = F.relu(output)
        return output

    def neural_network_tensor_layer(self, added_code, removed_code):
        output_one = self.W_nn_tensor_one(removed_code)
        output_one = torch.mul(output_one, added_code)
        output_one = torch.sum(output_one, dim=1).view(self.batch_size, 1)

        output_two = self.W_nn_tensor_two(removed_code)
        output_two = torch.mul(output_two, added_code)
        output_two = torch.sum(output_two, dim=1).view(self.batch_size, 1)

        W_output = torch.cat((output_one, output_two), dim=1)
        code = torch.cat((removed_code, added_code), dim=1)
        V_output = self.V_nn_tensor(code)
        return F.relu(W_output + V_output)

    def init_hidden_hunk(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()

    def init_hidden_sent(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()

    def init_hidden_word(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()


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


@MODEL_REGISTRY.register("CC2Vec")
class CC2VecBasedModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # replace the code diff embedding module with CC2Vec
        self.code_change_encoder = HierachicalRNN(config)

        # Since we replace code diff embedding module with CC2Vec,
        # the commit message encoder should be a independent CodeBERT
        self.msg_encoder = AutoModel.from_pretrained("microsoft/codebert-base")

        # The remain components for MGCC stay unchanged
        tabular_config = TabularConfig(
            num_labels=3,
            combine_feat_method="gating_on_cat_and_num_feats_then_sum",
            numerical_feat_dim=29,
            numerical_bn=False,
            mlp_dropout=0.15,
        )
        tabular_config.text_feat_dim = 768
        self.feature_combiner = TabularFeatCombiner(tabular_config)
        self.text_code_combiner = FeedForward(768 + 84, 768, 2048)
        self.classifier = nn.Linear(768, 3)

    def forward(
        self,
        input_ids,
        attention_mask,
        added_code,
        deleted_code,
        numerical_features,
        hid_state_hunk,
        hid_state_sent,
        hid_state_word,
        **kwargs,
    ):
        msg_embeding = self.msg_encoder(
            input_ids=input_ids,
            # token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ).pooler_output
        code_embeding = self.code_change_encoder(
            added_code,
            deleted_code,
            hid_state_hunk,
            hid_state_sent,
            hid_state_word,
        )

        combined = self.text_code_combiner(
            torch.cat((msg_embeding, code_embeding), dim=1)
        )

        commit_embeding = self.feature_combiner(
            combined,
            numerical_feats=numerical_features,
        )
        return self.classifier(commit_embeding)

    def init_hidden_hunk(self, batch_size=None):
        return self.code_change_encoder.init_hidden_hunk(batch_size)

    def init_hidden_sent(self, batch_size=None):
        return self.code_change_encoder.init_hidden_sent(batch_size)

    def init_hidden_word(self, batch_size=None):
        return self.code_change_encoder.init_hidden_word(batch_size)

    def set_train(self):
        self.code_change_encoder.set_train()

    def set_eval(self):
        self.code_change_encoder.set_eval()
