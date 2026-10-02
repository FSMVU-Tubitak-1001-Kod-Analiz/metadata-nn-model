from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaIntermediate, RobertaOutput, RobertaSelfOutput
from .roberta_self_attn_13_layers import RobertaSelfAttention13Layers
import torch
import torch.nn as nn

class RobertaAttention13Layers(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = RobertaSelfAttention13Layers(config)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_attention_mask=head_attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs