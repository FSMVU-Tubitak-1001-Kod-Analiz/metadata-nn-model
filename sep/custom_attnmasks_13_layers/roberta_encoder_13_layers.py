import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaEncoder
from .roberta_layer_13_layers import RobertaLayer13Layers

class RobertaEncoder13Layers(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [RobertaLayer(config) for _ in range(12)] +    # Original 12 layers
            [RobertaLayer13Layers(config)]                 # Custom 13th layer
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Only pass head_attention_mask to the 13th layer
            if i == 12:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_attention_mask=head_attention_mask,
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
