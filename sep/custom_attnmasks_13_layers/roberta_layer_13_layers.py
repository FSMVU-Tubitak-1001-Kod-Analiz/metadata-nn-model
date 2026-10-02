from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaIntermediate, RobertaOutput
from .roberta_self_attn_13_layers import RobertaSelfAttention13Layers
from .roberta_attn_13_layers import RobertaAttention13Layers

class RobertaLayer13Layers(RobertaLayer):
    def __init__(self, config):
        super(RobertaLayer, self).__init__()  # Bypass parent init
        self.attention = RobertaAttention13Layers(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_attention_mask=head_attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        
        # Feed-forward
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        outputs = (layer_output,) + self_attention_outputs[1:]
        return outputs
