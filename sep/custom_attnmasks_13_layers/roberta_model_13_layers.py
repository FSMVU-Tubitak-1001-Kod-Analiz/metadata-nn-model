from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaEmbeddings
from .roberta_encoder_13_layers import RobertaEncoder13Layers
import torch

class RobertaModel13Layers(RobertaModel):
    def __init__(self, config):
        super(RobertaModel, self).__init__(config)  # Bypass parent init
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder13Layers(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_attention_mask=None,  # Shape: [batch_size, num_heads, seq_len, seq_len]
        token_type_ids=None,
        position_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # Standard embedding and mask preparation
        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        
        # Generate token_type_ids if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        # Prepare extended attention mask
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=input_ids.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
        # Embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # Pass through encoder
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_attention_mask=head_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs