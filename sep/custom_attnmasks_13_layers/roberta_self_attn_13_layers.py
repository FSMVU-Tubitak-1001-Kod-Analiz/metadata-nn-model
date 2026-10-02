from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaConfig, RobertaEncoder, RobertaLayer, RobertaSelfAttention
import torch
import math

class RobertaSelfAttention13Layers(RobertaSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_attention_mask=None,
        output_attentions=False,
    ):
        # Standard query/key/value projections
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Reshape to separate heads
        batch_size, seq_len, _ = hidden_states.size()
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the default attention_mask (for padding, etc.)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if head_attention_mask is not None:
            # Use torch.min to enforce "strict" masking
            attention_scores = torch.min(
                attention_scores,
                head_attention_mask  # Mask statements and special tokens
            ).float()

        # Normalize and compute context
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)
        
        # Merge heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.all_head_size)
        
        # Return context and attention probabilities (if requested)
        return (context, attention_probs) if output_attentions else (context,)