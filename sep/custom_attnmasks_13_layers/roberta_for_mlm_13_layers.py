from transformers import RobertaForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from .roberta_model_13_layers import RobertaModel13Layers
import torch

class RobertaForMLM13Layers(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel13Layers(config)
        self.lm_head = RobertaLMHead(config)

    def forward(self, input_ids, attention_mask=None, head_attention_mask=None, labels=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            head_attention_mask=head_attention_mask,
        )
        logits = self.lm_head(outputs[0])
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)