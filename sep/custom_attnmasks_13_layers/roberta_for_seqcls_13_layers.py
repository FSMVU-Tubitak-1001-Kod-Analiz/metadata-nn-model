import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from .roberta_model_13_layers import RobertaModel13Layers
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead


class RobertaForSequenceClassification13Layers(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Use your custom 13-layer Roberta model
        self.roberta = RobertaModel13Layers(config)

        # Same classification head as Hugging Face
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_attention_mask=None,  # custom arg
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            head_attention_mask=head_attention_mask,
        )

        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        logits = self.classifier(sequence_output)  # internally applies dropout + tanh

        loss = None
        if labels is not None:
            if self.num_labels == 1:  # Regression task
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:  # Classification task
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
