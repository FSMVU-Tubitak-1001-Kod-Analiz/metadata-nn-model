import torch
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class RobertaClassificationCollator13Layers:
    def __init__(self, tokenizer, sep_token_id, max_length=768):
        self.tokenizer = tokenizer
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def __call__(self, examples):
        # Convert examples into encodings
        # Extract fields directly (already tokenized)
        input_ids = [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples]
        attention_mask = [torch.tensor(e["attention_mask"], dtype=torch.long) for e in examples]
        label = torch.tensor([e["label"] for e in examples], dtype=torch.long)

        # Pad sequences into a single tensor
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # Head attention mask (use your MLM helper exactly)
        head_attention_mask = self.generate_head_attention_mask(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "head_attention_mask": head_attention_mask,
        }

    def generate_head_attention_mask(self, code_inputs):
        custom_mask_list = [
            self.create_custom_mask(code_inputs[0], self.sep_token_id),
            self.create_custom_mask(code_inputs[0], self.sep_token_id),
            self.create_custom_mask(code_inputs[0], self.sep_token_id),
            self.create_custom_mask(code_inputs[0], self.sep_token_id),
            self.create_custom_mask(code_inputs[0], self.sep_token_id),
            self.create_custom_mask(code_inputs[0], self.sep_token_id),
            self.create_custom_mask(code_inputs[0], self.sep_token_id),
            self.create_custom_mask(code_inputs[0], self.sep_token_id),
            self.create_custom_mask(code_inputs[0], self.sep_token_id),
            self.create_custom_mask(code_inputs[0], self.sep_token_id),
            self.create_custom_mask(code_inputs[0], self.sep_token_id),
            self.create_custom_mask(code_inputs[0], self.sep_token_id)
        ]
    
        head_attention_mask = torch.stack(custom_mask_list, dim=0)
        head_attention_mask = head_attention_mask.unsqueeze(0)
        
        return head_attention_mask
    

    def create_custom_mask(self, token_ids, sep_token_id):
        
        # Convert token_ids to a numpy array
        token_ids = np.array(token_ids)
        seq_len = len(token_ids)

        # Create an array to store statement IDs
        statement_ids = np.zeros(seq_len, dtype=np.int32)
        current_statement = 0
        for i in range(seq_len):
            if token_ids[i] == sep_token_id:
                current_statement += 1
            statement_ids[i] = current_statement

        # print(f'Statement IDs: {statement_ids}')

        # Initialize an attention mask of shape (seq_len, seq_len) with False.
        attn_mask = np.zeros((seq_len, seq_len), dtype=bool)

        # For each token in the actual sequence, allow it to attend to tokens in the next statement.
        for i in range(seq_len):
            current_stmt = statement_ids[i]
            # Find positions among the actual tokens that belong to the next statement.
            next_positions = np.where(statement_ids[:seq_len] == (current_stmt + 1))[0]
            if len(next_positions) > 0:
                attn_mask[i, next_positions] = True

        # Create (no_padding_length x no_padding_length) block mask
        no_padding_length = len(token_ids[token_ids != 1])

        block_mask = np.zeros((seq_len, seq_len), dtype=np.bool_)
        block_mask[:no_padding_length, :no_padding_length] = True
        block_mask = torch.from_numpy(block_mask)

        # Combine both with a bitwise & operation
        attn_mask = (block_mask & attn_mask)

        
        attn_mask = np.where(attn_mask == False, -10000, 0)


        # The padded region (i >= seq_len) will remain all False.
        return torch.tensor(attn_mask)

