from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaModel
from roberta_model_13_layers import RobertaModel13Layers
from transformers import AutoTokenizer
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import Trainer, TrainingArguments
from roberta_for_mlm_13_layers import RobertaForMLM13Layers
from datasets import load_dataset
from roberta_data_collator_13_layers import RobertaCollator13Layers


def Main():
    
    # Load GCB's config settings
    conf = RobertaConfig.from_pretrained("microsoft/graphcodebert-base")

    # Change number of layers
    conf.num_hidden_layers = 13

    # Load GCB's base model
    base_model = RobertaModel.from_pretrained("microsoft/graphcodebert-base", config=conf)

    # Instance of custom roberta model
    model = RobertaForMLM13Layers(conf)

    # Load weights of GCB onto the custom model
    model.load_state_dict(base_model.state_dict(), strict=False)



    # Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

    # Add <sep> as a special token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<sep>"]})

    # Extract token ID
    sep_token_id = tokenizer.convert_tokens_to_ids("<sep>")

    model.resize_token_embeddings(len(tokenizer))

    

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train_step(batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # Generate custom head_attention_mask for the 13th layer
        head_attention_mask = generate_head_attention_mask(input_ids, sep_token_id)
        
        # Forward pass with custom mask
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_attention_mask=head_attention_mask,  # Passed to 13th layer
            labels=input_ids  # For MLM loss
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        return loss




    # Load the tiny dataset
    dataset = load_dataset("json", data_files="custom_attnmasks_13_layers/small_dataset.jsonl")['train']
    print(f'dataset:\n{dataset}')

    # Initialize tokenizer (using GraphCodeBERT's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

    # Tokenize function
    def tokenize_fn(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            padding="max_length",
            return_special_tokens_mask=True
    )

    # Apply tokenization
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["code", "language"])
    print(tokenized_dataset)  # Inspect first sample


    # Initialize
    collator = RobertaCollator13Layers(
        tokenizer=tokenizer,
        sep_token_id=sep_token_id,
        mlm_probability=0.15
    )

    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=2,
        collate_fn=collator
    )

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=2,
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset,
    #     data_collator=collator,
    # )

    # trainer.train()  # Works seamlessly with standard outputs

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_epochs = 2
    
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Move data to device (GPU/CPU)
            input_ids = batch["input_ids"].to(device)
            head_attention_mask = batch["head_attention_mask"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_attention_mask= head_attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optional: Logging
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

def mask_tokens(tokenizer, code):
    return tokenizer(code, truncation=True, padding="max_length", max_length=512, return_special_tokens_mask=True)


def generate_head_attention_mask(code_inputs, sep_token_id):
    print(code_inputs)
    custom_mask_list = [
        create_custom_mask(code_inputs[0], sep_token_id),
        create_custom_mask(code_inputs[0], sep_token_id),
        create_custom_mask(code_inputs[0], sep_token_id),
        create_custom_mask(code_inputs[0], sep_token_id),
        create_custom_mask(code_inputs[0], sep_token_id),
        create_custom_mask(code_inputs[0], sep_token_id),
        create_custom_mask(code_inputs[0], sep_token_id),
        create_custom_mask(code_inputs[0], sep_token_id),
        create_custom_mask(code_inputs[0], sep_token_id),
        create_custom_mask(code_inputs[0], sep_token_id),
        create_custom_mask(code_inputs[0], sep_token_id),
        create_custom_mask(code_inputs[0], sep_token_id)
    ]

    head_attention_mask = torch.stack(custom_mask_list, dim=0)
    head_attention_mask = head_attention_mask.unsqueeze(0)
    
    return head_attention_mask

def create_custom_mask(token_ids, sep_token_id):

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
    print(f'No Padding length = {no_padding_length}')

    block_mask = np.zeros((512, 512), dtype=np.bool_)
    block_mask[:no_padding_length, :no_padding_length] = True
    block_mask = torch.from_numpy(block_mask)

    # Combine both with a bitwise & operation
    attn_mask = (block_mask & attn_mask)


    attn_mask = np.where(attn_mask == False, -10000, 0)


    # The padded region (i >= seq_len) will remain all False.
    return torch.tensor(attn_mask)




if __name__ == "__main__":
    Main()