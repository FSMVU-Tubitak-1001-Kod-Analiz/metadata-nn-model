from torch.nn import Embedding

import sys
import os
HERE = os.path.dirname(__file__)                               
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))          
sys.path.insert(0, ROOT)

from datetime import datetime
# import logic.ast_model.ast_marker as astm
# import logic.transformer.tokenizing.parsing.ast_parser as astp
import model_trainer
# from logic.transformer.tokenizing.train_tokenizer import ASTTokenizer, ASTTokenizerRoberta
from unidecode import unidecode
from statement_sep.build_samples import SampleBuilder, DatasetType
from logic.transformer.utils import ModelType, build_tokenizer
from code_utils import increment_file
from pathlib import Path
from transformers import PreTrainedTokenizerFast, BertConfig, BertForMaskedLM, RobertaModel, RobertaConfig, RobertaForMaskedLM, AutoTokenizer
from custom_attnmasks_13_layers.roberta_for_mlm_13_layers import RobertaForMLM13Layers
from custom_attnmasks_13_layers.roberta_data_collator_13_layers import RobertaCollator13Layers
from transformers import DataCollatorForLanguageModeling


def mark_ast(files_folder, dataset_type, result_path):
    parser = astp.AstParser()
    marker = astm.AstMarker(parser)
    builder = SampleBuilder()
    builder.write_in_batches(files_folder, dataset_type, marker.build_input, result_path, max_workers=4)


def train_tokenizer(ast_file, model_type, tokenizer_path):
    if model_type == ModelType.BERT:
        ast_tok = ASTTokenizer()
    elif model_type == ModelType.ROBERTA:
        ast_tok = ASTTokenizerRoberta()
    else:
        raise Exception("Illegal argument type")
    ast_tok.train(ast_file, tokenizer_path)

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze 13th layer
    for param in model.roberta.encoder.layer[12].parameters():
        param.requires_grad = True

    # Unfreeze word embeddings
    for param in model.roberta.embeddings.word_embeddings.parameters():
        param.requires_grad = True

def construct_model(model_type, model_config_name, tokenizer, from_scratch, freeze_pretrained_part):
    if model_type == ModelType.BERT:
        config = BertConfig.from_pretrained(
            model_config_name,
            vocab_size=len(tokenizer)
        )

        if from_scratch:
            model = BertForMaskedLM(config)
            prefix = "ast_model_"
        else:
            model = BertForMaskedLM.from_pretrained(model_config_name)
            prefix = "ast_model_fine_tuned_"

    elif model_type == ModelType.ROBERTA:

        # Load Roberta's config settings
        conf = RobertaConfig.from_pretrained(model_config_name)
        
        if from_scratch: # Load pretrained roberta model
            model = RobertaForMaskedLM.from_pretrained(model_config_name, config=conf)

            prefix = "pretrained_roberta_model_"
        else:
            # Change number of layers
            conf.num_hidden_layers = 13

            # Load RoBERTa's base model
            base_model = RobertaForMaskedLM.from_pretrained(model_config_name, config=conf)

            # Instance of 13 layer roberta model
            model = RobertaForMLM13Layers(conf)
            print(f"First layer's weights before: {model.roberta.encoder.layer[0].attention.self.query.weight}")

            # Load weights of RoBERTa onto the custom model
            model.load_state_dict(base_model.state_dict(), strict=False)

            print(f"First layer's weights after: {model.roberta.encoder.layer[0].attention.self.query.weight}")
            model.resize_token_embeddings(len(tokenizer))

            if freeze_pretrained_part:
                prefix = "frozen_sep_model_fine_tuned_"
                freeze_params(model=model)

                # print("\nTrainable parameters after freezing:")
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         print(name)

            else:    
                prefix = "sep_model_fine_tuned_"

    else:
        raise Exception("Illegal model type.")

    return conf, model, prefix


def train_model(
        train_file_path,
        test_file_path,
        model_type,
        position_embeddings_multiplier=1,
        base_number_of_tokens=512,
        model_config_name="sentence-transformers/bert-base-nli-mean-tokens",
        data_ratio: tuple = (0., 5.),
        do_train_from_scratch=True,
        use_pretrained_tokenizer=True,
        freeze_pretrained_part=False
):
    """
    Train model function defined specifically for AST models. Only supply AST marked files, tokenizing will be done here.
    Args:
        do_train_from_scratch: If true, it will initialize the model with random weights. Else, it will load the pretrained model for the given config.
        use_pretrained_tokenizer: If true, it will train a new tokenizer else it will load from config.
        model_type: Only defined for Bert and Roberta for now. (and likely forever)
        model_config_name: HuggingFace API key of the model to use as base config. Default is "sentence-transformers/bert-base-nli-mean-tokens"
        data_ratio: Percentage of the data range to use in training. Default value is (0., 1.) to indicate all the data. First value denotes the start percentage of the data. For instance, for a value of (0.4, 1), data will be used from the value at 40%th position to the end.
        test_file_path: Path to the test file.
        train_file_path: Path to the train file
        tokenizer_path: Path to tokenizer. Tokenizer object will be built inside the function.
        base_number_of_tokens: 510 is base mate, works for all.
        position_embeddings_multiplier: How much to multiply position embeddings layer size by. Default is 1.
    """

    # Building SEP tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<sep>"]})
    sep_token_id = tokenizer.convert_tokens_to_ids("<sep>")

    config, model, prefix = construct_model(model_type, model_config_name, tokenizer, do_train_from_scratch, freeze_pretrained_part)
    config.max_position_embeddings *= position_embeddings_multiplier

    # Initialize custom collator
    collator = RobertaCollator13Layers(
        tokenizer=tokenizer,
        sep_token_id=sep_token_id,
        mlm_probability=0.15
    )

    # # Default collator
    # collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    trainer = model_trainer.ModelTrainer(model, tokenizer, collator, f"tensorboard_results/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    builder = SampleBuilder()

    def data_range(code): return slice(round(len(code) * data_ratio[0]), round(len(code) * data_ratio[1]))

    train_codes = builder.read_sample(train_file_path)
    train_codes = train_codes[data_range(train_codes)]
    
    test_codes = builder.read_sample(test_file_path)
    test_codes = test_codes[data_range(test_codes)]

    trainer.train(
        train_codes, 
        test_codes,
        "data/sep_temp_train/",
        "data/sep_temp_test/",
        False, # Keep tokenize results
        base_number_of_tokens * position_embeddings_multiplier,
        num_train_epochs=20,
        train_file=train_file_path,
        test_file=test_file_path,
        data_ratio=data_ratio,
        position_embedding_size=base_number_of_tokens * position_embeddings_multiplier,
        model_config_name=model_config_name,
        model_type=model_type.value,
    )


def run():
    # train_files_folder_path = "../../data/java-large/training"
    # test_files_folder_path = "../../data/java-large/test"

    # train_ast_path = increment_file("../../data/ast_results", "ast_data_train_", ".json")
    # test_ast_path = increment_file("../../data/ast_results", "ast_data_test_", ".json")

    # mark_ast(train_files_folder_path, DatasetType.TRAIN, train_ast_path)
    # mark_ast(test_files_folder_path, DatasetType.TRAIN, test_ast_path)

    # tokenizer_path = increment_file("../../tokenizers", "ast_tokenizer_roberta_", ".json")
    # train_tokenizer("/home/eislamoglu/PycharmProjects/transformer_init/data/ast_results/ast_data_train_1.json", ModelType.ROBERTA, tokenizer_path)

    # file_path = '/home/eislamoglu/PycharmProjects/transformer_init/data/ast_results/ast_data_train_2.json'
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f):
    #         data = json.loads(line)
    #         print(data)
    #         if i >= 9:
    #             break

    # train_path = "/home/nahmed/CodeSmellProject/Repos/transformers_init/statement_sep/statement_sep_output/10_percent_train.json"
    # test_path = "/home/nahmed/CodeSmellProject/Repos/transformers_init/statement_sep/statement_sep_output/10_percent_test.json"
    train_path = "C:/Users/FSM/Desktop/NIZAM/Code1001CodeSmellDetection/transformers_init/statement_sep/statement_sep_output/all_data_train.json"
    test_path = "C:/Users/FSM/Desktop/NIZAM/Code1001CodeSmellDetection/transformers_init/statement_sep/statement_sep_output/all_data_test.json"

    # tokenizer_path = increment_file("../tokenizers", "ast_tokenizer_roberta_", ".json")
    # train_tokenizer_roberta(train_ast_path, tokenizer_path)

    # tokenizer_path = "/home/eislamoglu/PycharmProjects/transformer_init/tokenizers/ast_tokenizer_roberta_2.json"
    # print("Done training tokenizer")

    train_model(
        train_path,
        test_path,
        ModelType.ROBERTA,
        model_config_name="roberta-base",
        data_ratio=(0., .25),
        do_train_from_scratch=False, # False for custom 13 layer roberta model, True for default pretrained roberta model
        use_pretrained_tokenizer=False,
        freeze_pretrained_part=True
    )


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # import ssl
    # import warnings # To suppress warnings if needed
    # from urllib3.exceptions import InsecureRequestWarning # For requests/urllib3

    # print("Attempting UNSAFE workaround: Globally disabling SSL certificate verification.")

    # os.environ['HF_HUB_VERIFY_SSL'] = '0'
    # os.environ['REQUESTS_CA_BUNDLE'] = ''
    # warnings.simplefilter('ignore', InsecureRequestWarning)

    # try:
    #     _create_unverified_https_context = ssl._create_unverified_context
    # except AttributeError:
    #     print("WARNING: ssl._create_unverified_context not found. Cannot apply global SSL unverified context.")
    #     pass
    # else:
    #     ssl._create_default_https_context = _create_unverified_https_context
    #     print("SUCCESS: Applied global SSL unverified context. All HTTPS requests from this script will not verify SSL certs.")
    
    run()
