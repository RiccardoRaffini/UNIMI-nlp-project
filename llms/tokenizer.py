import h5py
import numpy as np
from transformers import GPT2Tokenizer

RECIPES_ADDITIONAL_TOKENS = {
    'additional_special_tokens': [
        '<RECIPE_START>',
        '<RECIPE_END>',
        '<INPUT_START>',
        '<INPUT_END>',
        '<INPUT_NEXT>',
        '<INGREDIENT_START>',
        '<INGREDIENT_END>',
        '<INGREDIENT_NEXT>',
        '<INSTRUCTION_START>',
        '<INSTRUCTION_END>',
        '<INSTRUCTION_NEXT>'
    ]
}

if __name__ == '__main__':
    ## Setup tokenizer
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case = False)
    gpt_tokenizer.add_special_tokens(RECIPES_ADDITIONAL_TOKENS)
    end_token_index = gpt_tokenizer.convert_tokens_to_ids(['<RECIPE_END>'])[0]

    ## Create tokens dataset file
    train_filename = 'recipes_train_text_tokens.txt'
    test_filename = 'recipes_test_text_tokens.txt'

    with h5py.File('recipes_tokenized.h5', 'w') as dataset_file:
        for filename, partition in [(train_filename, 'train'), (test_filename, 'test')]:
            print(f'tokenizing {partition} set')
            complete_examples = []

            with open(filename, 'r') as data_file:
                last_tokens = []

                for line in data_file:
                    line_tokens = gpt_tokenizer.tokenize(line)
                    if len(line_tokens) > 1024:
                        continue

                    line_tokens_indices = gpt_tokenizer.convert_tokens_to_ids(line_tokens)

                    if (len(last_tokens) + len(line_tokens_indices)) <= 1024:
                        last_tokens += line_tokens_indices
                    else:
                        remaining_tokens = 1024 - len(last_tokens)
                        last_tokens.extend([end_token_index]*remaining_tokens)
                        complete_examples.append(last_tokens)
                        last_tokens = line_tokens_indices

            complete_examples_matrix = np.matrix(complete_examples)
            print(complete_examples_matrix.shape)
            dataset_file.create_dataset(partition, data=complete_examples_matrix)
    