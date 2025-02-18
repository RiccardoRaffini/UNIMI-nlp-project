import h5py
import numpy as np
import pandas as pd
from typing import Any

RECIPE_START_TOKEN = '<RECIPE_START>'
RECIPE_END_TOKEN = '<RECIPE_END>'
INPUT_START_TOKEN = '<INPUT_START>'
INPUT_END_TOKEN = '<INPUT_END>'
INPUT_NEXT_TOKEN = '<INPUT_NEXT>'
INGREDIENT_START_TOKEN = '<INGREDIENT_START>'
INGREDIENT_END_TOKEN = '<INGREDIENT_END>'
INGREDIENT_NEXT_TOKEN = '<INGREDIENT_NEXT>'
INSTRUCTION_START = '<INSTRUCTION_START>'
INSTRUCTION_END = '<INSTRUCTION_END>'
INSTRUCTION_NEXT = '<INSTRUCTION_NEXT>'

RECIPES_ADDITIONAL_TOKENS = {
    'additional_special_tokens': [
        RECIPE_START_TOKEN,
        RECIPE_END_TOKEN,
        INPUT_START_TOKEN,
        INPUT_END_TOKEN,
        INPUT_NEXT_TOKEN,
        INGREDIENT_START_TOKEN,
        INGREDIENT_END_TOKEN,
        INGREDIENT_NEXT_TOKEN,
        INSTRUCTION_START,
        INSTRUCTION_END,
        INSTRUCTION_NEXT
    ]
}

def raw_recipe_to_text_tokens(raw_recipe:pd.Series) -> str:
    tokenized_text = f'{RECIPE_START_TOKEN} {INPUT_START_TOKEN} ' + \
        f' {INPUT_NEXT_TOKEN} '.join(raw_recipe['Ingredients']) + \
        f' {INPUT_END_TOKEN} {INGREDIENT_START_TOKEN} ' + \
        f' {INGREDIENT_NEXT_TOKEN} '.join(raw_recipe['Ingredients']) + \
        f' {INGREDIENT_END_TOKEN} {INSTRUCTION_START}' + \
        f' {INSTRUCTION_NEXT} '.join(raw_recipe['Instructions']) + \
        f' {INSTRUCTION_END} {RECIPE_END_TOKEN}'
    
    return tokenized_text
    
def raw_recipes_to_text_tokens_file(raw_recipes:pd.DataFrame, filename:str = 'raw_recipes_tokenized.txt', verbose:bool = False) -> None:
    if verbose: print('Writing recipes tokens to', filename)
    with open(filename, 'w') as tokens_file:
        for index, recipe_row in raw_recipes.iterrows():
            if verbose and index % 20_000 == 0: print('current index:', index)

            try:
                recipe_tokens = raw_recipe_to_text_tokens(recipe_row)
                tokens_file.write(f'{recipe_tokens}\n')
            except:
                if verbose: print(f'exception on recipe at index {index}')

def text_tokens_files_to_dataset(dataset_filename:str, tokenizer:Any, train_filename:str = None, test_filename:str = None, end_token:str = RECIPE_END_TOKEN) -> None:
    with h5py.File(dataset_filename, 'w') as dataset_file:
        files_partitions = []
        if train_filename is not None:
            files_partitions.append((train_filename, 'train'))
        if test_filename is not None:
            files_partitions.append((test_filename, 'test'))

        end_token_index = tokenizer.convert_tokens_to_ids([end_token])[0]

        ## Iterate partitions
        for filename, partition in files_partitions:
            print(f'tokenizing {partition} set')
            complete_examples = []

            ## Read partition file
            with open(filename, 'r') as data_file:
                last_tokens = []

                for line in data_file:
                    ## Turn text token in numeric token
                    line_tokens = tokenizer.tokenize(line)
                    if len(line_tokens) > 1024: # skip too long examples
                        continue

                    line_tokens_indices = tokenizer.convert_tokens_to_ids(line_tokens)

                    ## Concatenate too short examples
                    if (len(last_tokens) + len(line_tokens_indices)) <= 1024:
                        last_tokens += line_tokens_indices
                    else:
                        remaining_tokens = 1024 - len(last_tokens)
                        last_tokens.extend([end_token_index]*remaining_tokens)
                        complete_examples.append(last_tokens)
                        last_tokens = line_tokens_indices

            ## Add partition to dataset
            complete_examples_matrix = np.matrix(complete_examples)
            print(partition, 'size:', complete_examples_matrix.shape)
            dataset_file.create_dataset(partition, data=complete_examples_matrix)
