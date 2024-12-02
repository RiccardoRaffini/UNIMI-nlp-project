import sys
sys.path.append('.')

import html
import numpy as np
import spacy
from datasets import Dataset, DatasetDict
from nltk.tokenize import word_tokenize

from commons.data import load_recipes3k_dataset

SEED = 1234
TRAIN_SIZE = 0.8
TRAIN_SAVE_PATH = 'datasets/ner_datasets/instructions-train.json'
TEST_SAVE_PATH = 'datasets/ner_datasets/instructions-test.json'

if __name__ == '__main__':
    ## Load base dataframe
    recipes_dataframe = load_recipes3k_dataset()
    recipes_dataframe_size = len(recipes_dataframe)

    ## Split dataframe in train and test partitions
    random_generator = np.random.default_rng(SEED)
    split_mask = random_generator.random(recipes_dataframe_size) < TRAIN_SIZE

    train_recipes_dataframe = recipes_dataframe[split_mask]
    test_recipes_dataframe = recipes_dataframe[~split_mask]

    ## Load partitions as dataset
    instructions_dataset = DatasetDict({
        'train': Dataset.from_pandas(train_recipes_dataframe['Instructions'].explode().to_frame()),
        'test': Dataset.from_pandas(test_recipes_dataframe['Instructions'].explode().to_frame())
    })

    print('Initial instructions datasets:')
    print(instructions_dataset)
    print('Examples:')
    print(instructions_dataset['train'][0])
    print(instructions_dataset['test'][0])

    ## Process dataset
    def lowercase_instruction(example):
        return {'Instructions': example['Instructions'].lower()}
    
    def escape_html(example):
        return {'Instructions': html.unescape(example['Instructions'])}

    processor = spacy.load('en_core_web_sm')
    def split_sentences(example):
        return {'Instructions': [sentence.text for sentence in processor(example['Instructions']).sents]}
    
    def expand_instructions(examples):
        return {'Instructions': [sentence for instruction in examples['Instructions'] for sentence in instruction]}
    
    def repeat_ids(example):
        return {'Id': [example['Id'] for _ in example['Instructions']]}
    
    def expand_instructions_ids(examples):
        return {
            'Id': [id for id_list in examples['Id'] for id in id_list],
            'Instructions': [sentence for instruction in examples['Instructions'] for sentence in instruction]
        }
    
    # def split_sentences(examples):
    #     return {'Instructions': [sentence.text for instruction in examples['Instructions'] for sentence in processor(instruction).sents]}

    def word_tokenize_instruction(example):
        return {'tokens': word_tokenize(example['Instructions'])}
    
    instructions_dataset = (
        instructions_dataset
        #.remove_columns('Id')
        .map(escape_html)
        .map(lowercase_instruction)
        .map(split_sentences)
        #.map(expand_instructions, batched=True)
        #.map(split_sentences, batched=True)
        .map(repeat_ids)
        .map(expand_instructions_ids, batched=True)
        .map(word_tokenize_instruction)
        .rename_columns({'Id': 'id', 'Instructions': 'instructions'})
    )

    print('Processed instructions datasets:')
    print(instructions_dataset)
    print('Examples:')
    print(instructions_dataset['train'][0])
    print(instructions_dataset['test'][0])

    ## Save datasets
    instructions_dataset['train'].to_json(TRAIN_SAVE_PATH)
    instructions_dataset['test'].to_json(TEST_SAVE_PATH)
