import sys
sys.path.append('.')

import pandas as pd
from datasets import load_dataset, ClassLabel

TRAIN_LOAD_PATH = 'datasets/ner_datasets/instructions-train.json'
TEST_LOAD_PATH = 'datasets/ner_datasets/instructions-test.json'
NER_TAGS_MAP = ClassLabel(num_classes=7, names=['O', 'B-FOOD', 'I-FOOD', 'B-TOOL', 'I-TOOL', 'B-ACTION', 'I-ACTION'])
#                                                0    1         2         3         4         5           6
TRAIN_LABELS_PATH = 'datasets/ner_datasets/instructions-labels-train.json'
TEST_LABELS_PATH = 'datasets/ner_datasets/instructions-labels-test.json'

AUTO_SAVE_INTERVAL = 25

if __name__ == '__main__':
    ## Load instructions dataset
    data_files = {'train': TRAIN_LOAD_PATH, 'test': TEST_LOAD_PATH}
    instructions_dataset = load_dataset('json', data_files=data_files)

    print('Instruction datasets:')
    print(instructions_dataset)

    ## Load labels datasets
    train_labels = pd.read_json(TRAIN_LABELS_PATH)
    test_labels = pd.read_json(TEST_LABELS_PATH)

    ## Get current annotation status
    train_starting_index = len(train_labels)
    test_starting_index = len(test_labels)

    if train_starting_index == 0:
        train_labels['labels'] = []

    if test_starting_index == 0:
        test_labels['labels'] = []

    train_end_index = instructions_dataset['train'].num_rows
    test_end_index = instructions_dataset['test'].num_rows

    ## Continue train annotation
    while train_starting_index <= train_end_index:
        example = instructions_dataset['train'][train_starting_index]
        tokens_number = len(example['tokens'])

        print(f'example {train_starting_index}/{train_end_index}')
        print(example['instructions'])
        print(example['tokens'])

        print(f'provide {tokens_number} labels:')
        labels_string = input()

        if labels_string == 'stop':
            break

        elif len(labels_string) != tokens_number:
            print('labels number and tokens number mismatch, repeat annotation')
            continue

        else:
            labels = NER_TAGS_MAP.int2str([int(s) for s in labels_string])
            train_labels.loc[train_starting_index] = [labels]

            train_starting_index += 1

        if train_starting_index % AUTO_SAVE_INTERVAL == 0:
            print('Saving train labels')
            train_labels.to_json(TRAIN_LABELS_PATH, orient='records')

    print('Saving train labels')
    train_labels.to_json(TRAIN_LABELS_PATH, orient='records')

    ## Continue test annotation
    while test_starting_index <= test_end_index:
        example = instructions_dataset['test'][test_starting_index]
        tokens_number = len(example['tokens'])

        print(f'example {test_starting_index}/{test_end_index}')
        print(example['instructions'])
        print(example['tokens'])

        print(f'provide {tokens_number} labels:')
        labels_string = input()

        if labels_string == 'stop':
            break

        elif len(labels_string) != tokens_number:
            print('labels number and tokens number mismatch, repeat annotation')
            continue

        else:
            labels = NER_TAGS_MAP.int2str([int(s) for s in labels_string])
            test_labels.loc[test_starting_index] = [labels]

            test_starting_index += 1

        if test_starting_index % AUTO_SAVE_INTERVAL == 0:
            print('Saving test labels')
            test_labels.to_json(TEST_LABELS_PATH, orient='records')

    print('Saving test labels')
    test_labels.to_json(TEST_LABELS_PATH, orient='records')
