import sys
sys.path.append('.')

from datasets import load_dataset, concatenate_datasets, ClassLabel, Sequence

TRAIN_DATA_PATH = 'datasets/ner_datasets/instructions-train.json'
TEST_DATA_PATH = 'datasets/ner_datasets/instructions-test.json'

TRAIN_LABELS_PATH = 'datasets/ner_datasets/instructions-labels-train.json'
TEST_LABELS_PATH = 'datasets/ner_datasets/instructions-labels-test.json'
NER_TAGS_MAP = ClassLabel(num_classes=7, names=['O', 'B-FOOD', 'I-FOOD', 'B-TOOL', 'I-TOOL', 'B-ACTION', 'I-ACTION'])

TRAIN_SAVE_PATH = 'datasets/ner_datasets/food-train.json'
TEST_SAVE_PATH = 'datasets/ner_datasets/food-test.json'

if __name__ == '__main__':
    ## Load datasets
    data_files = {'train': TRAIN_DATA_PATH, 'test': TEST_DATA_PATH}
    instructions_dataset = load_dataset('json', data_files=data_files)

    print('Initial instructions datasets:')
    print(instructions_dataset)
    print('Examples:')
    print(instructions_dataset['train'][0])
    print(instructions_dataset['test'][0])

    data_files = {'train': TRAIN_LABELS_PATH, 'test': TEST_LABELS_PATH}
    labels_dataset = load_dataset('json', data_files=data_files)

    print('Initial labels datasets:')
    print(labels_dataset)
    print('Examples:')
    print(labels_dataset['train'][0])
    print(labels_dataset['test'][0])

    ## Select data examples
    instructions_dataset['train'] = instructions_dataset['train'].select(range(labels_dataset['train'].num_rows))
    instructions_dataset['test'] = instructions_dataset['test'].select(range(labels_dataset['test'].num_rows))

    print('Sliced instructions datasets:')
    print(instructions_dataset)

    ## Merge data and labels
    def map_labels(example):
        return {'labels': [NER_TAGS_MAP.str2int(label) for label in example['labels']]}
    
    labels_dataset = labels_dataset.map(map_labels)

    instructions_dataset['train'] = concatenate_datasets([instructions_dataset['train'], labels_dataset['train']], axis=1)
    instructions_dataset['train'] = instructions_dataset['train'].rename_column('labels', 'ner_tags').cast_column('ner_tags', Sequence(NER_TAGS_MAP))
    instructions_dataset['test'] = concatenate_datasets([instructions_dataset['test'], labels_dataset['test']], axis=1)
    instructions_dataset['test'] = instructions_dataset['test'].rename_column('labels', 'ner_tags').cast_column('ner_tags', Sequence(NER_TAGS_MAP))

    print('Mrged instructions datasets:')
    print(instructions_dataset)
    print('Examples:')
    print(instructions_dataset['train'][0])
    print(instructions_dataset['test'][0])

    ## Save complete dataset
    instructions_dataset['train'].to_json(TRAIN_SAVE_PATH)
    instructions_dataset['test'].to_json(TEST_SAVE_PATH)
