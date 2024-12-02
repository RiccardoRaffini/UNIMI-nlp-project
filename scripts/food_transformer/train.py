import sys
sys.path.append('.')

import evaluate
import numpy as np
import torch
from datasets import load_dataset, ClassLabel, Sequence
from transformers import (
    AutoTokenizer, DataCollatorForTokenClassification,
    AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline
)

TRAIN_DATA_PATH = 'datasets/ner_datasets/food-train.json'
TEST_DATA_PATH = 'datasets/ner_datasets/food-test.json'
NER_TAGS_MAP = ClassLabel(num_classes=7, names=['O', 'B-FOOD', 'I-FOOD', 'B-TOOL', 'I-TOOL', 'B-ACTION', 'I-ACTION'])
EPOCHS = 10

if __name__ == '__main__':
    torch.cuda.set_device(0)

    ## Load datasets
    data_files = {'train': TRAIN_DATA_PATH, 'test': TEST_DATA_PATH}
    cooking_dataset = load_dataset('json', data_files=data_files)
    cooking_dataset['train'] = cooking_dataset['train'].cast_column('ner_tags', Sequence(NER_TAGS_MAP))
    cooking_dataset['test'] = cooking_dataset['test'].cast_column('ner_tags', Sequence(NER_TAGS_MAP))

    print('Cooking datasets:')
    print(cooking_dataset)
    print('Examples:')
    print(cooking_dataset['train'][0])
    print(cooking_dataset['test'][0])

    ## Preprocess data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
    def align_labels_with_tokens(labels, word_ids):
        new_labels = []

        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                new_labels.append(-100)
            else:
                label = labels[word_id]
                # change label to I-XXX if it is B-XXX (according to IOB)
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
        all_labels = examples['ner_tags']

        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs['labels'] = new_labels
        return tokenized_inputs
    
    tokenized_datasets = cooking_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=cooking_dataset['train'].column_names,
    )

    print('Processed datasets:')
    print(tokenized_datasets)
    print('Examples:')
    print(tokenized_datasets['train'][0])
    print(tokenized_datasets['test'][0])

    ## Define evaluation metrics
    metric = evaluate.load('seqeval')
    label_names = cooking_dataset['train'].features['ner_tags'].feature.names

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            'precision': all_metrics['overall_precision'],
            'recall': all_metrics['overall_recall'],
            'f1': all_metrics['overall_f1'],
            'accuracy': all_metrics['overall_accuracy'],
        }
    
    ## Defining model
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    device = torch.device('cuda')
    model = AutoModelForTokenClassification.from_pretrained(
        'bert-base-uncased',
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    args = TrainingArguments(
        'models/bert-finetuned-food-ner',
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        push_to_hub=False,
    )

    ## Training model
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    trainer.train()

    ## Save model
    trainer.save_model('models/bert-finetuned-food-ner')

    ## Testing model
    model_path = 'models/bert-finetuned-food-ner'
    token_classifier = pipeline(
        "token-classification", model=model_path, aggregation_strategy="simple"
    )

    print(cooking_dataset['test'][0]['instructions'])
    print(token_classifier(cooking_dataset['test'][0]['instructions']))
