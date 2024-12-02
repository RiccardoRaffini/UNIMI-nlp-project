import json
import numpy as np
import os
import pandas as pd
import random
import re
import spacy
import warnings
from collections import defaultdict
from spacy.matcher import PhraseMatcher
from spacy.tokens import DocBin, Span, Doc
from spacy.training.example import Example
from tqdm import tqdm
from typing import List, Tuple

TESTESET_LOCATION = 'datasets/ner_datasets/TASTEset.csv'
FOOD_SENTENCE_TEMPLATE_LOCATION = 'datasets/ner_datasets/food_sentence_template.json'

def load_food_data(dataset_filename:str=TESTESET_LOCATION) -> pd.DataFrame:
    ## Loading dataframe
    food_dataframe = pd.read_csv(dataset_filename)

    ## Parsing entities
    ingredients_entities = []
    for index in food_dataframe.index:
        loaded_entities = json.loads(food_dataframe.at[index, 'ingredients_entities'])
        entities = []

        for entity in loaded_entities:
            span = eval(entity['span'])
            start = span[0][0]
            end = span[0][1]
            entities.append((start, end, entity['type']))

        ingredients_entities.append({'entities': entities})

    food_dataframe['ingredients_entities'] = ingredients_entities

    return food_dataframe

def extract_foods(dataframe:pd.DataFrame, food_label:str='FOOD') -> List[str]:
    foods = []

    for text, annotations in dataframe.itertuples(index=False, name=None):
        for start, end, label in annotations['entities']:
            if label == food_label:
                foods.append(text[start:end])

    return foods

def load_food_sentence_templates(templates_filename:str=FOOD_SENTENCE_TEMPLATE_LOCATION) -> List[str]:
    with open(templates_filename, 'r') as template_file:
        sentence_templates = json.load(template_file)

    return sentence_templates

def filter_spans(spans:List[Span]) -> List[Span]:
    last_end = -1
    last_length = 0
    valid_spans = []

    for i in range(len(spans)):
        length = spans[i].end_char - spans[i].start_char

        if spans[i].start_char >= last_end:
            valid_spans.append(spans[i])
            last_end = spans[i].end_char
            last_length = length

        elif last_length < length:
            valid_spans[-1] = spans[i]
            last_end = spans[i].end_char
            last_length = length
    
    return valid_spans

def create_document_bin(data:List[Doc]) -> DocBin:
    document_bin = DocBin(store_user_data=True)

    for document in tqdm(data):
        document_bin.add(document)

    return document_bin

def main():
    ## Load foods
    food_data = load_food_data()
    foods = extract_foods(food_data)

    print('Food examples:')
    print(foods[:10])

    ## Load sentences
    sentence_templates = load_food_sentence_templates()
    sentence_templates_number = len(sentence_templates)
    pattern_to_replace = '{}'

    print('Sentence examples:')
    print(sentence_templates[random.randint(0, sentence_templates_number-1)])
    print(sentence_templates[random.randint(0, sentence_templates_number-1)])
    print(sentence_templates[random.randint(0, sentence_templates_number-1)])

    ## Generate sentences
    ner_model = spacy.blank('en')
    DATA = []

    remaining_foods = len(foods)
    random.shuffle(foods)

    while remaining_foods > 0:
        sentence = sentence_templates[random.randint(0, sentence_templates_number-1)]
        replace_spots = re.findall(pattern_to_replace, sentence)

        sentence_foods = set()
        for spot in replace_spots:
            remaining_foods -= 1
            food = foods[remaining_foods]

            sentence = sentence.replace(spot, food, 1)
            sentence_foods.add(food)

        sentence_document = ner_model.make_doc(sentence)
        food_patterns = [ner_model(food) for food in sentence_foods]

        #print(sentence_document.text)

        matcher = PhraseMatcher(ner_model.vocab)
        matcher.add('FOOD', None, *food_patterns)
        sentence_matches = matcher(sentence_document)

        spans = []
        for match_id, start, end in sentence_matches:
            span = sentence_document[start:end]
            #print(f'\t{span.text} {span.label_} {span.start_char} {span.end_char} {sentence_document.text[span.start_char:span.end_char] in sentence_foods}')

            span = Span(sentence_document, start, end, label=match_id)
            spans.append(span)

        spans = filter_spans(spans)
        sentence_document.ents = spans

        #sentence_examples = Example(sentence_document, sentence_document)
        #DATA.append(sentence_examples)
        DATA.append(sentence_document)

    ## Saving data
    data_size = len(DATA)
    split = 0.8
    split_index = int(round(split * data_size))

    train_data = DATA[:split_index]
    train_document_bin = create_document_bin(train_data)
    train_bin_path = os.path.join(os.path.dirname(__file__), 'train_data.spacy')
    train_document_bin.to_disk(train_bin_path)
    print(f'Saved train documents data to {train_bin_path}')

    test_data = DATA[split_index:]
    test_document_bin = create_document_bin(test_data)
    test_bin_path = os.path.join(os.path.dirname(__file__), 'test_data.spacy')
    test_document_bin.to_disk(test_bin_path)
    print(f'Saved test documents data to {test_bin_path}')

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    main()