import sys
sys.path.append('.')

import json
import numpy as np
import pandas as pd
import random
from argparse import ArgumentParser
from scipy.spatial import distance
from tqdm import tqdm
from typing import List

from commons.data import load_foodcom_review_dataset
from commons.scoring import create_tf_idf_vector_space, tf_idf_vector, knn_vectors, average_score, classification
from commons.utils import dataframe_information
from experiments.datasets_information import plot_bar_categories, plot_box_instructions_comparison, plot_box_ingredients_comparison, plot_box_generic_comparison

arguments_parser = ArgumentParser()
arguments_parser.add_argument('--comparison_filename', required=False, type=str, default='experiments/generated_recipes.json')
arguments_parser.add_argument('--k', required=False, type=int, default=5)

def main():
    arguments = arguments_parser.parse_args()

    ## Reference dataset
    print('Reference dataset definition...')
    seed = 1234

    recipes_dataset = load_foodcom_review_dataset()
    train_dataset = recipes_dataset.sample(n=60_000, random_state=seed)

    available_recipes = recipes_dataset.drop(train_dataset.index, axis=0)                       # remove recipes used in training
    available_recipes = available_recipes[available_recipes['Instructions'].str.len()  >= 2]    # filter by instructions number
    available_recipes = available_recipes[available_recipes['Instructions'].str.len()  <= 12]   # filter by instructions number
    available_recipes = available_recipes[available_recipes['Ingredients'].str.len() >= 2]      # filter by ingrdients number
    available_recipes = available_recipes[available_recipes['Ingredients'].str.len() <= 13]     # filter by ingrdients number
    reference_recipes = available_recipes.sample(n=100, random_state=seed)
    reference_recipes = reference_recipes.reset_index()

    print(dataframe_information(reference_recipes))

    ## Define reference vector space
    print('Vector space definition...')

    reference_documents = list(map(lambda instructions: '\n'.join(instructions).lower(), reference_recipes['Instructions']))
    documents_vector_space, terms_map = create_tf_idf_vector_space(reference_documents)
    documents_vector_space = documents_vector_space.transpose().to_numpy()

    print('Vector space:')
    print(documents_vector_space)

    ## Load recipes file
    filename = arguments.comparison_filename
    with open(filename, 'r') as data_file:
        documents_data = json.load(data_file)

    ## Iterate on triplets
    comparison_data = []
    columns = ['Ingredients', 'Instructions', 'Category', 'Rating', 'Distance']
    model_recipes = pd.DataFrame(columns=columns)
    algorithm_recipes = pd.DataFrame(columns=columns)

    print('Comparing recipes...')
    for documents_tuple in tqdm(documents_data):
        ## Reference document
        reference_document = documents_tuple['reference']
        reference_index = reference_document['index']
        reference_instructions = '\n'.join(map(lambda s: s.lower(), reference_document['instructions']))
        reference_vector = documents_vector_space[reference_index] # tf_idf_vector(reference_instructions, reference_documents).to_numpy()

        print(f'Comparing recipe at index {reference_index}')

        ## Model document
        model_document = documents_tuple['model']
        model_instructions = '\n'.join(map(lambda s: s.lower(), model_document['instructions']))
        model_vector = tf_idf_vector(model_instructions, reference_documents).to_numpy()

        ## Algorithm document
        algorithm_document = documents_tuple['algorithm']
        algorithm_instructions = '\n'.join(map(lambda s: s.lower(), algorithm_document['instructions']))
        algorithm_vector = tf_idf_vector(algorithm_instructions, reference_documents).to_numpy()

        print('vectors (non-zero indices):')
        print(reference_vector.nonzero()[0])#, reference_vector[reference_vector.nonzero()[0]])
        print(model_vector.nonzero()[0])#, model_vector[model_vector.nonzero()[0]])
        print(algorithm_vector.nonzero()[0])#, algorithm_vector[algorithm_vector.nonzero()[0]])
        print('reference vector intersections:')
        print('m:', set(reference_vector.nonzero()[0].tolist()).intersection(model_vector.nonzero()[0].tolist()))
        print('a:', set(reference_vector.nonzero()[0].tolist()).intersection(algorithm_vector.nonzero()[0].tolist()))

        ## Determine reference distances
        reference_model_distance = distance.cosine(reference_vector, model_vector)
        reference_algorithm_distance = distance.cosine(reference_vector, algorithm_vector)

        print('distances', reference_model_distance, reference_algorithm_distance)

        ## Determine nearest neighbors
        model_neighbors_indices, model_neighbors_vectors, model_neighbors_distances = knn_vectors(model_vector, documents_vector_space, arguments.k)
        algorithm_neighbors_indices, algorithm_neighbors_vectors, algorithm_neighbors_distances = knn_vectors(algorithm_vector, documents_vector_space, arguments.k)
        
        ## Compute scores of generated recipes (weighted avg of knn)
        model_neighbors_scores = [reference_recipes.iloc[index]['Rating'] for index in model_neighbors_indices]
        model_neighbors_weights = np.ones(arguments.k) - model_neighbors_distances
        model_score = average_score(model_neighbors_scores, model_neighbors_weights)

        algorithm_neighbors_scores = [reference_recipes.iloc[index]['Rating'] for index in algorithm_neighbors_indices]
        algorithm_neighbors_weights = np.ones(arguments.k) - algorithm_neighbors_distances
        algorithm_score = average_score(algorithm_neighbors_scores, algorithm_neighbors_weights)

        print('scores', model_score, algorithm_score)

        ## Category classification
        reference_category = reference_recipes.iloc[reference_index]['Category']
        model_neighbors_categories = [reference_recipes.iloc[index]['Category'] for index in model_neighbors_indices]
        model_category = classification(model_neighbors_categories)
        algorithm_neighbors_categories = [reference_recipes.iloc[index]['Category'] for index in algorithm_neighbors_indices]
        algorithm_category = classification(algorithm_neighbors_categories)

        print('categories', reference_category, model_category, algorithm_category)

        ## Count instructions number
        reference_instructions_number = len(reference_document['instructions'])
        model_instructions_number = len(model_document['instructions'])
        algorithm_instructions_number = len(algorithm_document['instructions'])

        print('instructions number', reference_instructions_number, model_instructions_number, algorithm_instructions_number)

        ## Count ingredients number
        reference_ingredients_number = len(reference_document['ingredients'])
        model_ingredients_number = len(model_document['ingredients'])
        algorithm_ingredients_number = len(algorithm_document['ingredients'])

        print('ingredients number', reference_ingredients_number, model_ingredients_number, algorithm_ingredients_number)

        ## Define model and algorithm datasets
        model_recipes.loc[reference_index] = [model_document['ingredients'], model_document['instructions'], model_category, model_score, reference_model_distance]
        algorithm_recipes.loc[reference_index] = [algorithm_document['ingredients'], algorithm_document['instructions'], algorithm_category, algorithm_score, reference_algorithm_distance]

    ## Compare results
    plot_bar_categories(model_recipes, ['skyblue', 'deepskyblue'])
    plot_bar_categories(algorithm_recipes, ['gold', 'goldenrod'])
    plot_box_instructions_comparison([reference_recipes, model_recipes, algorithm_recipes], ['Reference', 'Model', 'Algorithm'], False)
    plot_box_ingredients_comparison([reference_recipes, model_recipes, algorithm_recipes], ['Reference', 'Model', 'Algorithm'], False)
    plot_box_generic_comparison([model_recipes, algorithm_recipes], ['Model', 'Algorithm'], 'Distance', 'Reference distance', False)
    plot_box_generic_comparison([reference_recipes, model_recipes, algorithm_recipes], ['Reference', 'Model', 'Algorithm'], 'Rating', 'Rating', False)

if __name__ == '__main__':
    main()