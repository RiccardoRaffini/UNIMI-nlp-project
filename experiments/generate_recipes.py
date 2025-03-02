import sys
sys.path.append('.')

import json
import numpy as np
import random
import torch
from argparse import ArgumentParser
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List

from algorithms.crossovers import *
from algorithms.evolutionary_algorithms import RecipeEvolutionaryAlgorithm
from algorithms.fitness_evaluators import *
from algorithms.generate import RandomTypeNodeIndexSelector, RandomActionGroupNodeIndexSelector
from algorithms.mutations import *
from algorithms.population_selectors import *
from algorithms.utils import generate_recipes_population
from commons.data import load_foodcom_review_dataset
from commons.recipes import Action, Ingredient, Tool, RecipeMatrices
from commons.text import TokensSequenceToText, RecipeGraphToText
from llms.tokenizer import RECIPES_ADDITIONAL_TOKENS
from llms.generate import generate, logits_filtering

## Model arguments
model_arguments_parser = ArgumentParser()
model_arguments_parser.add_argument('--model_path', required=False, type=str, default='models/gpt2-finetuned-recipe-generation/checkpoint-final')
model_arguments_parser.add_argument('--length', required=False, type=int, default=1000)
model_arguments_parser.add_argument('--temperature', required=False, type=float, default=1.0)
model_arguments_parser.add_argument('--top_k', required=False, type=int, default=0)
model_arguments_parser.add_argument('--top_p', required=False, type=float, default=0.9)
model_arguments_parser.add_argument('--no_cuda', required=False, type=bool, default=False)
model_arguments_parser.add_argument('--seed', required=False, type=int, default=1234)

## Algorithm arguments
algorithm_arguments_parser = ArgumentParser()
algorithm_arguments_parser.add_argument('--relation_matrices_name', required=False, type=str, default='models/evolutionary_algorithms/global_recipes_matrices_60k')
algorithm_arguments_parser.add_argument('--initial_population_size', required=False, type=str, default=10)
algorithm_arguments_parser.add_argument('--epochs', required=False, type=int, default=10)
algorithm_arguments_parser.add_argument('--seed', required=False, type=int, default=1234)

## Common arguments
common_arguments_parser = ArgumentParser()
common_arguments_parser.add_argument('--output_filename', required=False, type=str, default='experiments/generated_recipes.json')
common_arguments_parser.add_argument('--save_interval', required=False, type=int, default=10)

def main():

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
    reference_dataset = available_recipes.sample(n=100, random_state=seed)
    reference_dataset = reference_dataset.reset_index()

    ## Define language model
    print('Model definition...')

    model_arguments = model_arguments_parser.parse_args()
    model_arguments.device = 'cuda' if torch.cuda.is_available() and not model_arguments.no_cuda else 'cpu'
    torch.manual_seed(model_arguments.seed)

    model_tokenizer = GPT2Tokenizer.from_pretrained(model_arguments.model_path, do_lower_case=True)
    model_tokenizer.add_special_tokens(RECIPES_ADDITIONAL_TOKENS)
    model = GPT2LMHeadModel.from_pretrained(model_arguments.model_path)
    model.resize_token_embeddings(len(model_tokenizer))
    model.to(model_arguments.device)
    model.eval()

    if model.config.max_position_embeddings < model_arguments.length:
        model_arguments.length = model.config.max_position_embeddings

    def tokenize_input(input:List[str]) -> str:
        tokenized = '<START_RECIPE> <INPUT_START> ' + \
        ' <INPUT_NEXT> '.join(input) + \
        ' <INPUT_END>'
        return tokenized

    ## Define evolutionary algorithm
    print('Algorithm definition...')

    algorithm_arguments = algorithm_arguments_parser.parse_args()
    random.seed(algorithm_arguments.seed)
    np.random.seed(algorithm_arguments.seed)

    print(f'loading {algorithm_arguments.relation_matrices_name} matrices...')
    recipe_matrices = RecipeMatrices.load(algorithm_arguments.relation_matrices_name)
    recipe_matrices.compile()

    recipes_evaluator = RecipeScoreEvaluator(recipe_matrices)
    algorithm_configuration = {
        'fitness evaluator': recipes_evaluator,
        'population selector': ElitismSelector(recipes_evaluator),
        'mutation methods': [
            SplitMixNodeMutation(recipe_matrices),
            DeleteActionNodeMutation(recipe_matrices),
            InsertActionNodeMutation(recipe_matrices, insert_mixing_actions=True),
            ReplaceActionNodeMutation(recipe_matrices),
            AddIngredientToActionNodeMutation(recipe_matrices),
            InsertActionToIngredientNodeMutation(recipe_matrices),
            AddToolNodeMutation(recipe_matrices)
        ],
        'mutation probabilities': [0.2, 0.05, 0.2, 0.2, 0.2, 0.05, 0.1],
        'mutation nodes selectors': [
            RandomActionGroupNodeIndexSelector('mix'),
            RandomTypeNodeIndexSelector(Action),
            RandomTypeNodeIndexSelector(Action),
            RandomTypeNodeIndexSelector(Action),
            RandomTypeNodeIndexSelector(Action),
            RandomTypeNodeIndexSelector(Ingredient),
            RandomTypeNodeIndexSelector(Tool),
        ]
    }

    recipe_evolutionary_algorithm = RecipeEvolutionaryAlgorithm(algorithm_configuration)

    ## Iterate reference recipes
    common_arguments = common_arguments_parser.parse_args()
    generation_data = dict()
    generated_recipes = []
    save_interval = common_arguments.save_interval

    model_output_processor = TokensSequenceToText()
    lemmer = WordNetLemmatizer()
    algorithm_output_processor = RecipeGraphToText()

    for index, recipe_row in tqdm(reference_dataset.iterrows(), 'Reference recipes'):
        print(f'Recipe {index}')

        reference_ingredients = recipe_row['Ingredients']
        reference_instructions = recipe_row['Instructions']
        print('Reference recipe:')
        print(reference_ingredients)
        print(reference_instructions)

        ## Generate model recipe
        print('Model recipe generation...')

        input_text_tokens = tokenize_input(reference_ingredients)
        input_tokens = model_tokenizer.encode(input_text_tokens)

        repeat_generation = True
        repeat_count = 1
        while repeat_generation:
            repeat_generation = False

            try:
                output = generate(model_arguments, model, model_tokenizer, input_tokens)
                output = output[0, len(input_tokens):].tolist()
                text_output = model_tokenizer.decode(output, clean_up_tokenization_spaces = True)

                if '<RECIPE_END>' not in text_output:
                    raise ValueError('Recipe generation failed, too long recipe')
                
            except ValueError as e:
                repeat_generation = True
                repeat_count += 1

                if repeat_count == 5:
                    raise e
                
                print('Repeating recipe generation')
        
        model_ingredients = model_output_processor.ingredients(text_output)
        model_instructions = model_output_processor.instructions(text_output)
        print('Model recipe:')
        print(model_ingredients)
        print(model_instructions)

        ## Generate algorithm recipe
        print('Algorithm recipe generation...')

        base_ingredients = [lemmer.lemmatize(ingredient) for ingredient in reference_ingredients]
        initial_recipe_population = generate_recipes_population(
            recipe_matrices=recipe_matrices,
            individuals_number=algorithm_arguments.initial_population_size,
            required_ingredients=base_ingredients,
            additional_ingredients=None,
            ingredients_number=len(reference_ingredients),
            min_addition_size=0,
            max_addition_size=0
        )

        recipe_evolutionary_algorithm.set_population(initial_recipe_population)
        epochs_data = recipe_evolutionary_algorithm.run(algorithm_arguments.epochs)
        final_recipe_population = recipe_evolutionary_algorithm.get_population()
        
        last_epoch_fitness = epochs_data[algorithm_arguments.epochs - 1]['final_population_score']
        best_individual_index = np.argmax(last_epoch_fitness)
        best_individual:RecipeIndividual = final_recipe_population.individual_at(best_individual_index)
        best_individual.simplify_graph()

        algorithm_ingredients = algorithm_output_processor.ingredients(best_individual)
        algorithm_instructions = algorithm_output_processor.instructions(best_individual)
        print('Algorithm recipe:')
        print(algorithm_ingredients)
        print(algorithm_instructions)

        ## Save triplet of original-model-algorithm
        generated_recipes.append({
            'reference': {'index': index, 'ingredients': reference_ingredients, 'instructions': reference_instructions},
            'model': {'ingredients': model_ingredients, 'instructions': model_instructions},
            'algorithm': {'ingredients': algorithm_ingredients, 'instructions': algorithm_instructions}
        })

        ## Save generated recipes
        if index % save_interval == 0:
            print('Saving generated recipes...')
            with open(common_arguments.output_filename, 'w') as save_file:
                json.dump(generated_recipes, save_file, indent='\t')

    ## Final save
    print('Saving generated recipes...')
    with open(common_arguments.output_filename, 'w') as save_file:
        json.dump(generated_recipes, save_file, indent='\t')

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`*.')
    warnings.filterwarnings('ignore', '1Torch was not compiled with flash attention.')

    main()