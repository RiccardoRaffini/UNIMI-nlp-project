import sys
sys.path.append('.')

import numpy as np
import os
import random
import statistics
from argparse import ArgumentParser

from algorithms.crossovers import *
from algorithms.evolutionary_algorithms import RecipeEvolutionaryAlgorithm
from algorithms.fitness_evaluators import *
from algorithms.mutations import *
from algorithms.population_selectors import *
from algorithms.utils import generate_recipes_population
from commons.recipes import Action, Ingredient, Tool, RecipeMatrices
from commons.text import RecipeGraphToText

arguments_parser = ArgumentParser()
arguments_parser.add_argument('--relation_matrices_name', required=True, type=str, default=None)
arguments_parser.add_argument('--prompt', required=False, type=str, default=None)
arguments_parser.add_argument('--initial_population_size', required=False, type=str, default=10)
arguments_parser.add_argument('--epochs', required=False, type=int, default=10)
arguments_parser.add_argument('--save_initial_population', action='store_true')
arguments_parser.add_argument('--save_final_population', action='store_true')
arguments_parser.add_argument('--output_directory', required=False, type=str, default='evolution_run')
arguments_parser.add_argument('--seed', required=False, type=int, default=1234)

class RandomTypeNodeIndexSelector:
    def __init__(self, node_type:type):
        self._selection_type = node_type

    def select_index(self, recipe_individual:RecipeIndividual) -> int:
        type_node_indices = []
        for node_index in recipe_individual.nodes():
            node = recipe_individual.get_node(node_index)
            if type(node['object']) == self._selection_type:
                type_node_indices.append(node_index)

        if not type_node_indices:
            return None

        selected_index = random.choice(type_node_indices)
        return selected_index

    def __call__(self, recipe_individual:RecipeIndividual) -> int:
        selected_index = self.select_index(recipe_individual)
        return selected_index
    
class RandomActionGroupNodeIndexSelector:
    def __init__(self, group:str):
        self._selection_group = group

    def select_index(self, recipe_individual:RecipeIndividual) -> int:
        group_node_indices = []
        for node_index in recipe_individual.nodes():
            node = recipe_individual.get_node(node_index)
            if type(node['object']) == Action and node['object'].group == self._selection_group:
                group_node_indices.append(node_index)

        if not group_node_indices:
            return None

        selected_index = random.choice(group_node_indices)
        return selected_index
    
    def __call__(self, recipe_individual:RecipeIndividual) -> int:
        selected_index = self.select_index(recipe_individual)
        return selected_index

def main():
    ## Argumnets
    arguments = arguments_parser.parse_args()
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)

    ## Load recipe relation matrices
    print(f'Loading {arguments.relation_matrices_name} matrices...')
    recipe_matrices = RecipeMatrices.load(arguments.relation_matrices_name)
    recipe_matrices.compile()

    ## Check user input
    recipe_prompt = arguments.prompt
    if recipe_prompt is None:
        print('No prompt provided in arguments')
        recipe_prompt = input('Provide new prompt now as a comma-separated list of ingredients:')
    recipe_ingredients = recipe_prompt.split(',')

    ## Define evolutionary algorithm
    recipes_evaluator = RecipeScoreEvaluator(recipe_matrices)
    algorithm_configuration = {
        'fitness evaluator': recipes_evaluator,
        'population selector': ElitismSelector(recipes_evaluator),
        'mutation methods': [
            SplitMixNodeMutation(recipe_matrices),
            DeleteActionNodeMutation(recipe_matrices),
            InsertActionNodeMutation(recipe_matrices),
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

    ## Generate initial population
    print('Generating initial recipes...')
    initial_recipe_population = generate_recipes_population(
        recipe_matrices=recipe_matrices,
        individuals_number=arguments.initial_population_size,
        required_ingredients=recipe_ingredients,
        additional_ingredients=None,
        ingredients_number=len(recipe_ingredients),
        min_addition_size=0,
        max_addition_size=4
    )

    if arguments.save_initial_population:
        os.makedirs(f'{arguments.output_directory}/initial_population', exist_ok=True)
        for i in range(initial_recipe_population.size()):
            individual = initial_recipe_population.individual_at(i)
            individual.to_gviz(f'{arguments.output_directory}/initial_population/individual_{i}.png')

    ## Run evolutionary algorithm
    print('Running algorithm...')
    recipe_evolutionary_algorithm.set_population(initial_recipe_population)
    epochs_data = recipe_evolutionary_algorithm.run(arguments.epochs)
    final_recipe_population = recipe_evolutionary_algorithm.get_population()

    for i in range(arguments.epochs):
        epoch_data = epochs_data[i]
        print(f'epoch {i} mean fitness: {statistics.mean(epoch_data["final_population_score"])}, individuals fitness: {epoch_data["final_population_score"]}')

    if arguments.save_final_population:
        os.makedirs(f'{arguments.output_directory}/final_population', exist_ok=True)
        for i in range(final_recipe_population.size()):
            individual = final_recipe_population.individual_at(i)
            individual.to_gviz(f'{arguments.output_directory}/final_population/individual_{i}.png')

    ## Get best individual
    last_epoch_fitness = epochs_data[arguments.epochs - 1]['final_population_score']
    best_individual_index = np.argmax(last_epoch_fitness)
    best_individual:RecipeIndividual = final_recipe_population.individual_at(best_individual_index)
    best_individual.to_gviz(f'{arguments.output_directory}/best_recipe.png')

    ## Print best individual
    print('Best recipe:')
    output_processor = RecipeGraphToText()
    ingredients = output_processor.ingredients(best_individual)
    instructions = output_processor.instructions(best_individual)
    markdown = output_processor.markdown(best_individual)
    print(markdown)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`*.')
    warnings.filterwarnings('ignore', '1Torch was not compiled with flash attention.')

    main()