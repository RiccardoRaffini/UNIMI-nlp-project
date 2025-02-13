import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from typing import Callable, List, Tuple

from algorithms.populations import RecipeIndividual, RecipePopulation
from commons.action_groups import inverse_groups, groups
from commons.matrices import MixedIngredientsMatrix, ActionsIngredientsMatrix
from commons.recipes import Recipe, RecipeGraph, RecipeMatrices, Ingredient, Action

def k_likely_mix_ingredients(mix_ingredients_matrix:MixedIngredientsMatrix, ingredient:str, k:int = 10) -> List[str]:
    ## Find top k likely mixed ingredients
    ingredient_index = mix_ingredients_matrix.label_to_index(ingredient, False)
    if ingredient_index == -1:
        return None
    
    ingredient_row = mix_ingredients_matrix.get_csr_matrix().getrow(ingredient_index)
    ingredient_row = ingredient_row[0, ingredient_row.nonzero()[1]]
    ingredient_row = mix_ingredients_matrix.get_full_matrix()[ingredient_index]
    top_k_indices = np.argpartition(ingredient_row, -k)[-k:]
    
    ## Get top k labels
    ingredients_labels = mix_ingredients_matrix.get_labels()
    top_k_ingredients = [ingredients_labels[index] for index in top_k_indices]

    return top_k_ingredients

def prepare_ingredient(group_actions_ingredients_matrix:ActionsIngredientsMatrix, ingredient:str, threshold:float = 0.5, epsilon:float = 0.01) -> bool:
    ingredient_index = group_actions_ingredients_matrix.label_to_column_index(ingredient, False)
    prepare_index = group_actions_ingredients_matrix.label_to_row_index('prepare', False)
    heat_index = group_actions_ingredients_matrix.label_to_row_index('heat', False)
    
    if ingredient_index == -1 or prepare_index == -1 or heat_index == -1:
        prepare_heat_ratio = 0
    else:
        prepare_value = group_actions_ingredients_matrix.get_csr_matrix().getrow(prepare_index)[0, ingredient_index]
        heat_value = group_actions_ingredients_matrix.get_csr_matrix().getrow(heat_index)[0, ingredient_index]
        prepare_heat_ratio = prepare_value / heat_value if heat_value else 0

    random_value = np.random.normal(0, epsilon)

    if prepare_heat_ratio + random_value < threshold:
        return False

    return True

def heat_ingredient(group_actions_ingredients_matrix:ActionsIngredientsMatrix, ingredient:str, threshold:float = 0.5, epsilon:float = 0.01) -> bool:
    ingredient_index = group_actions_ingredients_matrix.label_to_column_index(ingredient, False)
    prepare_index = group_actions_ingredients_matrix.label_to_row_index('prepare', False)
    heat_index = group_actions_ingredients_matrix.label_to_row_index('heat', False)
    
    if ingredient_index == -1 or prepare_index == -1 or heat_index == -1:
        prepare_heat_ratio = 0
    else:
        prepare_value = group_actions_ingredients_matrix.get_csr_matrix().getrow(prepare_index)[0, ingredient_index]
        heat_value = group_actions_ingredients_matrix.get_csr_matrix().getrow(heat_index)[0, ingredient_index]
        prepare_heat_ratio = prepare_value / heat_value if heat_value else 0

    random_value = np.random.normal(0, epsilon)

    if 1 - prepare_heat_ratio + random_value < threshold:
        return False

    return True

def generate_recipe_individual(
    recipe_matrices:RecipeMatrices,
    required_ingredients:List[str], additional_ingredients:List[str] = None,
    ingredients_number:int = 3,
    min_addition_size:int = 0, max_addition_size:int = 4
    ) -> RecipeIndividual:
    ## Constrains checks
    assert len(required_ingredients) > 0, 'at least one main ingredient must be specified for individual generation'

    ## Find likely mixed ingredients
    mixing_ingredients = dict()
    if additional_ingredients is None:
        additional_ingredients = []

    for ingredient in required_ingredients + additional_ingredients:
        possible_ingredients = k_likely_mix_ingredients(recipe_matrices.base_ingredients_base_ingredients, ingredient, k=max_addition_size)
        mixing_ingredients[ingredient] = possible_ingredients

    ## Extract random mixing ingredients
    for ingredient, possible_mixing_ingredients in mixing_ingredients.items():
        ingredients_number = random.randint(min_addition_size, max_addition_size+1)
        selected_ingredients = set(random.choices(possible_mixing_ingredients, k=ingredients_number))
        selected_ingredients.discard(ingredient)
        mixing_ingredients[ingredient] = selected_ingredients

    ## Create recipe individual
    recipe_individual = RecipeIndividual()

    ## Create mix nodes with main ingredients
    tree_roots = []
    ingredients_pool = required_ingredients.copy()
    remaining_number = ingredients_number - len(ingredients_pool)
    if remaining_number > 0:
        ingredients_pool.extend(random.choices(additional_ingredients, k=remaining_number))

    for main_ingredient in ingredients_pool:
        nodes_indices = []

        ingredient = Ingredient(main_ingredient)
        ingredient_index = recipe_individual.add_ingredient_node(ingredient)
        nodes_indices.append(ingredient_index)

        for mix_ingredient in mixing_ingredients[main_ingredient]:
            ingredient = Ingredient(mix_ingredient)
            ingredient_index = recipe_individual.add_ingredient_node(ingredient)
            nodes_indices.append(ingredient_index)

        mix_action = random.choice(inverse_groups['mix'])
        mix_action = Action(mix_action, 'mix')

        mix_node_index = recipe_individual.add_action_node(mix_action, nodes_indices)
        tree_roots.append(mix_node_index)

    ## Join mix nodes
    mix_action = random.choice(inverse_groups['mix'])
    mix_action = Action(mix_action, 'mix')
    mix_node_index = recipe_individual.add_action_node(mix_action, tree_roots)
    recipe_individual._root = mix_node_index

    ## Enanche individual
    for node_index in recipe_individual.nodes():
        node = recipe_individual.get_node(node_index)

        if type(node['object']) is Ingredient:
            if heat_ingredient(recipe_matrices.group_actions_base_ingredients, node['object'].base_object, threshold=0.70):
                parent_index = recipe_individual.get_edges(node_2=node_index)[0][0]
                recipe_individual.remove_edge(parent_index, node_index)

                heat_action = random.choice(inverse_groups['heat'])
                heat_action = Action(heat_action, 'heat')
                heat_action_node_index = recipe_individual.add_action_node(heat_action, [node_index])
                recipe_individual.add_generic_edge(parent_index, heat_action_node_index, {'type': 'primary'})

            if prepare_ingredient(recipe_matrices.group_actions_base_ingredients, node['object'].base_object, threshold=0.5):
                parent_index = recipe_individual.get_edges(node_2=node_index)[0][0]
                recipe_individual.remove_edge(parent_index, node_index)

                prepare_action = random.choice(inverse_groups['prepare'])
                prepare_action = Action(prepare_action, 'prepare')
                prepare_action_node_index = recipe_individual.add_action_node(prepare_action, [node_index])
                recipe_individual.add_generic_edge(parent_index, prepare_action_node_index, {'type': 'primary'})

        else:
            add_action = random.random() > 0.5
            if add_action:
                parent_indices = recipe_individual.get_edges(node_2=node_index)
                parent_index = parent_indices[0][0] if parent_indices else None

                action, group = random.choice(list(groups.items()))
                action = Action(action, group)
                action_node_index = recipe_individual.add_action_node(action, [node_index])

                if parent_index is None: # root
                    recipe_individual._root = action_node_index
                else:
                    recipe_individual.remove_edge(parent_index, node_index)
                    recipe_individual.add_generic_edge(parent_index, action_node_index, {'type': 'primary'})

    return recipe_individual

def generate_recipes_population(
    recipe_matrices:RecipeMatrices,
    individuals_number:int,
    required_ingredients:List[str], additional_ingredients:List[str] = None,
    ingredients_number:int = 3,
    min_addition_size:int = 0, max_addition_size:int = 4
    ) -> RecipePopulation:
    recipe_population = RecipePopulation()

    for _ in range(individuals_number):
        recipe_individual = generate_recipe_individual(recipe_matrices, required_ingredients, additional_ingredients, ingredients_number, min_addition_size, max_addition_size)
        recipe_population.add_individual(recipe_individual)

    return recipe_population

def random_recipes_population(recipe_universe:pd.DataFrame, individuals_number:int) -> Tuple[RecipePopulation, List[int]]:
    recipes_population = RecipePopulation()
    recipes_indices = []

    for _ in tqdm(range(individuals_number)):
        recipe_index = random.randint(0, len(recipe_universe)-1)
        recipes_indices.append(recipe_index)

        recipe = recipe_universe.iloc[recipe_index]
        recipe = Recipe.from_dataframe_row(recipe)
        recipe_graph = RecipeGraph.from_recipe(recipe)
        recipe_graph.simplify_graph()
        recipe_individual = RecipeIndividual.from_recipe_graph(recipe_graph)
        recipes_population.add_individual(recipe_individual)

    return recipes_population, recipes_indices
