import itertools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List

from algorithms.populations import RecipeIndividual
from commons.recipes import RecipeMatrices, Ingredient, Action

T = TypeVar('T')

class FitnessEvaluator(ABC, Generic[T]):
    @abstractmethod
    def evaluate(self, individual:T) -> float:
        raise NotImplementedError()
    
    def __call__(self, individual:T) -> float:
        evaluation = self.evaluate(individual)
        return evaluation

## ==========

class RecipeNodesCountEvaluator(FitnessEvaluator[RecipeIndividual]):
    def __init__(self, nodes_limit:int = 100):
        super(RecipeNodesCountEvaluator, self).__init__()

        self._nodes_limit = nodes_limit

    def evaluate(self, individual:RecipeIndividual) -> float:
        nodes_number = len(individual.nodes())
        score = 1 - nodes_number / self._nodes_limit

        return score

class RecipeScoreEvaluator(FitnessEvaluator[RecipeIndividual]):
    def __init__(self, recipe_matrices:RecipeMatrices, threshold:float = 0.5, epsilon:float = 0.1, minimum_actions:int = 3, minimum_ingredients:int = 3):
        super(RecipeScoreEvaluator, self).__init__()

        self._recipe_matrices = recipe_matrices
        self._threshold = threshold
        self._epsilon = epsilon
        self._minimum_actions = minimum_actions
        self._minimum_ingredients = minimum_ingredients

    def evaluate(self, individual:RecipeIndividual) -> float:
        individual_tree_score = self._tree_score(individual)

        return individual_tree_score
    
    def _heat_score(self, ingredient:Ingredient) -> float:
        group_actions_base_ingredients_matrix = self._recipe_matrices.group_actions_base_ingredients
        heat_index = group_actions_base_ingredients_matrix.label_to_row_index('heat', add_not_existing=False)
        ingredient_index = group_actions_base_ingredients_matrix.label_to_column_index(ingredient.base_object)

        if heat_index != -1 and ingredient_index != -1:
            group_actions_base_ingredients_matrix = group_actions_base_ingredients_matrix.get_sparse_matrix().todok()
            heat_score = group_actions_base_ingredients_matrix[heat_index, ingredient_index]
        else:
            heat_score = 0

        return heat_score

    def _prepare_score(self, ingredient:Ingredient) -> float:
        group_actions_base_ingredients_matrix = self._recipe_matrices.group_actions_base_ingredients
        prepare_index = group_actions_base_ingredients_matrix.label_to_row_index('prepare', add_not_existing=False)
        ingredient_index = group_actions_base_ingredients_matrix.label_to_column_index(ingredient.base_object)

        if prepare_index != -1 and ingredient_index != -1:
            group_actions_base_ingredients_matrix = group_actions_base_ingredients_matrix.get_sparse_matrix().todok()
            prepare_score = group_actions_base_ingredients_matrix[prepare_index, ingredient_index]
        else:
            prepare_score = 0

        return prepare_score

    def _binary_heat_score(self, ingredient:Ingredient, heated:bool) -> int:
        preparation_score = self._prepare_score(ingredient)
        heat_score = self._heat_score(ingredient)
        inverse_ph_ration = 1 - (preparation_score / heat_score) if heat_score else 0

        if heated == False and inverse_ph_ration > (self._threshold + self._epsilon):
            binary_heat_score = 0
        elif heated == True and inverse_ph_ration < (self._threshold - self._epsilon):
            binary_heat_score = 0
        else:
            binary_heat_score = 1

        return binary_heat_score

    def _binary_prepare_score(self, ingredient:Ingredient, prepared:bool) -> int:
        preparation_score = self._prepare_score(ingredient)
        heat_score = self._heat_score(ingredient)
        ph_ration = preparation_score / heat_score if heat_score else 1

        if prepared == False and ph_ration > (self._threshold + self._epsilon):
            binary_preparation_score = 0
        elif prepared == True and ph_ration < (self._threshold + self._epsilon):
            binary_preparation_score = 0
        else:
            binary_preparation_score = 1

        return binary_preparation_score

    def _ingredient_duplicate_actions_score(self, ingredient:Ingredient) -> float:
        ## Find applied actions
        applied_actions = ingredient.applied_actions
        
        ## Compute duplicate score
        ingredient_duplicate_actions_score = self._duplicate_actions_score(applied_actions)

        return ingredient_duplicate_actions_score
    
    def _tree_score(self, tree_individual:RecipeIndividual) -> float:
        ## Computes main scores
        ingredients_scores = []
        actions_scores = []
        mixing_actions_scores = []
        actions = []

        for node_index in tree_individual.nodes():
            node = tree_individual.get_node(node_index)

            ## Mixing action node
            if type(node['object']) == Action and node['object'].group == 'mix':
                mixing_action_score = self._mix_node_score(tree_individual, node)
                mixing_actions_scores.append(mixing_action_score)
                actions.append(node['object'])

            ## Generic action node
            elif type(node['object']) == Action:
                action_score = self._action_node_score(tree_individual, node)
                actions_scores.append(action_score)
                actions.append(node['object'])

            ## Ingredient node
            elif type(node['object']) == Ingredient:
                ingredient_score = self._ingredient_node_score(node)
                ingredients_scores.append(ingredient_score)

        ## Compute final nodes score
        ingredients_number = len(ingredients_scores)
        actions_number = len(actions_scores)
        mixing_actions_number = len(mixing_actions_scores)

        nodes_score = (sum(ingredients_scores) + sum(actions_scores) + sum(mixing_actions_scores)) / (ingredients_number + actions_number + mixing_actions_number)

        ## Compute additional scores
        minimum_actions = int(actions_number >= self._minimum_actions)
        minimum_ingredients = int(ingredients_number >= self._minimum_ingredients)

        duplicate_actions_score = self._duplicate_actions_score(actions)

        ## Compute final tree score
        tree_score = nodes_score * duplicate_actions_score * minimum_actions * minimum_ingredients

        return tree_score
    
    def _duplicate_actions_score(self, actions:List[Action]) -> float:
        ## Get actions names
        applied_actions = [action.action for action in actions]
        unique_applied_actions = set(applied_actions)

        ## Count duplicate action
        applied_actions_number = len(applied_actions)
        duplicated_actions_number = applied_actions_number - len(unique_applied_actions)

        ## Compute duplicate score
        if applied_actions_number == 0:
            duplicate_actions_score = 1
        else:
            duplicate_actions_score = (applied_actions_number - duplicated_actions_number) / applied_actions_number

        return duplicate_actions_score
    
    def _ingredient_node_score(self, ingredient_node:dict) -> float:
        ## Extract ingredient
        ingredient = ingredient_node['object']
        prepared = 'prepare' in ingredient.applied_actions_groups
        heated = 'heat' in ingredient.applied_actions_groups

        ## Compute base scores
        multiplier = 0.5
        prepare_score = self._binary_prepare_score(ingredient, prepared)
        heat_score = self._binary_heat_score(ingredient, heated)
        duplicate_actions_score = self._ingredient_duplicate_actions_score(ingredient)

        ## Compute final ingredient node score
        ingredient_node_score = multiplier * (prepare_score + heat_score) * duplicate_actions_score

        return ingredient_node_score

    def _action_node_score(self, tree:RecipeIndividual, action_node:dict) -> float:
        ## Find all ingredients in sub tree
        ingredients = []

        nodes_queue = [action_node['index']]
        while nodes_queue:
            node_index = nodes_queue.pop(0)
            node = tree.get_node(node_index)

            if type(node['object']) == Ingredient:
                ingredients.append(node['object'])
            elif type(node['object']) == Action:
                nodes_queue.extend(tree.get_children_indices(node_index))

        if len(ingredients) == 0:
            return 0

        ## Count valid actions in matrices
        actions_ingredients = self._recipe_matrices.actions_base_ingredients
        action = action_node['object']
        action_index = actions_ingredients.label_to_row_index(action.action, False)
        if action_index == -1:
            return 0

        actions_ingredients_matrix = actions_ingredients.get_csr_matrix()

        valid_actions_number = 0
        for ingredient in ingredients:
            ingredient_index = actions_ingredients.label_to_column_index(ingredient.base_object, False)

            if ingredient_index != -1 and actions_ingredients_matrix[action_index, ingredient_index] > 0:
                valid_actions_number += 1

        ## Compute final action node score
        ingredients_number = len(ingredients)
        action_node_score = valid_actions_number / ingredients_number

        return action_node_score

    def _mix_node_score(self, tree:RecipeIndividual, mix_node:dict) -> float:
        ## Find all ingredients in sub tree
        ingredients = []

        nodes_queue = [mix_node['index']]
        while nodes_queue:
            node_index = nodes_queue.pop(0)
            node = tree.get_node(node_index)

            if type(node['object']) == Ingredient:
                ingredients.append(node['object'])
            elif type(node['object']) == Action:
                nodes_queue.extend(tree.get_children_indices(node_index))

        if len(ingredients) == 0:
            return 0

        ## Create ingredient pairs and count valid mixings in matrices
        ingredients_pairs = itertools.combinations(ingredients, r=2)
        ingredients_ingredients = self._recipe_matrices.base_ingredients_base_ingredients
        ingredients_ingredients_matrix = ingredients_ingredients.get_csr_matrix()

        ingredients_pairs_number = 0
        valid_ingredients_pairs_number = 0
        for ingredient_1, ingredient_2 in ingredients_pairs:
            ingredients_pairs_number += 1
            ingredient_1_index = ingredients_ingredients.label_to_index(ingredient_1.base_object, False)
            ingredient_2_index = ingredients_ingredients.label_to_index(ingredient_2.base_object, False)

            if ingredient_1_index != -1 and ingredient_2_index != -1 and ingredients_ingredients_matrix[ingredient_1_index, ingredient_2_index] > 0:
                valid_ingredients_pairs_number += 1

        if ingredients_pairs_number == 0:
            return 0

        ## Compute final mix node score
        mix_node_score = valid_ingredients_pairs_number / ingredients_pairs_number

        return mix_node_score
