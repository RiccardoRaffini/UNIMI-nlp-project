import random
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from algorithms.populations import RecipeIndividual
from commons.action_groups import groups, inverse_groups
from commons.recipes import RecipeMatrices, Action, Ingredient, Tool

T = TypeVar('T')

class Mutation(ABC, Generic[T]):
    def __call__(self, individual:T) -> T:
        raise NotImplementedError
    
    @abstractmethod
    def is_valid(self, individual:T) -> bool:
        raise NotImplementedError
    
## ==========
    
class RecipeNodeMutation(Mutation):
    def __init__(self, recipes_matrices:RecipeMatrices):
        super(RecipeNodeMutation, self).__init__()

        self._recipes_matrices = recipes_matrices

    def __call__(self, individual:RecipeIndividual, node_index:int) -> RecipeIndividual:
        ## Check mutation is valid
        assert self.is_valid(individual, node_index), 'cannot apply mutation to this individual'

        ## Mutate individual
        self._mutate(individual, node_index)

        return individual
    
    @abstractmethod
    def is_valid(self, individual:RecipeIndividual, node_index:int) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def _mutate(self, individual:RecipeIndividual, node_index:int) -> None:
        raise NotImplementedError

class SplitMixNodeMutation(RecipeNodeMutation):
    def __init__(self, recipes_matrices:RecipeMatrices):
        super(SplitMixNodeMutation, self).__init__(recipes_matrices)

        self._available_mixing_actions = inverse_groups['mix']

    def is_valid(self, individual:RecipeIndividual, node_index:int) -> bool:
        ## Check node is a mixing action
        node = individual.get_node(node_index)
        node_object = node['object']
        if not (type(node_object) is Action and node_object.group == 'mix'):
            return False

        ## Check children number
        childs_number = len(individual.get_children_indices(node_index))
        return childs_number > 2

    def _mutate(self, individual:RecipeIndividual, node_index:int) -> None:
        ## Split child nodes for upper and lower
        childrens = individual.get_children_indices(node_index)
        split_index = random.randint(1, len(childrens)-1)
        upper_childrens = childrens[:split_index]
        lower_childrens = childrens[split_index:]

        ## Remove edges of lower childrens
        lower_primary_childrens = []
        lower_secondary_childrens = []
        for lower_child in lower_childrens:
            child_edge_attributes = individual._graph.edges[node_index, lower_child]
            if child_edge_attributes['type'] == 'primary':
                lower_primary_childrens.append(lower_child)
            else: # child_edge_attributes['type'] == 'secondary':
                lower_secondary_childrens.append(lower_child)

            individual.remove_edge(node_index, lower_child)

        ## Create new mixing node
        chosen_index = random.randint(0, len(self._available_mixing_actions)-1)
        lower_mixing_action = self._available_mixing_actions[chosen_index]
        lower_mixing_action = Action(lower_mixing_action, 'mix')
        lower_mixing_action_node_index = individual.add_action_node(lower_mixing_action, lower_primary_childrens, lower_secondary_childrens)

        ## Connect actions
        individual.add_generic_edge(node_index, lower_mixing_action_node_index, {'type': 'primary'})

class DeleteActionNodeMutation(RecipeNodeMutation):
    def __init__(self, recipes_matrices:RecipeMatrices, delete_mixing_action:bool = False):
        super(DeleteActionNodeMutation, self).__init__(recipes_matrices)

        self._delete_mixing_action = delete_mixing_action

    def __call__(self, individual:RecipeIndividual, node_index:int):
        assert self.is_valid(individual, node_index), 'cannot apply mutation to this individual'

        node = individual.get_node(node_index)
        mixing_node = node['object'].group == 'mix'
        if not mixing_node or self._delete_mixing_action:
            self._mutate(individual, node_index)

        return individual
    
    def is_valid(self, individual:RecipeIndividual, node_index:int) -> bool:
        ## Check node is an action
        node = individual.get_node(node_index)
        if not type(node['object']) is Action:
            return False

        ## Check parent node existance
        parent_edges = individual.get_edges(node_2=node_index)
        return len(parent_edges) > 0
    
    def _mutate(self, individual:RecipeIndividual, node_index:int) -> None:
        ## Get parent node
        parent_node_index = individual.get_edges(node_2=node_index)[0][0]

        ## Get children indices
        child_nodes_edges = individual.get_edges(node_1=node_index)

        ## Replace edges
        for child_edge in child_nodes_edges:
            edge_attributes = individual._graph.edges[*child_edge]
            individual.remove_edge(*child_edge)
            individual.add_generic_edge(parent_node_index, child_edge[1], edge_attributes)

        ## Remove node
        individual.remove_node(node_index)

class InsertActionNodeMutation(RecipeNodeMutation):
    def __init__(self, recipes_matrices:RecipeMatrices, insert_mixing_actions:bool = False):
        super(InsertActionNodeMutation, self).__init__(recipes_matrices)

        self._insert_mixing_actions = insert_mixing_actions

        if self._insert_mixing_actions:
            self._available_actions = list(groups.items())
        else:
            self._available_actions = list(filter(lambda ag: ag[1] != 'mix', groups.items()))
    
    def is_valid(self, individual:RecipeIndividual, node_index:int) -> bool:
        ## Check node is an action
        node = individual.get_node(node_index)
        if not type(node['object']) is Action:
            return False

        ## Check parent node existance
        parent_edges = individual.get_edges(node_2=node_index)
        return len(parent_edges) > 0

    def _mutate(self, individual:RecipeIndividual, node_index:int) -> None:
        ## Find new action
        action_group = random.choice(self._available_actions)
        action = Action(*action_group)

        ## Get parent node and remove edge
        parent_node_index = individual.get_edges(node_2=node_index)[0][0]
        individual.remove_edge(parent_node_index, node_index)

        ## Insert new action node
        new_node_index = individual.add_action_node(action, [node_index])
        individual.add_generic_edge(parent_node_index, new_node_index, {'type': 'primary'})

class ReplaceActionNodeMutation(RecipeNodeMutation):
    def __init__(self, recipes_matrices:RecipeMatrices, limit_to_action_group:bool = True):
        super(ReplaceActionNodeMutation, self).__init__(recipes_matrices)

        self._limit_to_action_group = limit_to_action_group

    def is_valid(self, individual:RecipeIndividual, node_index:int) -> bool:
        ## Check node is an action
        node = individual.get_node(node_index)
        if not type(node['object']) is Action:
            return False

        ## Check parent node existance
        parent_edges = individual.get_edges(node_2=node_index)
        return len(parent_edges) > 0
    
    def _mutate(self, individual:RecipeIndividual, node_index:int) -> None:
        ## Find new action
        if self._limit_to_action_group:
            ## Get action group
            node = individual.get_node(node_index)
            group = node['object'].group

            if group is None:
                return # no alternative action
            
            ## Select actions in the same group
            possible_actions = [(action, group) for action in inverse_groups[group]]
        else:
            ## Select all actions
            possible_actions = list(groups.items())

        action_group = random.choice(possible_actions)
        action = Action(*action_group)

        ## Get parent node and remove edge
        parent_node_index = individual.get_edges(node_2=node_index)[0][0]
        parent_edge_type = individual._graph.edges[parent_node_index, node_index]['type']
        individual.remove_edge(parent_node_index, node_index)

        ## Remove children edges
        children_indices = individual.get_children_indices(node_index)
        primary_children = []
        secondary_children = []

        for child_index in children_indices:
            if individual._graph.edges[node_index, child_index]['type'] == 'primary':
                primary_children.append(child_index)
            else: # secondary
                secondary_children.append(child_index)

            individual.remove_edge(node_index, child_index)

        ## Remove old action node
        individual.remove_node(node_index)

        ## Add new action node
        new_node_index = individual.add_action_node(action, primary_children, secondary_children)
        individual.add_generic_edge(parent_node_index, new_node_index, {'type': parent_edge_type})

class AddIngredientToActionNodeMutation(RecipeNodeMutation):
    def __init__(self, recipes_matrices:RecipeMatrices, limit_to_mix_actions:bool = True, limit_to_action_group_ingredients:bool = True):
        super(AddIngredientToActionNodeMutation, self).__init__(recipes_matrices)

        self._limit_to_mix_actions = limit_to_mix_actions
        self._limit_to_action_group_ingrdients = limit_to_action_group_ingredients

    def is_valid(self, individual:RecipeIndividual, node_index:int) -> bool:
        ## Check node is an action
        node = individual.get_node(node_index)
        if not type(node['object']) is Action:
            return False
        
        return True
        
    def _mutate(self, individual:RecipeIndividual, node_index:int) -> None:
        ## Check node group
        node = individual.get_node(node_index)
        group = node['object'].group if node['object'].group else node['object'].action

        if self._limit_to_mix_actions and group != 'mix': 
            return
        
        ## Find new ingredient
        ingredients_labels = self._recipes_matrices.group_actions_base_ingredients.get_labels()[1]
        if self._limit_to_action_group_ingrdients:
            ## Get valid ingredients
            group_index = self._recipes_matrices.group_actions_base_ingredients.label_to_row_index(group, False)
            ingredients_indices = self._recipes_matrices.actions_base_ingredients.get_csr_matrix().getrow(group_index).nonzero()[1]
            possible_ingredients = [ingredients_labels[index] for index in ingredients_indices]
        else:
            ## Select all (seen) ingredients
            possible_ingredients = ingredients_labels

        ingredient_text = random.choice(possible_ingredients)
        ingredient = Ingredient(ingredient_text)

        ## Add new ingredient
        new_node_index = individual.add_ingredient_node(ingredient)
        individual.add_generic_edge(node_index, new_node_index, {'type': 'primary'})

class ReplaceIngredientNodeMutation(RecipeNodeMutation):
    def __init__(self, recipes_matrices:RecipeMatrices, limit_to_action_ingredients:bool = True):
        super(ReplaceIngredientNodeMutation, self).__init__(recipes_matrices)

        self._limit_to_action_ingrdients = limit_to_action_ingredients

    def is_valid(self, individual:RecipeIndividual, node_index:int) -> bool:
        ## Check node is an ingredient
        node = individual.get_node(node_index)
        if not type(node['object']) == Ingredient:
            return False
    
        ## Check parent node existance
        parent_edges = individual.get_edges(node_2=node_index)
        return len(parent_edges) > 0
    
    def _mutate(self, individual:RecipeIndividual, node_index:int) -> None:
        ## Get parent node
        parent_node_index = individual.get_edges(node_2=node_index)[0][0]
        edge_attributes = individual._graph.edges[parent_node_index, node_index]

        ## Find new ingredient
        ingredients_labels = self._recipes_matrices.actions_base_ingredients.get_labels()[1]
        possible_ingredients = None
        if self._limit_to_action_ingrdients:
            ## Get parent action
            parent_node = individual.get_node(parent_node_index)
            action = parent_node['object'].action

            ## Get valid ingredients
            action_index = self._recipes_matrices.actions_base_ingredients.label_to_row_index(action, False)
            ingredients_indices = self._recipes_matrices.actions_base_ingredients.get_csr_matrix().getrow(action_index).nonzero()[1]
            possible_ingredients = [ingredients_labels[index] for index in ingredients_indices]
        elif self._limit_to_action_ingrdients == False or possible_ingredients is None:
            ## Select all (seen) ingredients
            possible_ingredients = ingredients_labels

        ingredient_text = random.choice(possible_ingredients)
        ingredient = Ingredient(ingredient_text)

        ## Remove old ingredient
        individual.remove_edge(parent_node_index, node_index)
        individual.remove_node(node_index)

        ## Add new ingredient
        new_node_index = individual.add_ingredient_node(ingredient)
        individual.add_generic_edge(parent_node_index, new_node_index, edge_attributes)

class InsertActionToIngredientNodeMutation(RecipeNodeMutation):
    def __init__(self, recipes_matrices:RecipeMatrices, limit_to_seen_actions:bool = True):
        super(InsertActionToIngredientNodeMutation, self).__init__(recipes_matrices)
        
        self._limit_to_seen_actions = limit_to_seen_actions
    
    def is_valid(self, individual:RecipeIndividual, node_index:int) -> bool:
        ## Check node is an ingredient
        node = individual.get_node(node_index)
        if not type(node['object']) == Ingredient:
            return False
    
        ## Check parent node existance
        parent_edges = individual.get_edges(node_2=node_index)
        return len(parent_edges) > 0

    def _mutate(self, individual:RecipeIndividual, node_index:int) -> None:
        ## Find new action
        actions_labels = self._recipes_matrices.actions_base_ingredients.get_labels()[0]
        if self._limit_to_seen_actions:
            ## Get node ingredient
            node = individual.get_node(node_index)
            ingredient_name = node['object'].base_object

            ## Get valid actions
            ingredient_index = self._recipes_matrices.actions_base_ingredients.label_to_column_index(ingredient_name, False)
            actions_indices = self._recipes_matrices.actions_base_ingredients.get_csr_matrix().getcol(ingredient_index).nonzero()[0]
            possible_actions = [actions_labels[index] for index in actions_indices]
        else:
            ## Select all (seen) actions
            possible_actions = actions_labels

        action_name = random.choice(possible_actions)
        action_group = groups[action_name]
        action = Action(action_name, action_group)

        ## Insert new action
        parent_node_index = individual.get_edges(node_2=node_index)[0][0]
        individual.remove_edge(parent_node_index, node_index)
        new_node_index = individual.add_action_node(action, [node_index])
        individual.add_generic_edge(parent_node_index, new_node_index, {'type': 'primary'})

class AddToolNodeMutation(RecipeNodeMutation):
    def __init__(self, recipes_matrices:RecipeMatrices, limit_to_action_tools:bool = True):
        super(AddToolNodeMutation, self).__init__(recipes_matrices)

        self._limit_to_action_tools = limit_to_action_tools

    def is_valid(self, individual:RecipeIndividual, node_index:int) -> bool:
        ## Check node is an action
        node = individual.get_node(node_index)
        if not type(node['object']) == Action:
            return False
        
        return True
    
    def _mutate(self, individual:RecipeIndividual, node_index:int) -> None:
        ## Find new tool
        tools_labels = self._recipes_matrices.group_actions_tools.get_labels()[1]
        possible_tools = None
        if self._limit_to_action_tools:
            ## Get node tool
            parent_node = individual.get_node(node_index)
            group = parent_node['object'].group if parent_node['object'].group else parent_node['object'].action

            ## Get valid tools
            group_index = self._recipes_matrices.group_actions_tools.label_to_row_index(group, False)
            tools_indices = self._recipes_matrices.group_actions_tools.get_csr_matrix().getrow(group_index).nonzero()[1]
            possible_tools = [tools_labels[index] for index in tools_indices]
        else:
            ## Select all (seen) tools
            possible_tools = tools_labels

        if possible_tools is None:
            return

        tool_text = random.choice(tools_labels)
        tool = Tool(tool_text)

        ## Add new tool
        new_node_index = individual.add_tool_node(tool)
        individual.add_generic_edge(node_index, new_node_index, {'type': 'secondary'})

class ReplaceToolNodeMutation(RecipeNodeMutation):
    def __init__(self, recipes_matrices:RecipeMatrices, limit_to_action_tools:bool = True, skip_if_not_available:bool = True):
        super(ReplaceToolNodeMutation, self).__init__(recipes_matrices)

        self._limit_to_action_tools = limit_to_action_tools
        self._skip_if_not_available = skip_if_not_available # if false uses all tools
    
    def is_valid(self, individual:RecipeIndividual, node_index:int) -> bool:
        ## Check node is a tool
        node = individual.get_node(node_index)
        if not type(node['object']) == Tool:
            return False
    
        ## Check parent node existance
        parent_edges = individual.get_edges(node_2=node_index)
        return len(parent_edges) > 0
    
    def _mutate(self, individual:RecipeIndividual, node_index:int) -> None:
        ## Get parent node
        parent_node_index = individual.get_edges(node_2=node_index)[0][0]
        edge_attributes = individual._graph.edges[parent_node_index, node_index]

        ## Find new tool
        tools_labels = self._recipes_matrices.group_actions_tools.get_labels()[1]
        possible_tools = None
        if self._limit_to_action_tools:
            ## Get node tool
            parent_node = individual.get_node(parent_node_index)
            group = parent_node['object'].group if parent_node['object'].group else parent_node['object'].action

            ## Get valid tools
            group_index = self._recipes_matrices.group_actions_tools.label_to_row_index(group, False)
            tools_indices = self._recipes_matrices.group_actions_tools.get_csr_matrix().getrow(group_index).nonzero()[1]
            possible_tools = [tools_labels[index] for index in tools_indices]

        elif not self._limit_to_action_tools:
            ## Select all (seen) tools
            possible_tools = tools_labels

        if possible_tools is None and self._skip_if_not_available:
            return

        tool_text = random.choice(tools_labels)
        tool = Tool(tool_text)

        ## Remove old tool
        individual.remove_edge(parent_node_index, node_index)
        individual.remove_node(node_index)

        ## Add new tool
        new_node_index = individual.add_tool_node(tool)
        individual.add_generic_edge(parent_node_index, new_node_index, edge_attributes)
