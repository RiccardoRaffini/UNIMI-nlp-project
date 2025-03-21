from __future__ import annotations

import itertools
import networkx as nx
import os
import pandas as pd
import pygraphviz
import textwrap
import zipfile
from abc import ABC
from collections import deque
from functools import reduce
from typing import List, Tuple, Dict, Any, Union, Optional

import commons.nlp_utils as nlputils
from algorithms.matrices import AdjacencyMatrix, ActionsIngredientsMatrix, MixedIngredientsMatrix, ActionsToolsMatrix
from commons.action_groups import groups

class Action:
    """
    A class that allows to represent a culinary action.
    """

    def __init__(self, action:str, action_group:str=None):
        self._action = action
        self._action_group = action_group

    @property
    def action(self) -> str:
        return self._action
    
    @property
    def group(self) -> Optional[str]:
        return self._action_group

    @property
    def full_action(self) -> str:
        return f'{self._action} ({self._action_group if self._action_group else "n/a"})'

    def __str__(self):
        return self._action
    
    def __repr__(self):
        return self._action

class RecipeObject(ABC):
    """
    An abstract base class that allows to represent objects involved in a
    cooking recipe, such object are characterzed by a name and eventually a set
    of adjectives.
    """

    def __init__(self, name:str, adjectives:List[str]=None):
        self._name = name
        self._adjectives = []
        if adjectives is not None:
            self._adjectives = adjectives.copy()

    @property
    def base_object(self) -> str:
        return self._name
    
    @property
    def adjectives(self) -> List[str]:
        return self._adjectives.copy()
    
    @property
    def full_object(self) -> str:
        if self._adjectives:
            return f'{" ".join(self._adjectives)} {self._name}'
        else:
            return self.base_object
    
    def base_similarity(self, other:'RecipeObject') -> float:
        """
        Computes the jaccard similarity (IoU) between the tokens composing the
        base form of this object and the tokens of the given object base form.

        Args:
            other (RecipeObject): recipe object with which to measure similarity.

        Returns:
            float: jaccard similarity of these objects in their base form.
        """

        self_tokens = set(self.base_object.split())
        other_tokens = set(other.base_object.split())
        similarity = len(self_tokens.intersection(other_tokens)) / len(self_tokens.union(other_tokens))

        return similarity

    def full_similarity(self, other:'RecipeObject') -> float:
        """
        Computes the jaccard similarity (IoU) between the tokens composing the
        full form of this object and the tokens of the given object full form.

        Args:
            other (RecipeObject): recipe object with which to measure similarity.

        Returns:
            float: jaccard similarity of these objects in their full form.
        """
        
        self_tokens = set(self.full_object.split())
        other_tokens = set(other.full_object.split())
        similarity = len(self_tokens.intersection(other_tokens)) / len(self_tokens.union(other_tokens))

        return similarity
    
    def __str__(self):
        return self.full_object
    
    def __repr__(self):
        return self.full_object
    
class Ingredient(RecipeObject):
    """
    A class that allows to represent a cooking ingredient.
    """

    def __init__(self, name:str, adjectives:List[str]=None, actions:List[Action]=None):
        super(Ingredient, self).__init__(name, adjectives)

        self._actions_list = []
        if actions:
            self._actions_list = actions.copy()
        self.use_default_group_names = True

    def add_applied_action(self, action:Action) -> None:
        self._actions_list.append(action)

    def set_applied_actions(self, actions:List[Action]) -> None:
        self._actions_list = actions.copy()

    @property
    def applied_actions(self) -> List[Action]:
        return self._actions_list.copy()
    
    @property
    def applied_actions_names(self) -> List[str]:
        return [action.action for action in self._actions_list]
    
    @property
    def applied_actions_groups(self) -> List[str]:
        return [(action.action if not action.group and self.use_default_group_names else action.group) for action in self._actions_list]
    
    @property
    def full_object(self) -> str:
        object_ = super().full_object

        if self.applied_actions_names:
            object_ = f'{" ".join(sorted(set(self.applied_actions_names)))} {object_}'

        return object_

class Tool(RecipeObject):
    """
    A class that allows to represent a cooking tool.
    """

    def __init__(self, name:str, adjectives:List[str]=None):
        super(Tool, self).__init__(name, adjectives)

class Miscellaneous(RecipeObject):
    """
    A class that allows to represent a generic culinary object.
    """

    def __init__(self, name:str, adjectives:List[str]=None):
        super(Miscellaneous, self).__init__(name, adjectives)

class Recipe:
    """
    A class that allows to represent a cooking recipe. Recipes are characterized
    by an _id_, a *name*, a *description*, a *category*, a collection of
    *ingredients*, their *quantities* and a sequence os *instructions* to prepare
    such recipe.

    The class also provides a convenient method to create a recipe from a
    :module:`pandas` :class:`DataFrame` row or :class:`Series`, but they must
    follow an appropriate naming convention.
    """

    processor:nlputils.RecipeProcessor = None

    @classmethod
    def set_recipe_processor(cls, processor:nlputils.RecipeProcessor) -> None:
        """
        Sets a new recipe processor to use during recipes initialization.

        Args:
            processor (RecipeProcessor): recipe processor to assign.
        """

        cls.processor = processor

    @classmethod
    def from_dataframe_row(cls, dataframe_row:pd.Series) -> 'Recipe':
        """
        Returns a new :class:`Recipe` instance by accessing the information
        given as :class:`pandas.DataFrame` row or :class:`pandas.Series`.
        The row attributes must follow an appropriate naming convention.

        Args:
            dataframe_row (pd.Series): dataframe row or series containing the
            new recipe information.

        Returns:
            Recipe: new recipe instance.
        """

        return cls(
            dataframe_row.name, dataframe_row['Name'], dataframe_row['Description'], dataframe_row['Category'],
            dataframe_row['Ingredients'], dataframe_row['IngredientQuantities'],
            dataframe_row['Instructions']
        )

    def __init__(self,
        id:int, name:str, description:str, category:str,
        ingredients:List[str], ingredient_quantities:List[str],
        instructions:List[str]
    ) -> None:
        ## Base fields
        self._id = id
        self._name = name
        self._description = description
        self._category = category

        ## Raw fields
        self._raw_ingredients = ingredients.copy()
        self._raw_ingredient_quantities = ingredient_quantities.copy()
        self._raw_instructions = instructions.copy()

        ## Processed instruction steps
        self._steps_ingredients = None
        self._steps_tools = None
        self._steps_actions = None

        ## Process new recipe
        self._process_recipe()

    @property
    def steps_ingredients(self) -> List[List[str]]:
        return self._steps_ingredients.copy()

    @property
    def steps_tools(self) -> List[List[str]]:
        return self._steps_tools.copy()

    @property
    def steps_actions(self) -> List[List[Tuple[str, List[str], List[str]]]]:
        return self._steps_actions.copy()

    def _process_recipe(self) -> None:
        """
        Processes recipe's raw fields to obtain its internal representation using
        the recipe processor assigned to ths class.
        """

        ## Check that processor is set
        assert self.processor is not None, 'cannot process a recipe without a processor'

        ## Process instructions
        processed_instructions = self.processor.process_instructions(self._raw_instructions)

        self._steps_ingredients, self._steps_tools, self._steps_actions = reduce(
            lambda a, b: (a[0] + b[1], a[1] + b[2], a[2] + b[3]),
            processed_instructions,
            ([], [], [])
        )

class RecipeGraph:
    @classmethod
    def from_recipe(cls, recipe:Recipe, additional_configuration:Dict[str, Any] = None) -> 'RecipeGraph':
        ## Create empty recipe graph
        recipe_graph = cls(additional_configuration=additional_configuration)
        
        ## Iterate recipe's steps
        last_subtree_root_index = -1
        for step_ingredients, step_tools, step_actions in zip(recipe.steps_ingredients, recipe.steps_tools, recipe.steps_actions):
            ## Check empty step
            if len(step_actions) == 0:
                continue

            ## Iterate actions in step
            previous_action_index = -1
            for action, primary_objects, secondary_objects in step_actions:
                ## Create primary object nodes
                primary_objects_indices = []
                for object in primary_objects:
                    if object in step_ingredients:
                        index = recipe_graph.add_ingredient_node(object)
                    elif object in step_tools:
                        index = recipe_graph.add_tool_node(object)
                    else:
                        index = recipe_graph.add_misc_node(object)

                    primary_objects_indices.append(index)

                ## Create secondary object nodes
                secondary_objects_indices = []
                for object in secondary_objects:
                    if object in step_ingredients:
                        index = recipe_graph.add_ingredient_node(object)
                    elif object in step_tools:
                        index = recipe_graph.add_tool_node(object)
                    else:
                        index = recipe_graph.add_misc_node(object)

                    secondary_objects_indices.append(index)

                ## Connect to previous steps
                if previous_action_index != -1:
                    primary_objects_indices.append(previous_action_index)
                elif last_subtree_root_index != -1:
                    primary_objects_indices.append(last_subtree_root_index)

                ## Add action node
                action_index = recipe_graph.add_action_node(action, primary_objects_indices, secondary_objects_indices)
                previous_action_index = action_index

            last_subtree_root_index = previous_action_index

        recipe_graph._root = last_subtree_root_index

        return recipe_graph

    def simplify_graph(self) -> None:
        previous_node = self.get_node(self.get_root_index())
        current_node = None

        while previous_node:
            ## Find next action node
            # TODO: handle multi-actions children
            current_node = None
            for child_index in self.get_children_indices(previous_node['index']):
                child_node = self.get_node(child_index)
                if type(child_node['object']) == Action:
                    current_node = child_node
                    break

            ## No next action foun
            if current_node is None:
                break

            ## Check if current and previus action are equal
            if current_node['object'].action == previous_node['object'].action:
                ## Remove edge between same actions
                actions_edge = self.get_edges(previous_node['index'], current_node['index'])[0]
                self.remove_edge(*actions_edge)

                for child_index in self.get_children_indices(current_node['index']):
                    ## Move current node children to parent node
                    child_edge_attributes = self._graph.edges[current_node['index'], child_index]
                    self.remove_edge(current_node['index'], child_index)
                    self.add_generic_edge(previous_node['index'], child_index, child_edge_attributes)

                ## Remove current node
                self.remove_node(current_node['index'])
                current_node = None

            ## Remove empty action nodes
            elif len(self.get_children_indices(current_node['index'])) == 0:
                self.remove_edge(previous_node['index'], current_node['index'])
                self.remove_node(current_node['index'])
                current_node = None

            else:
                previous_node = current_node
                current_node = None

    def apply_actions_to_ingredients(self, starting_index:int = None) -> None:
        ## Clear actions
        self.remove_actions_from_ingredients()

        ## Traverse graph
        if starting_index is None:
            nodes_actions_queue = [(self._root, [])]
        else:
            nodes_actions_queue = [(starting_index, [])]

        while nodes_actions_queue:
            node_index, actions = nodes_actions_queue.pop(0)
            node = self.get_node(node_index)

            ## Apply action to ingredient
            if type(node['object']) == Ingredient:
                node['object'].set_applied_actions(actions)
            
            ## Extend actions
            elif type(node['object']) == Action:
                for child_index in self.get_children_indices(node_index):
                    nodes_actions_queue.append((child_index, actions + [node['object']]))

    def remove_actions_from_ingredients(self) -> None:
        for node_index in self.nodes():
            node = self.get_node(node_index)
            node_object = node['object']

            if type(node_object) == Ingredient:
                node_object.set_applied_actions([])
                node['label'] = node_object.full_object if self._full_label else node_object.base_object

    def __init__(self, additional_configuration:Dict[str, Any] = None, show_full_label:bool = False, show_action_group:bool = False):
        ## Underlying graph
        self._graph = nx.DiGraph()
        self._root = None
        self._full_label = show_full_label
        self._action_group = show_action_group
        self._indices_count = 0

        ## Graph configuration
        self._graph_configuration = {
            'node_attributes': {
                'Ingredient': {'style': 'rounded,filled', 'fillcolor': '#ffe070', 'shape': 'ellipse'},
                'Tool': {'style': 'rounded,filled', 'fillcolor': '#c6f7a6', 'shape': 'ellipse'},
                'Action': {'style': 'rounded,filled', 'fillcolor': '#bcd9f5', 'shape': 'box'},
                'Miscellaneous': {'style': 'rounded,filled', 'fillcolor': '#f0f0f0', 'shape': 'ellipse'},
            }
        }

        if additional_configuration is not None:
            self._graph_configuration = self._graph_configuration | additional_configuration

    def nodes(self) -> List[int]:
        return list(self._graph.nodes.keys())

    def get_node(self, node_index:int) -> Dict[str, Any]:
        return self._graph.nodes[node_index]
    
    def get_root_index(self) -> int:
        return self._root
    
    def get_children_indices(self, node_index:int) -> List[int]:
        edges = self._graph.edges(node_index)
        if not edges:
            return []
        
        _, children_indices = zip(*edges)

        return list(children_indices)

    def _get_next_index(self) -> int:
        index = self._indices_count
        self._indices_count += 1
        return index

    def add_ingredient_node(self, ingredient:Ingredient) -> int:
        node_index = self._get_next_index()
        node_text = ingredient.full_object if self._full_label else ingredient.base_object
        return self.add_recipe_node(node_index, ingredient, node_text)

    def add_tool_node(self, tool:Tool) -> int:
        node_index = self._get_next_index()
        node_text = tool.full_object if self._full_label else tool.base_object
        return self.add_recipe_node(node_index, tool, node_text)
    
    def add_misc_node(self, object:Miscellaneous) -> int:
        node_index = self._get_next_index()
        node_text = object.full_object if self._full_label else object.base_object
        return self.add_recipe_node(node_index, object, node_text)

    def add_action_node(self, action:Action, primary_objects_nodes:List[int], secondary_objects_nodes:List[str] = None) -> int:
        node_index = self._get_next_index()
        node_text = action.full_action if self._action_group else action.action
        self.add_recipe_node(node_index, action, node_text)

        for primary_index in primary_objects_nodes:
            self.add_generic_edge(node_index, primary_index, {'type': 'primary'})

        if secondary_objects_nodes:
            for secondary_index in secondary_objects_nodes:
                self.add_generic_edge(node_index, secondary_index, {'type': 'secondary'})
                self._graph.nodes[secondary_index]['style'] = 'rounded,filled,dashed'
        
        return node_index

    def add_recipe_node(self, index:int, object:Union[RecipeObject, Action], label:str) -> int:
        type_ = type(object).__name__
        node_attributes = {'index': index, 'label': label, 'type': type_, 'object': object}
        node_attributes.update(self._graph_configuration['node_attributes'][type_])

        self._graph.add_node(index, **node_attributes)
        return index
    
    def remove_node(self, index:int) -> None:
        self._graph.remove_node(index)

    def add_generic_edge(self, node_1:int, node_2:int, attributes:Dict[str, Any] = {}) -> None:
        self._graph.add_edge(node_1, node_2, **attributes)

    def remove_edge(self, node_1:int, node_2:int) -> None:
        self._graph.remove_edge(node_1, node_2)

    def get_edges(self, node_1:int = None, node_2:int = None) -> List[Any]:
        assert not (node_1 is None and node_2 is None), 'at least one node must be specified'

        if node_1 is not None and node_2 is not None:
            return list(filter(lambda e: e == (node_1, node_2), self._graph.edges))

        elif node_1 is not None:
            return list(filter(lambda e: e[0] == node_1, self._graph.edges))

        elif node_2 is not None:
            return list(filter(lambda e: e[1] == node_2, self._graph.edges))

    def to_gviz(self, filename:str) -> None:
        agraph:pygraphviz.AGraph = nx.nx_agraph.to_agraph(self._graph)

        agraph.node_attr['style'] = 'rounded,filled'
        agraph.graph_attr['rankdir'] = 'TB'

        for node in agraph.iternodes():
            node.attr['label'] = f"<{'<BR />'.join(textwrap.wrap(node.attr['label'], 10, break_long_words=False))}>"

        agraph.layout(prog='dot')
        agraph.draw(filename)
        #agraph.write('recipe_graph.dot')

    def __str__(self):
        pass

    def __repr__(self):
        pass

class RecipeMatrices:
    """
    A class that groups all relational matrices relative to a single recipe or
    a group of recipes, obtainable from their recipe graph.
    Those matrices express ingredient-ingredient, action-ingredient and
    action-tool relations.

    The class also provides convenient methods to create the matrices from a
    :class:`RecipeGraph` instance or directly from a :class:`Recipe` instance.
    """

    @classmethod
    def from_recipe(cls, recipe:Recipe, compiled:bool = True) -> 'RecipeMatrices':
        """
        Returns a new :class:`RecipeMatrices` instance by processing the information
        given as a :class:`Recipe`.

        Args:
            recipe (:class:`Recipe`): recipe to process.
            compiled (:class:`bool`, optional): whether or not to compile the final
            matrices. Defaults to True.

        Returns:
            :class:`RecipeMatrices`: new recipes matrices instance.
        """

        recipe_graph = RecipeGraph.from_recipe(recipe)
        recipe_matrices = cls.from_recipe_graph(recipe_graph, compiled)

        return recipe_matrices

    @classmethod
    def from_recipe_graph(cls, recipe_graph:RecipeGraph, compiled:bool = True) -> 'RecipeMatrices':
        """
        Returns a new :class:`RecipeMatrices` instance by processing the information
        given as a :class:`RecipeGraph`.

        Args:
            recipe_graph (:class:`RecipeGraph`): recipe graph to process.
            compiled (:class:`bool`, optional): whether or not to compile the final
            matrices. Defaults to True.

        Returns:
            :class:`RecipeMatrices`: new recipe matrices instance.
        """

        recipe_matrices = cls()
        recipe_matrices.process_recipe_graph(recipe_graph)

        if compiled:
            recipe_matrices.compile()

        return recipe_matrices

    def __init__(self):
        """
        Initializes a new instance of the :class:`RecipeMatrices` class containing
        the new empty matrices to handle.
        """

        ## Action groups
        self._action_groups = groups.copy()
        self._mixing_actions = {action for action, group in self._action_groups.items() if group == 'mix'}

        ## Handled matrices
        self._actions_ingredients = ActionsIngredientsMatrix()
        self._ingredients_ingredients = MixedIngredientsMatrix()
        self._actions_base_ingredients = ActionsIngredientsMatrix()
        self._base_ingredients_base_ingredients = MixedIngredientsMatrix()
        self._group_actions_ingredients = ActionsIngredientsMatrix()
        self._group_actions_base_ingredients = ActionsIngredientsMatrix()
        self._group_actions_tools = ActionsToolsMatrix()

    @property
    def actions_ingredients(self) -> ActionsIngredientsMatrix:
        return self._actions_ingredients
    
    @property
    def ingredients_ingredients(self) -> MixedIngredientsMatrix:
        return self._ingredients_ingredients
    
    @property
    def actions_base_ingredients(self) -> ActionsIngredientsMatrix:
        return self._actions_base_ingredients
    
    @property
    def base_ingredients_base_ingredients(self) -> MixedIngredientsMatrix:
        return self._base_ingredients_base_ingredients
    
    @property
    def group_actions_ingredients(self) -> ActionsIngredientsMatrix:
        return self._group_actions_ingredients
    
    @property
    def group_actions_base_ingredients(self) -> ActionsIngredientsMatrix:
        return self._group_actions_base_ingredients
    
    @property
    def group_actions_tools(self) -> ActionsToolsMatrix:
        return self._group_actions_tools

    def process_recipe_graph(self, recipe_graph:RecipeGraph) -> None:
        """
        Processes the given recipe graph using the extracted information to
        update the relational matrices handled by this instance.

        Args:
            recipe_graph (RecipeGraph): recipe graph to process.
        """

        ## Clear actions list of ingredients
        recipe_graph.remove_actions_from_ingredients()

        ## DFS graph traversing
        nodes_stack = [[(recipe_graph.get_root_index(), set())]] # assumes root is an Action

        while nodes_stack:
            actions_sequence = nodes_stack.pop()
            current_action = actions_sequence[-1]
            node_index, action_ingredients = current_action

            ## Extract node information
            node = recipe_graph.get_node(node_index)
            sequence_action:Action = node['object']
            action_group = sequence_action.group if sequence_action.group else sequence_action.action
            is_mixing_action = sequence_action.group == 'mix' or sequence_action in self._mixing_actions
            actions_sequence[-1] = (sequence_action, is_mixing_action, action_ingredients)
            children_indices = recipe_graph.get_children_indices(node_index)
            
            ## Extract linked ingredients and actions
            linked_actions = []
            
            for child_index in children_indices:
                child_node = recipe_graph.get_node(child_index)

                if child_node['type'] == 'Action':
                    linked_actions.append(child_index)

                elif child_node['type'] == 'Ingredient':
                    ingredient = child_node['object']
                    action_ingredients.add(ingredient)

                elif child_node['type'] == 'Tool':
                    ## Add entry in action-tool matrix
                    tool = child_node['object']
                    self._group_actions_tools.add_entry(action_group, tool.full_object)

            ## Update matrices
            ### Add action labels
            self._actions_ingredients.label_to_row_index(sequence_action.action)
            self._actions_base_ingredients.label_to_row_index(sequence_action.action)
            self._group_actions_ingredients.label_to_row_index(action_group)
            self._group_actions_base_ingredients.label_to_row_index(action_group)
            self._group_actions_tools.label_to_row_index(action_group)

            ### Apply all actions to current ingredients
            for sequence_action, is_mixing, sequence_action_ingredients in actions_sequence[::-1]:
                ## Update matrices
                for ingredient in action_ingredients:
                    ## Add ingredient labels
                    self._actions_ingredients.label_to_column_index(ingredient.full_object)
                    self._ingredients_ingredients.label_to_index(ingredient.full_object)
                    self._actions_base_ingredients.label_to_column_index(ingredient.base_object)
                    self._base_ingredients_base_ingredients.label_to_index(ingredient.base_object)
                    self._group_actions_ingredients.label_to_column_index(ingredient.full_object)
                    self._group_actions_base_ingredients.label_to_column_index(ingredient.base_object)

                    ## Add values to matrices
                    action_group = sequence_action.group if sequence_action.group else sequence_action.action
                    self._actions_ingredients.add_entry(sequence_action.action, ingredient.full_object)
                    self._actions_base_ingredients.add_entry(sequence_action.action, ingredient.base_object)
                    self._group_actions_ingredients.add_entry(action_group, ingredient.full_object)
                    self._group_actions_base_ingredients.add_entry(action_group, ingredient.base_object)

                ### Mix current ingredients with previous ingredients (if necessary)
                if is_mixing and action_ingredients:
                    sequence_action_ingredients = [Ingredient(ing.base_object, ing.adjectives) for ing in sequence_action_ingredients]
                    for ingredients_pair in itertools.product(sequence_action_ingredients, action_ingredients):
                        self._ingredients_ingredients.add_entry(ingredients_pair[0].full_object, ingredients_pair[1].full_object)
                        self._base_ingredients_base_ingredients.add_entry(ingredients_pair[0].base_object, ingredients_pair[1].base_object)

                ## Apply action
                for ingredient in action_ingredients:
                    ingredient.add_applied_action(sequence_action)

            ## Add new actions to queue
            for linked_action in linked_actions:
                nodes_stack.append(actions_sequence + [(linked_action, set())])

    def compile(self) -> None:
        """
        Compiles all the handled relation matrices. (Note that the compiled collection
        can still be updated by processing new recipe graphs, but the new values
        cannot be accessed unless the matrices are compiled again)
        """

        self._actions_ingredients.compile()
        self._ingredients_ingredients.compile()
        self._actions_base_ingredients.compile()
        self._base_ingredients_base_ingredients.compile()
        self._group_actions_ingredients.compile()
        self._group_actions_base_ingredients.compile()
        self._group_actions_tools.compile()

    def save(self, collection_name:str) -> None:
        """
        Saves the relation matrices compising this collection in an archive folder
        with the given name.
        (Stored files follow a strict naming convention related to the class
        definition)

        Args:
            collection_name (:class:`str`): name to assign to the archive folder.
        """

        collection_filename = collection_name + '.zip'
        names_matrices:Dict[str, AdjacencyMatrix] = {
            'actions_ingredients': self.actions_ingredients,
            'ingredients_ingredients': self.ingredients_ingredients,
            'actions_base_ingredients': self.actions_base_ingredients,
            'base_ingredients_base_ingredients': self.base_ingredients_base_ingredients,
            'group_actions_ingredients': self.group_actions_ingredients,
            'group_actions_base_ingredients': self.group_actions_base_ingredients,
            'group_actions_tools': self.group_actions_tools
        }
        
        with zipfile.ZipFile(collection_filename, 'w', zipfile.ZIP_DEFLATED) as collection_zip:
            for name, matrix in names_matrices.items():
                matrix.save_to_files(f'{name}.npz', f'{name}.json')
                collection_zip.write(f'{name}.npz')
                collection_zip.write(f'{name}.json')
                os.remove(f'{name}.npz')
                os.remove(f'{name}.json')

    @classmethod
    def load(cls, collection_name:str) -> 'RecipeMatrices':
        """
        Returns a new :class:`RecipeMatrices` instance by loading a collection
        archive storing the information of the handled matrices.

        Args:
            collection_name (:class:`str`): name of the archive folder.

        Returns:
            :class:`RecipeMatrices`: new recipes matrices instance.
        """

        collection_filename = collection_name
        if not collection_name.endswith('.zip'):
            collection_filename = collection_name + '.zip'

        matrices_names_types = {
            'actions_ingredients': ActionsIngredientsMatrix,
            'ingredients_ingredients': MixedIngredientsMatrix,
            'actions_base_ingredients': ActionsIngredientsMatrix,
            'base_ingredients_base_ingredients': MixedIngredientsMatrix,
            'group_actions_ingredients': ActionsIngredientsMatrix,
            'group_actions_base_ingredients': ActionsIngredientsMatrix,
            'group_actions_tools': ActionsToolsMatrix
        }
        recipe_matrices = cls()

        with zipfile.ZipFile(collection_filename, 'r') as collection_zip:
            for matrix_name, matrix_type in matrices_names_types.items():
                collection_zip.extract(f'{matrix_name}.npz')
                collection_zip.extract(f'{matrix_name}.json')
                setattr(recipe_matrices, f'_{matrix_name}', matrix_type.load_from_files(f'{matrix_name}.npz', f'{matrix_name}.json'))
                os.remove(f'{matrix_name}.npz')
                os.remove(f'{matrix_name}.json')

        return recipe_matrices
