import networkx as nx
import pandas as pd
import pygraphviz
import re
from functools import reduce
from typing import List, Tuple, Dict, Any

from commons.nlp_utils import RecipeProcessor

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

    processor:RecipeProcessor = None

    @classmethod
    def set_recipe_processor(cls, processor:RecipeProcessor) -> None:
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

        ## Processed fields
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

        assert self.processor is not None, 'cannot process a recipe without a processor'

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

        return recipe_graph

    def __init__(self, additional_configuration:Dict[str, Any] = None):
        ## Underlying graph
        self._graph = nx.Graph()

        ## Graph configuration
        self._graph_configuration = {
            'node_attributes': {
                'ingredient': {'fillcolor': '#ffe070', 'shape': 'ellipse'},
                'tool': {'fillcolor': '#c6f7a6', 'shape': 'ellipse'},
                'action': {'fillcolor': '#bcd9f5', 'shape': 'box'},
                'misc': {'fillcolor': '#f0f0f0', 'shape': 'circle'},
            }
        }

        if additional_configuration is not None:
            self._graph_configuration = self._graph_configuration | additional_configuration

    def add_ingredient_node(self, ingredient_name:str) -> int:
        node_index = len(self._graph.nodes)
        return self.add_recipe_node(node_index, ingredient_name, 'ingredient')

    def add_tool_node(self, tool_name:str) -> int:
        node_index = len(self._graph.nodes)
        return self.add_recipe_node(node_index, tool_name, 'tool')

    def add_action_node(self, action_text:str, primary_objects_nodes:List[int], secondary_objects_nodes:List[str] = None) -> int:
        node_index = len(self._graph.nodes)
        self.add_recipe_node(node_index, action_text, 'action')

        for primary_index in primary_objects_nodes:
            self.add_generic_edge(node_index, primary_index, {'type': 'primary'})

        for secondary_index in secondary_objects_nodes:
            self.add_generic_edge(node_index, secondary_index, {'type': 'secondary'})
        
        return node_index
    
    def add_misc_node(self, text:str) -> int:
        node_index = len(self._graph.nodes)
        return self.add_recipe_node(node_index, text, 'misc')

    def add_recipe_node(self, index:int, label:str, type:str) -> int:
        node_attributes = {'label': label}
        node_attributes.update(self._graph_configuration['node_attributes'][type])

        self._graph.add_node(index, **node_attributes)
        return index

    def add_generic_edge(self, node_1:int, node_2:int, attributes:Dict[str, Any] = {}) -> None:
        self._graph.add_edge(node_1, node_2, **attributes)

    def to_gviz(self, filename:str) -> None:
        agraph:pygraphviz.AGraph = nx.nx_agraph.to_agraph(self._graph)

        agraph.node_attr['style'] = 'rounded,filled'
        agraph.graph_attr['rankdir'] = 'BT'

        for node in agraph.iternodes():
            node.attr['label'] = f"<{re.sub(' ', '<BR />', node.attr['label'])}>"

        agraph.layout(prog='dot')
        agraph.draw(filename)
        #agraph.write('recipe_graph.dot')

    def __str__(self):
        pass

    def __repr__(self):
        pass
