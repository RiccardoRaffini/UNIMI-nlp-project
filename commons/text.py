import random
import re
from abc import ABC, abstractmethod
from typing import Tuple, List, TypeVar, Generic, Any

from commons.recipes import RecipeGraph, Ingredient, Action, Tool

T = TypeVar('T')

class RawOutputToText(ABC, Generic[T]):
    @abstractmethod
    def ingredients(self, raw_output:T) -> List[str]:
        raise NotImplementedError
    
    @abstractmethod
    def instructions(self, raw_output:T) -> List[str]:
        raise NotImplementedError

    def markdown(self, raw_output:T, ingredients:List[str] = None, instructions:List[str] = None) -> str:
        if ingredients is None:
            ingredients = self.ingredients(raw_output)

        if instructions is None:
            instructions = self.instructions(raw_output)

        markdown_text = \
            '### Ingredients:\n' + \
            '\n'.join(map(lambda ingredient: f'- {ingredient}', ingredients)) + \
            '\n### Instructions:\n' + \
            '\n'.join([f'{index+1}. {instruction}' for index, instruction in enumerate(instructions)])
        
        return markdown_text
    
    def __call__(self, raw_output:T) -> Tuple[List[str], List[str], str]:
        ingredients = self.ingredients(raw_output)
        instructions = self.instructions(raw_output)
        markdown = self.markdown(raw_output, ingredients, instructions)

        return ingredients, instructions, markdown

class RecipeGraphToText(RawOutputToText):
    def ingredients(self, raw_output:RecipeGraph) -> List[str]:
        ingredients = set()

        for node_index in raw_output.nodes():
            node = raw_output.get_node(node_index)
            if type(node['object']) == Ingredient:
                ingredients.add(node['object'].base_object)

        return list(ingredients)
    
    def instructions(self, raw_output:RecipeGraph) -> List[str]:
        instructions_sequence = []

        def dfs_instructions(node_index:int) -> Tuple[Any, List[str]]:
            node = raw_output.get_node(node_index)['object']
            children_indices = raw_output.get_children_indices(node_index)
            children_number = len(children_indices)
            instructions = []

            children_objects = {'primary': [], 'secondary': [], 'tools': []}
            for child_index in children_indices:
                edge_type = raw_output._graph.edges[node_index, child_index]['type']
                child_node, child_instructions = dfs_instructions(child_index)

                if type(child_node) == Tool:
                    children_objects['tools'].append(child_node.base_object)
                elif type(child_node) == Action:
                    pass
                else:
                    children_objects[edge_type].append(child_node.base_object)

                instructions.extend(child_instructions)

            if type(node) == Action:
                if not children_objects['primary'] and not children_objects['secondary'] and not children_objects['tools'] and instructions:
                    if children_number > 1:
                        objects = random.choice(['them', 'the results of previous steps'])
                        instructions.append(f'{node.action} {objects}')
                    elif children_number == 1:
                        link = random.choice([';', ' and', ' then', ' and then'])
                        instructions[-1] += f'{link} {node.action} it'

                else:
                    primary_text = ''
                    if len(children_objects['primary']) > 1:
                        primary_text = ', '.join(children_objects['primary'][:-1]) + f' and {children_objects["primary"][-1]}'
                    elif len(children_objects['primary']) == 1:
                        weights = [0.3, 0.3, 0.19, 0.01, 0.2]
                        if children_objects["primary"][0] in {'a', 'e', 'i', 'o', 'u'}:
                            weights = [0.3, 0.3, 0.05, 0.3, 0.05]
                        prefix_text = random.choices(['', 'the ', 'a ', 'an ', 'some '], weights)[0]

                        primary_text = f'{prefix_text}{children_objects["primary"][0]}'

                    secondary_text = ''
                    if len(children_objects['secondary']) > 0:
                        prefix_text = 'it with ' if not primary_text else ' with '
                        secondary_text = prefix_text + ', '.join(children_objects['secondary'])

                    tools_text = ''
                    if len(children_objects['tools']) > 0:
                        if not primary_text and not secondary_text:
                            prefix_text = 'it using ' if instructions_sequence else ''
                        else:
                            prefix_text = ' using '

                        tools_text = prefix_text + ', '.join(children_objects['tools'])

                    instruction = f'{node.action} {primary_text}{secondary_text}{tools_text}'
                    instructions.append(instruction)

            return node, instructions

        _, instructions_sequence = dfs_instructions(raw_output._root)

        return instructions_sequence

class TokensSequenceToText(RawOutputToText):
    def ingredients(self, raw_output:str) -> List[str]:
        ingredients_start_index = re.search(' <INGREDIENT_START> ', raw_output).end()
        ingredients_end_index = re.search(' <INGREDIENT_END> ', raw_output).start()
        ingredient_raw_output = raw_output[ingredients_start_index:ingredients_end_index]

        ingredients = ingredient_raw_output.split(' <INGREDIENT_NEXT> ')
        ingredients = list(map(lambda s: s.strip(), ingredients))

        return ingredients

    def instructions(self, raw_output:str) -> List[str]:
        instructions_start_index = re.search(' <INSTRUCTION_START> ', raw_output).end()
        instructions_end_index = re.search(' <INSTRUCTION_END> ', raw_output).start()
        instructions_raw_output = raw_output[instructions_start_index:instructions_end_index]

        instructions = instructions_raw_output.split(' <INSTRUCTION_NEXT> ')
        instructions = list(map(lambda s: s.strip(), instructions))

        return instructions
