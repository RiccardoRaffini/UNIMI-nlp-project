import re
from abc import ABC, abstractmethod
from typing import Tuple, List, TypeVar, Generic

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

        def dfs_graph(node_index):
            objects = {'primary': [], 'secondary': [], 'tools': []}

            for child_index in raw_output.get_children_indices(node_index):
                child_node = dfs_graph(child_index)
                edge_type = raw_output._graph.edges[node_index, child_index]['type']

                if type(child_node) == Tool:
                    objects['tools'].append(child_node.base_object)
                elif type(child_node) == Action:
                    #objects[edge_type].append('result of previous instruction')
                    pass
                else:
                    objects[edge_type].append(child_node.base_object)

            node = raw_output.get_node(node_index)['object']
            if type(node) == Action:
                if not objects['primary'] and not objects['secondary'] and not objects['tools'] and instructions_sequence:
                    instructions_sequence[-1] += f'; {node.action} it'

                else:
                    if len(objects['primary']) > 1:
                        primary_text = ', '.join(objects['primary'][:-1]) + f' and {objects["primary"][-1]}'
                    else:
                        primary_text = ', '.join(objects['primary'])

                    secondary_text = ''
                    if len(objects['secondary']) > 0:
                        prefix_text = 'it with ' if not primary_text else ' with '
                        secondary_text = prefix_text + ', '.join(objects['secondary'])

                    tools_text = ''
                    if len(objects['tools']) > 0:
                        if not primary_text and not secondary_text:
                            prefix_text = 'it using ' if instructions_sequence else ''
                        else:
                            prefix_text = ' using '

                        tools_text = prefix_text + ', '.join(objects['tools'])

                    instruction_text = f'{node.action} {primary_text}{secondary_text}{tools_text}'
                    instructions_sequence.append(instruction_text)

            return node

        dfs_graph(raw_output._root)
        instructions = instructions_sequence[::-1]

        return instructions

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
