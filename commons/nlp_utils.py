from __future__ import annotations

import itertools
import spacy, spacy.tokens
import nltk
import torch
from pathlib import Path
from spacy.displacy import render
from spacy.language import Language
from transformers import pipeline
from typing import Tuple, List, Any, Optional, Dict, Set, Union

import commons.recipes as recipes
from commons.action_groups import groups
from commons.matrices import AdjacencyMatrix, ActionsIngredientsMatrix, MixedIngredientsMatrix, ActionsToolsMatrix

LANGUAGE_MODEL_NAME = 'en_core_web_trf' # 'en_core_web_trf' 'en_core_web_lg' 'en_core_web_sm' 'it_core_news_lg' 'it_core_news_sm'
FOOD_MODEL_PATH = 'models/bert-finetuned-food-ner'

class RecipeProcessor():
    """
    A class which allows to process recipes textual instructions and return them
    in a list of list format, easier to further process and which highlights the
    foods, tools, cooking actions in the texts and how they are related.
    """

    def __init__(self,
        food_model_path:str = FOOD_MODEL_PATH,
        language_model_name:str=LANGUAGE_MODEL_NAME,
        ignored_prepositions:Set[str] = set(),
        additional_prepositions:Set[str] = set(),
        ignored_verbs:Set[str] = set(),
        additional_verbs:Set[str] = set(),
        ignored_objects:Set[str] = set(),
        additional_objects:Set[str] = set(),
        device:str=('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        """
        Initializes a new instance of :class:`RecipeProcessor` with all the
        required attributes.

        Args:
            **food_model_path** (str, optional): path to the folder where the food
            NER model is located. Defaults to FOOD_MODEL_PATH.

            **language_model_name** (str, optional): name of the package or name
            model that should be loaded by :module:`spacy` as base language
            model. Defaults to LANGUAGE_MODEL_NAME.

            **ignored_prepositions** (Set[str], optional): set of strings representing
            english prepositions that should be ignored during instructions
            processing. Defaults to empty set.

            **additional_prepositions** (Set[str], optional): set of strings
            representing english words that should also be considered as
            preposition during instructions processing. Defaults to empty set.

            **ignored_verbs** (Set[str], optional): set of strings representing english
            verbs that should be ignored during instructions processing. Defaults
            to empty set.

            **additional_verbs** (Set[str], optional): set of strings representing
            english words that should also be considered during instructions
            processing. Defaults to empty set.

            **ignored_objects** (Set[str], optional): set of strings representing
            objects or terms that should be ignored during instructions processing.
            Defaults to empty set.

            **additional_objects** (Set[str], optional): set of strings representing
            objects or terms that should also be considered during instructions
            processing. Defaults to empty set.

            **device** (str, optional): type or name of device on which the food
            model shoul be executed. Defaults to 'cuda' if a torch-capable GPU
            device is available.
        """
        
        ## Specialized food tokens classifier
        self._food_model_path = food_model_path
        self._food_model = pipeline(
            task='token-classification',
            model=self._food_model_path,
            aggregation_strategy='simple',
            device=device
        )

        ## Generic language model
        self._language_model_name = language_model_name
        self._language_model = spacy.load(self._language_model_name, disable=['ner', 'textcat'])
        self._lemmatizer = spacy.load(self._language_model_name, disable=['parser', 'ner', 'textcat'])
        
        @Language.component('custom_sentence_end_semicolon')
        def custom_sentence_end_semicolon(document):
            for token in document[:-1]:
                if token.text == ';':
                    document[token.i+1].is_sent_start = True

            return document
        
        self._language_model.add_pipe('custom_sentence_end_semicolon', before='parser')
        spacy.tokens.Token.set_extension('further_processing', default=True)

        ## Additional processing parameters
        self._ignored_prepositions = ignored_prepositions.copy()
        self._additional_prepositions = additional_prepositions.copy()
        self._ignored_verbs = ignored_verbs.copy()
        self._additional_verbs = additional_verbs.copy()
        self._ignored_objects = ignored_objects.copy()
        self._additional_objects = additional_objects.copy()
        self._action_groups = groups.copy()
        self._mixing_actions = {action for action, group in self._action_groups.items() if group == 'mix'}

    def _split_instruction_in_steps(self, instruction:str) -> List[str]:
        """
        Splits a textual instruction in smaller sentences, namely _steps_.

        Args:
            **instruction** (str): string representing the instruction to process.

        Returns:
            List[str]: list of steps that compose the original instruction.
        """

        return nltk.sent_tokenize(instruction)
    
    def _token_entities_overlaps(self, token:spacy.tokens.Token, entities:List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Checks whether or not the given text token overlaps (completely or
        paritially) with any of the elements in the given list of entities.
        (Tokens and entities should be extracted from the same text)

        Args:
            **token** (Token): token to compare.
            **entities** (List[Dict[str, Any]]): list of dictionaries containing
            information regarding the entities.

        Returns:
            Optional[Dict[str, Any]]: the dictionary of the first overlapping
            entity or `None` if no overlapping entity is found. 
        """

        token_start, token_end = token.idx, token.idx+len(token)

        for entity in entities:
            entity_start, entity_end = entity['start'], entity['end']
            if max(token_start, entity_start) < min(token_end, entity_end):
                return entity
            
        return None
    
    def process_object(self, object_tree:spacy.tokens.Token, valid_ingredients:List[Dict[str, Any]], valid_tools:List[Dict[str, Any]]) \
        -> Optional[recipes.RecipeObject]:
        """
        Processes a generic sequence (tree) of tokens possibly representing a
        recipe object, using the provided collections of items as reference to
        find the main object noun and all its adjectives.

        Returns:
            Optional[recipes.RecipeObject]: the recipe object found or `None`
            if no recipe object is found.
        """

        ## Find noun and adjectives
        noun = object_tree
        adjectives = []

        for child_node in object_tree.children:
            if child_node.dep_ == 'amod':
                adjectives.append(child_node.text)

            elif child_node.dep_ == 'prep':
                return self.process_object(next(child_node.children), valid_ingredients, valid_tools)

        ## Check valid object
        overlapping_ingredient = self._token_entities_overlaps(noun, valid_ingredients)
        overlapping_tool = self._token_entities_overlaps(noun, valid_tools)

        if overlapping_ingredient and overlapping_ingredient['word'] not in self._ignored_objects:
            ## Create ingredient
            adjectives = [adjective for adjective in adjectives if adjective not in overlapping_ingredient['word'].split()]
            name = ' '.join([token.lemma_ for token in self._lemmatizer(overlapping_ingredient['word'])])
            return recipes.Ingredient(name, adjectives)
        
        elif overlapping_tool and overlapping_tool['word'] not in self._ignored_objects:
            ## Create tool
            adjectives = [adjective for adjective in adjectives if adjective not in overlapping_tool['word'].split()]
            name = ' '.join([token.lemma_ for token in self._lemmatizer(overlapping_tool['word'])])
            return recipes.Tool(name, adjectives)

        elif noun.text in self._additional_objects:
            ## Create misc object
            return recipes.Miscellaneous(noun.text, adjectives)

        ## No valid object found
        return None
    
    def process_instructions(self, instructions:List[str]) \
        -> List[Tuple[
            List[str],
            List[Set[recipes.RecipeObject]],
            List[Set[recipes.RecipeObject]],
            List[List[Tuple[recipes.Action, List[recipes.RecipeObject], List[recipes.RecipeObject]]]]
        ]]:
        """
        Calls :func:`process_instruction` on each instruction contained in the
        given list, returning a list of processed instructions.

        Args:
            instructions (List[str]): list of strings representing the instructions
            to process.

        Returns:
            List[Tuple[
                List[str],
                List[Set[recipes.RecipeObject]],
                List[Set[recipes.RecipeObject]],
                List[List[Tuple[recipe.Action, List[recipes.RecipeObject], List[recipes.RecipeObject]]]]
            ]]:
            a list containing the groups of processed information, a group for each instruction.
        """

        return [self.process_instruction(instruction) for instruction in instructions]

    def process_instruction(self, instruction:str) \
        -> Tuple[
            List[str],
            List[Set[recipes.RecipeObject]],
            List[Set[recipes.RecipeObject]],
            List[List[Tuple[recipes.Action, List[recipes.RecipeObject], List[recipes.RecipeObject]]]]
        ]:
        """
        Processes a string representing a *recipe instruction*, splitting it in
        *steps* and subsequent *sentences* in them.
        Each sentence is then traversed in order to extract valid cooking actions
        and find ingredients/tools meaningful to such actions.

        This returns four lists containing the information extracted, grouped by step:
        - List of steps text
        - List of steps ingredients
        - List of steps tools
        - List of steps actions

        Args:
            **instruction** (str): string representing the instruction to process.

        Returns:
            Tuple[
                List[str],
                List[Set[recipes.RecipeObject]],
                List[Set[recipes.RecipeObject]],
                List[List[Tuple[recipes.Action, List[recipes.RecipeObject], List[recipes.RecipeObject]]]]
            ]:
            four list of texts, ingredients, tools and actions.
        """

        instruction_steps = []
        instruction_ingredients = []
        instruction_tools = []
        instruction_actions = []

        ## Split instruction in steps
        instruction = instruction.lower()
        instruction_steps = self._split_instruction_in_steps(instruction)

        ## Process steps
        for step_text in instruction_steps:
            ## Annotate step
            step_annotation = self._language_model(step_text)

            ## Find entities in step
            step_food_entities = self._food_model(step_text)
            step_ingredient_entities = [entity for entity in step_food_entities if entity['entity_group'] == 'FOOD']
            step_tool_entities = [entity for entity in step_food_entities if entity['entity_group'] == 'TOOL']
            step_action_entities = [entity for entity in step_food_entities if entity['entity_group'] == 'ACTION']

            step_ingredients = set()
            step_tools = set()
            step_actions = []

            ## Traverse sentence-trees in step
            for sentence in step_annotation.sents:
                processing_queue = []
                root = sentence.root

                if root.pos_ == 'VERB':
                    processing_queue.append(root)

                while processing_queue:
                    root:spacy.tokens.Token = processing_queue.pop(0)
                    skip_adding = False

                    ## Check if verb is a valid action otherwise skip
                    overlapping_action = self._token_entities_overlaps(root, step_action_entities)
                    if not (overlapping_action or root.lemma_ in self._additional_verbs) or root.lemma_ in self._ignored_verbs:
                        continue

                    ## Find sentence components
                    action_text = ' '.join([token.lemma_ for token in self._lemmatizer(overlapping_action['word'])])
                    action_group = self._action_groups[action_text] if action_text in self._action_groups else None
                    action = recipes.Action(action_text, action_group)
                    action_primary_objects = []
                    action_secondary_objects = []

                    for child_node in root.children:
                        ## Add sentence primary objects
                        if child_node.dep_ == 'dobj':
                            objects = [child_node]

                            while objects:
                                current_object = objects.pop(0)
                                object = self.process_object(current_object, step_ingredient_entities, step_tool_entities)

                                if isinstance(object, recipes.Ingredient):
                                    step_ingredients.add(object)
                                    action_primary_objects.append(object)
                                elif isinstance(object, recipes.Tool):
                                    step_tools.add(object)
                                    action_primary_objects.append(object)
                                elif isinstance(object, recipes.Miscellaneous):
                                    action_primary_objects.append(object)

                                ## Find chained objects
                                for sub_child_node in current_object.children:
                                    if sub_child_node.dep_ == 'conj':
                                        objects.append(sub_child_node)

                        ## Add sentence secondary objects
                        elif (child_node.dep_ == 'prep' or child_node.lemma_ in self._additional_prepositions) and child_node.lemma_ not in self._ignored_prepositions:
                            for sub_child_node in child_node.children:
                                if sub_child_node.dep_ == 'pobj':
                                    objects = [sub_child_node]

                                    while objects:
                                        current_object = objects.pop(0)
                                        object = self.process_object(current_object, step_ingredient_entities, step_tool_entities)

                                        if isinstance(object, recipes.Ingredient):
                                            step_ingredients.add(object)
                                            action_secondary_objects.append(object)
                                        elif isinstance(object, recipes.Tool):
                                            step_tools.add(object)
                                            action_secondary_objects.append(object)
                                        elif isinstance(object, recipes.Miscellaneous):
                                            action_secondary_objects.append(object)

                                        for sub_child_node in current_object.children:
                                            if sub_child_node.dep_ == 'conj':
                                                objects.append(sub_child_node)

                        ## Add connected sentences
                        elif child_node.dep_ in {'conj', 'dep'} and root._.further_processing:
                            if child_node.pos_ == 'VERB':
                                processing_queue.append(child_node)

                        ## Handle sub actions
                        elif child_node.dep_ == 'advcl' and root._.further_processing:
                            if child_node.pos_ == 'VERB':
                                processing_queue.insert(0, root)
                                processing_queue.insert(0, child_node)

                                skip_adding = True

                    if skip_adding:
                        root._.further_processing = False
                        continue

                    ## Add sentence to step list
                    step_actions.append((action, action_primary_objects, action_secondary_objects))

            ## Add step list to instruction list
            instruction_ingredients.append(step_ingredients)
            instruction_tools.append(step_tools)
            instruction_actions.append(step_actions)

        return instruction_steps, instruction_ingredients, instruction_tools, instruction_actions
    
    def save_annotation_render(self, instruction:str, filename:str='annotation.svg') -> None:
        """
        Creates a graphical visualization of the dependencies (POS and NER)
        occurring in the given textual instruction.

        Args:
            **instruction** (str): string representing the instruction to process.
            **filename** (str, optional): filename where to save the visualization.
            Defaults to 'annotation.svg'.
        """

        instruction_annotations = self._language_model(instruction)
        svg_image = render(instruction_annotations, jupyter=False)

        save_path = Path(filename)
        with save_path.open('w', encoding='utf-8') as file:
            file.write(svg_image)

    def process_matrices(self, recipe_graph:recipes.RecipeGraph) -> List[AdjacencyMatrix]:
        ## Define empty matrices
        actions_ingredients_matrix = ActionsIngredientsMatrix()
        ingredients_ingredients_matrix = MixedIngredientsMatrix()
        actions_base_ingredients_matrix = ActionsIngredientsMatrix()
        base_ingredients_base_ingredients = MixedIngredientsMatrix()
        group_actions_ingredients = ActionsIngredientsMatrix()
        group_actions_base_ingredients = ActionsIngredientsMatrix()
        group_actions_tools = ActionsToolsMatrix()
        
        ## DFS graph traversing
        nodes_stack = [[(recipe_graph.get_root_index(), set())]] # assumes root is an Action

        while nodes_stack:
            actions_sequence = nodes_stack.pop()
            current_action = actions_sequence[-1]
            node_index, action_ingredients = current_action

            ## Extract node information
            node = recipe_graph.get_node(node_index)
            action:recipes.Action = node['object']
            action_group = action.group if action.group else action.action
            is_mixing_action = action.group == 'mix' or action in self._mixing_actions
            actions_sequence[-1] = (action, is_mixing_action, action_ingredients)
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
                    group_actions_tools.add_entry(action_group, tool.full_object, 1)

            ## Update matrices
            ### Add action labels
            actions_ingredients_matrix.label_to_row_index(action.action)
            actions_base_ingredients_matrix.label_to_row_index(action.action)
            group_actions_ingredients.label_to_row_index(action_group)
            group_actions_base_ingredients.label_to_row_index(action_group)
            group_actions_tools.label_to_row_index(action_group)

            ### Add ingredient labels
            for ingredient in action_ingredients:
                actions_ingredients_matrix.label_to_column_index(ingredient.full_object)
                ingredients_ingredients_matrix.label_to_index(ingredient.full_object)
                actions_base_ingredients_matrix.label_to_column_index(ingredient.base_object)
                base_ingredients_base_ingredients.label_to_index(ingredient.base_object)
                group_actions_ingredients.label_to_column_index(ingredient.full_object)
                group_actions_base_ingredients.label_to_column_index(ingredient.base_object)

            ### Apply all actions to current ingredients
            for action, is_mixing, ingredients in actions_sequence:
                for ingredient in action_ingredients:
                    actions_ingredients_matrix.add_entry(action.action, ingredient.full_object, 1)
                    actions_base_ingredients_matrix.add_entry(action.action, ingredient.base_object, 1)
                    group_actions_ingredients.add_entry(action_group, ingredient.full_object, 1)
                    group_actions_base_ingredients.add_entry(action_group, ingredient.base_object, 1)

                ### Mix current ingredients with previous ingredients (if necessary)
                if is_mixing and action_ingredients:
                    for ingredients_pair in itertools.product(ingredients, action_ingredients):
                        ingredients_ingredients_matrix.add_entry(ingredients_pair[0].full_object, ingredients_pair[1].full_object, 1)
                        base_ingredients_base_ingredients.add_entry(ingredients_pair[0].base_object, ingredients_pair[1].base_object, 1)

            ## Add new actions to queue
            for linked_action in linked_actions:
                nodes_stack.append(actions_sequence + [(linked_action, set())])

        ## Compile matrices
        matrices_list:List[AdjacencyMatrix] =  [
            actions_ingredients_matrix, ingredients_ingredients_matrix,
            actions_base_ingredients_matrix, base_ingredients_base_ingredients,
            group_actions_ingredients, group_actions_base_ingredients,
            group_actions_tools
        ]

        for matrix in matrices_list:
            matrix.compile()

        return matrices_list
