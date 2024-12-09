import spacy
import nltk
import torch
from pathlib import Path
from spacy.displacy import render
from spacy.language import Language
import spacy.tokens
from transformers import pipeline
from typing import Tuple, List, Any, Optional, Dict, Set

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
        self._language_model = spacy.load(self._language_model_name)
        
        @Language.component('custom_sentence_end_semicolon')
        def custom_sentence_end_semicolon(document):
            for token in document[:-1]:
                if token.text == ';':
                    document[token.i+1].is_sent_start = True

            return document
        
        self._language_model.add_pipe('custom_sentence_end_semicolon', before='parser')

        ## Additional processing parameters
        self._ignored_prepositions = ignored_prepositions.copy()
        self._additional_prepositions = additional_prepositions.copy()
        self._ignored_verbs = ignored_verbs.copy()
        self._additional_verbs = additional_verbs.copy()

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
    
    def process_instructions(self, instructions:List[str]) -> List[Tuple[List[str], List[Set[str]], List[Set[str]], List[List[Tuple[str, List[str], List[str]]]]]]:
        """
        Calls :func:`process_instruction` on each instruction contained in the
        given list, returning a list of processed instructions.

        Args:
            instructions (List[str]): list of strings representing the instructions
            to process.

        Returns:
            List[Tuple[List[str], List[Set[str]], List[Set[str]], List[List[Tuple[str, List[str], List[str]]]]]]:
            a list containing the groups of processed information, a group for each instruction.
        """

        return [self.process_instruction(instruction) for instruction in instructions]

    def process_instruction(self, instruction:str) -> Tuple[List[str], List[Set[str]], List[Set[str]], List[List[Tuple[str, List[str], List[str]]]]]:
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
            Tuple[List[str], List[Set[str]], List[Set[str]], List[List[Tuple[str, List[str], List[str]]]]]:
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

                    ## Check if verb is a valid action else skip
                    overlapping_action = self._token_entities_overlaps(root, step_action_entities)
                    if not (overlapping_action or root.lemma_ in self._additional_verbs) or root.lemma_ in self._ignored_verbs:
                        continue

                    ## Find sentence components
                    action_text = overlapping_action['word']
                    action_primary_objects = []
                    action_secondary_objects = []

                    for child_node in root.children:
                        ## Add sentence primary objects
                        if child_node.dep_ == 'dobj':
                            objects = [child_node]

                            while objects:
                                current_object = objects.pop(0)

                                overlapping_ingredient = self._token_entities_overlaps(current_object, step_ingredient_entities)
                                overlapping_tool = self._token_entities_overlaps(current_object, step_tool_entities)
                                if overlapping_ingredient:
                                    step_ingredients.add(overlapping_ingredient['word'])
                                    action_primary_objects.append(overlapping_ingredient['word'])
                                elif overlapping_tool:
                                    step_tools.add(overlapping_tool['word'])
                                    action_primary_objects.append(overlapping_tool['word'])

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

                                        overlapping_ingredient = self._token_entities_overlaps(current_object, step_ingredient_entities)
                                        overlapping_tool = self._token_entities_overlaps(current_object, step_tool_entities)
                                        if overlapping_ingredient:
                                            step_ingredients.add(overlapping_ingredient['word'])
                                            action_secondary_objects.append(overlapping_ingredient['word'])
                                        elif overlapping_tool:
                                            step_tools.add(overlapping_tool['word'])
                                            action_secondary_objects.append(overlapping_tool['word'])
                                        else:
                                            action_secondary_objects.append(current_object.text)

                                        for sub_child_node in current_object.children:
                                            if sub_child_node.dep_ == 'conj':
                                                objects.append(sub_child_node)

                        ## Add connected sentences
                        elif child_node.dep_ == 'conj' or child_node.dep_ == 'dep':
                            if child_node.pos_ == 'VERB':
                                processing_queue.append(child_node)

                    ## Add sentence to step list
                    step_actions.append((action_text, action_primary_objects, action_secondary_objects))

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
