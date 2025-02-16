import sys
sys.path.append('.')

import functools
import pandas as pd
from sklearn.model_selection import train_test_split

from commons.data import load_foodcom_review_dataset
from commons.recipes import Recipe

RECIPE_START_TOKEN = '<RECIPE_START>'
RECIPE_END_TOKEN = '<RECIPE_END>'
INPUT_START_TOKEN = '<INPUT_START>'
INPUT_END_TOKEN = '<INPUT_END>'
INPUT_NEXT_TOKEN = '<INPUT_NEXT>'
INGREDIENT_START_TOKEN = '<INGREDIENT_START>'
INGREDIENT_END_TOKEN = '<INGREDIENT_END>'
INGREDIENT_NEXT_TOKEN = '<INGREDIENT_NEXT>'
INSTRUCTION_START = '<INSTRUCTION_START>'
INSTRUCTION_END = '<INSTRUCTION_END>'
INSTRUCTION_NEXT = '<INSTRUCTION_NEXT>'

def recipe_to_text_tokens(recipe:Recipe) -> str:
    recipe_ingredients = functools.reduce(lambda acc, ingrs: acc.union(ingrs), recipe.steps_ingredients, set())
    recipe_ingredients = set(map(lambda ingredient: ingredient.base_object, recipe_ingredients))

    recipe_instructions_sequence = []
    for steps in recipe.steps_actions:
        steps_texts = []
        for step in steps:
            action, ingredients, tools = step
            step_text = action.action
            if ingredients:
                ingredients = map(lambda ingredient: ingredient.base_object, ingredients)
                step_text += f' {", ".join(ingredients)}'
            if tools:
                tools = map(lambda tool: tool.base_object, tools)
                step_text += f' with {", ".join(tools)}'

            steps_texts.append(step_text)

        if len(steps_texts) == 0:
            continue
        elif len(steps_texts) > 1:
            instruction = '; '.join(steps_texts[:-1])
            instruction += f' and {steps_texts[-1]}'
        else:
            instruction = steps_texts[0]
        recipe_instructions_sequence.append(instruction)

    tokenized_text = f'{RECIPE_START_TOKEN} {INPUT_START_TOKEN} ' + \
        f' {INPUT_NEXT_TOKEN} '.join(recipe_ingredients) + \
        f' {INPUT_END_TOKEN} {INGREDIENT_START_TOKEN} ' + \
        f' {INGREDIENT_NEXT_TOKEN} '.join(recipe_ingredients) + \
        f' {INGREDIENT_END_TOKEN} {INSTRUCTION_START}' + \
        f' {INSTRUCTION_NEXT} '.join(recipe_instructions_sequence) + \
        f' {INSTRUCTION_END} {RECIPE_END_TOKEN}'
    
    return tokenized_text

def recipes_to_text_tokens_file(raw_recipes:pd.DataFrame, filename:str = 'recipes_tokenized.txt', verbose:bool = False) -> None:
    assert Recipe.processor is not None, 'a recipe processor must be set for Recipe class'

    if verbose: print('Writing recipes tokens to', filename)
    with open(filename, 'w') as tokens_file:
        for index, recipe_row in raw_recipes.iterrows():
            if verbose and index % 20_000 == 0: print('current index:', index)

            try:
                recipe = Recipe.from_dataframe_row(recipe_row)
                recipe_tokens = recipe_to_text_tokens(recipe)
                tokens_file.write(f'{recipe_tokens}\n')
            except:
                if verbose: print(f'exception on recipe at index {index}')

def raw_recipe_to_text_tokens(raw_recipe:pd.Series) -> str:
    tokenized_text = f'{RECIPE_START_TOKEN} {INPUT_START_TOKEN} ' + \
        f' {INPUT_NEXT_TOKEN} '.join(raw_recipe['Ingredients']) + \
        f' {INPUT_END_TOKEN} {INGREDIENT_START_TOKEN} ' + \
        f' {INGREDIENT_NEXT_TOKEN} '.join(raw_recipe['Ingredients']) + \
        f' {INGREDIENT_END_TOKEN} {INSTRUCTION_START}' + \
        f' {INSTRUCTION_NEXT} '.join(raw_recipe['Instructions']) + \
        f' {INSTRUCTION_END} {RECIPE_END_TOKEN}'
    
    return tokenized_text
    
def raw_recipes_to_text_tokens_file(raw_recipes:pd.DataFrame, filename:str = 'raw_recipes_tokenized.txt', verbose:bool = False) -> None:
    if verbose: print('Writing recipes tokens to', filename)
    with open(filename, 'w') as tokens_file:
        for index, recipe_row in raw_recipes.iterrows():
            if verbose and index % 20_000 == 0: print('current index:', index)

            try:
                recipe_tokens = raw_recipe_to_text_tokens(recipe_row)
                tokens_file.write(f'{recipe_tokens}\n')
            except:
                if verbose: print(f'exception on recipe at index {index}')

if __name__ == '__main__':
    ## Sample data
    recipes_dataset = load_foodcom_review_dataset()
    recipes_dataset = recipes_dataset.sample(frac=0.25)
    print(len(recipes_dataset))

    ## Split data
    recipes_train_dataset, recipes_test_dataset = train_test_split(recipes_dataset, test_size=0.05)
    recipes_train_dataset.reset_index(drop=True, inplace=True)
    recipes_test_dataset.reset_index(drop=True, inplace=True)
    print(len(recipes_train_dataset), len(recipes_test_dataset))

    ## Create tokens files
    raw_recipes_to_text_tokens_file(recipes_train_dataset, filename='recipes_train_text_tokens.txt', verbose=True)
    raw_recipes_to_text_tokens_file(recipes_test_dataset, filename='recipes_test_text_tokens.txt', verbose=True)
