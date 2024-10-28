import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import dataframe_information, field_string_to_list, field_time_to_string, field_ingredients_quantities, dictionary_time_to_string

FOODCOM_RECIPES_DATASET_LOCATION = './datasets/food_com_recipes_review/recipes.csv'
FOODCOM_REVIEWS_DATASET_LOCATION = './datasets/food_com_recipes_review/reviews.csv'
FOODCOM_TAGS_DATASET_LOCATION = './datasets/food_com_tags/recipes_ingredients.csv'
RECIPENLG_DATASET_LOCATION = './datasets/recipenlg/full_dataset.csv'
RECIPES3K_DATASET_LOCATION = './datasets/recipes3k/*.json'


def load_foodcom_review_dataset(
    recipes_filename:str = FOODCOM_RECIPES_DATASET_LOCATION,
    reviews_filename:str = FOODCOM_REVIEWS_DATASET_LOCATION
) -> pd.DataFrame:
    ## Recipes data
    recipe_columns = [
        'RecipeId', 'Name', 'TotalTime', 'Description',
        'RecipeCategory', 'RecipeIngredientQuantities', 'RecipeIngredientParts',
        'RecipeInstructions',

        'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
        'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
    ]
    recipe_columns_rename = {
        'RecipeId': 'Id', 'TotalTime': 'Time', 'RecipeCategory': 'Category',
        'RecipeIngredientQuantities': 'IngredientQuantities',
        'RecipeIngredientParts': 'Ingredients', 'RecipeInstructions': 'Instructions'
    }
    recipes_dataset = pd.read_csv(recipes_filename, usecols=recipe_columns)
    recipes_dataset.rename(columns=recipe_columns_rename, inplace=True)
    recipes_dataset.set_index('Id', inplace=True)
    recipes_dataset['Description'] = recipes_dataset['Description'].fillna('Not available')
    recipes_dataset['Category'] = recipes_dataset['Category'].fillna('Uncategorized')
    recipes_dataset.dropna(inplace=True)
    recipes_dataset['Time'] = recipes_dataset['Time'].map(field_time_to_string)
    recipes_dataset['Ingredients'] = recipes_dataset['Ingredients'].map(field_string_to_list)
    recipes_dataset['IngredientQuantities'] = recipes_dataset['IngredientQuantities'].map(field_string_to_list)
    recipes_dataset['Instructions'] = recipes_dataset['Instructions'].map(field_string_to_list)

    ## Reviews data
    review_columns = ['RecipeId', 'Rating']
    review_columns_rename = {'RecipeId': 'Id'}
    reviews_dataset = pd.read_csv(reviews_filename, usecols=review_columns).dropna()
    reviews_dataset.rename(columns=review_columns_rename, inplace=True)
    reviews_dataset = reviews_dataset.groupby('Id', as_index=False).mean()
    reviews_dataset.set_index('Id', inplace=True)

    ## Composing recipes and reviews
    composite_dataset = recipes_dataset.join(reviews_dataset, how='inner')

    return composite_dataset

def load_foodcom_tags_dataset(filename:str = FOODCOM_TAGS_DATASET_LOCATION) -> pd.DataFrame:
    recipe_columns = ['id', 'name', 'description', 'ingredients', 'ingredients_raw', 'steps', 'tags']
    recipe_columns_rename = {
        'id': 'Id', 'name': 'Name', 'description': 'Description', 'ingredients': 'Ingredients',
        'ingredients_raw': 'IngredientQuantities', 'steps': 'Instructions', 'tags': 'Category'
    }
    recipes_dataset = pd.read_csv(filename, usecols=recipe_columns)
    recipes_dataset.rename(columns=recipe_columns_rename, inplace=True)
    recipes_dataset.set_index('Id', inplace=True)
    recipes_dataset['Description'] = recipes_dataset['Description'].fillna('Not available')
    recipes_dataset.dropna(inplace=True)
    recipes_dataset['Ingredients'] = recipes_dataset['Ingredients'].map(field_string_to_list)
    recipes_dataset['IngredientQuantities'] = recipes_dataset['IngredientQuantities'].map(field_string_to_list)
    recipes_dataset['Instructions'] = recipes_dataset['Instructions'].map(field_string_to_list)
    recipes_dataset['Category'] = recipes_dataset['Category'].map(field_string_to_list)
    #recipes_dataset['IngredientQuantities'] = recipes_dataset.apply(lambda row: field_ingredients_quantities(row['Ingredients'], row['IngredientQuantities']), axis=1)

    return recipes_dataset

def load_recipenlg_dataset(filename:str = RECIPENLG_DATASET_LOCATION) -> pd.DataFrame:
    recipe_columns = ['Unnamed: 0', 'title', 'ingredients', 'directions', 'NER']
    recipe_columns_rename = {
        'Unnamed: 0': 'Id', 'title': 'Name', 'ingredients': 'IngredientQuantities',
        'directions': 'Instructions', 'NER': 'Ingredients'
    }
    recipes_dataset = pd.read_csv(filename, usecols=recipe_columns)
    recipes_dataset.rename(columns=recipe_columns_rename, inplace=True)
    recipes_dataset.set_index('Id', inplace=True)
    recipes_dataset.dropna(inplace=True)
    recipes_dataset['Ingredients'] = recipes_dataset['Ingredients'].map(field_string_to_list)
    recipes_dataset['IngredientQuantities'] = recipes_dataset['IngredientQuantities'].map(field_string_to_list)
    recipes_dataset['Instructions'] = recipes_dataset['Instructions'].map(field_string_to_list)
    recipes_dataset['IngredientQuantities'] = recipes_dataset.apply(lambda row: field_ingredients_quantities(row['Ingredients'], row['IngredientQuantities']), axis=1)

    return recipes_dataset

def load_recipes3k_dataset(filename:str = RECIPES3K_DATASET_LOCATION) -> pd.DataFrame:
    files = glob.glob(filename)
    dataframes = []
    for file in files:
        dataframes.append(pd.read_json(file))

    recipes_columns = [
        'id', 'name', 'description', 'rattings', 'ingredients', 'steps',# 'nutrients',
        'times', 'serves', 'difficult', 'maincategory'#, 'subcategory', 'dish_type'
    ]
    recipes_columns_rename = {
        'id':'Id', 'name': 'Name', 'description': 'Description', 'rattings': 'Rating',
        'ingredients': 'Ingredients', 'steps': 'Instructions',# 'nutrients': 'Nutrients',
        'times': 'Time', 'serves': 'Serves', 'difficult': 'Difficulty',
        'subcategory': 'Subcategory', 'dish_type': 'DishType', 'maincategory': 'Category'
    }
    recipes_dataset = pd.concat(dataframes)[recipes_columns]
    recipes_dataset.rename(columns=recipes_columns_rename, inplace=True)
    recipes_dataset.set_index('Id', inplace=True)
    recipes_dataset.loc[recipes_dataset['Ingredients'].map(lambda l: len(l) == 0), 'Ingredients'] = np.nan
    #recipes_dataset.loc[recipes_dataset['Nutrients'].map(lambda d: len(d) == 0), 'Nutrients'] = np.nan
    recipes_dataset.loc[recipes_dataset['Time'].map(lambda d: len(d) == 0), 'Time'] = np.nan
    recipes_dataset.dropna(inplace=True)
    recipes_dataset['Time'] = recipes_dataset['Time'].map(dictionary_time_to_string)

    return recipes_dataset


if __name__ == '__main__':
    #food_dataset = load_foodcom_review_dataset()
    #food_dataset = load_foodcom_tags_dataset()
    #food_dataset = load_recipenlg_dataset()
    food_dataset = load_recipes3k_dataset()

    print(dataframe_information(food_dataset))
