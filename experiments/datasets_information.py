import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List

from commons.data import load_foodcom_review_dataset
from commons.utils import dataframe_information

def plot_bar_categories(dataset:pd.DataFrame, colors:List[str] = ['orange', 'orangered']):
    size = dataset['Category'].size
    categories_count = dataset['Category'].groupby(dataset['Category']).count().sort_values(ascending=False)
    categories_count_normalized = categories_count.div(size)

    ax = categories_count_normalized.head(20).plot.bar(
        color=colors[0], ec=colors[1]
    )
    ax.set_xlabel('Categories')
    ax.set_ylabel('Frequency (normalized)')
    plt.show()

def plot_hist_instructions(dataset:pd.DataFrame, colors:List[str] = ['orange', 'orangered']):
    size = dataset['Instructions'].size
    instructions_numbers = dataset['Instructions'].str.len()
    instructions_count = instructions_numbers.groupby(instructions_numbers).count().sort_index()
    instructions_count_normalized = instructions_count.div(size)

    ax = instructions_numbers.plot.hist(
        bins=instructions_count.size, density=True, range=[0, max(instructions_count.index)+1],
        color=colors[0], ec=colors[1]
    )
    ax.set_xlabel('Instructions number')
    ax.set_ylabel('Frequency (normalized)')
    plt.show()

def plot_box_instructions(dataset:pd.DataFrame):
    instructions_numbers = dataset['Instructions'].str.len()
    
    ax = instructions_numbers.plot.box(vert=False)
    ax.set_label('Instructions number')
    plt.show()

def plot_hist_ingredients(dataset:pd.DataFrame, colors:List[str] = ['orange', 'orangered']):
    size = dataset['Ingredients'].size
    ingredients_numbers = dataset['Ingredients'].str.len()
    ingredients_count = ingredients_numbers.groupby(ingredients_numbers).count().sort_index()
    ingredients_count_normalized = ingredients_count.div(size)

    ax = ingredients_numbers.plot.hist(
        bins=ingredients_count.size, density=True, range=[0, max(ingredients_count.index)+1],
        color=colors[0], ec=colors[1]
    )
    ax.set_xlabel('Ingredients number')
    ax.set_ylabel('Frequency (normalized)')
    plt.show()

def plot_box_ingredients(dataset:pd.DataFrame):
    ingredients_numbers = dataset['Ingredients'].str.len()
    
    ax = ingredients_numbers.plot.box(vert=False)
    ax.set_label('Ingredients number')
    plt.show()

def plot_box_instructions_comparison(datasets:List[pd.DataFrame], names:List[str], vertical:bool = True):
    instructions_lengths = [dataset['Instructions'].str.len() for dataset in datasets]

    _, ax = plt.subplots(figsize=(5, 7) if vertical else (7, 3))
    ax.boxplot(instructions_lengths, widths=0.8, vert=vertical)
    if vertical:
        ax.set_xticklabels(names)
        ax.set_xlabel('Datasets')
        ax.set_ylabel('Instructions number')
    else:
        ax.set_yticklabels(names)
        ax.set_ylabel('Datasets')
        ax.set_xlabel('Instructions number')

    plt.show()

def plot_box_ingredients_comparison(datasets:List[pd.DataFrame], names:List[str], vertical:bool = True):
    ingredients_lengths = [dataset['Ingredients'].str.len() for dataset in datasets]

    _, ax = plt.subplots(figsize=(5, 7) if vertical else (7, 3))
    ax.boxplot(ingredients_lengths, widths=0.8, vert=vertical)
    if vertical:
        ax.set_xticklabels(names)
        ax.set_xlabel('Datasets')
        ax.set_ylabel('Ingredients number')
    else:
        ax.set_yticklabels(names)
        ax.set_ylabel('Datasets')
        ax.set_xlabel('Ingredients number')

    plt.show()

def plot_box_generic_comparison(datasets:List[pd.DataFrame], names:List[str], attribute:str, label:str, vertical:bool = True):
    attribute_values = [dataset[attribute] for dataset in datasets]

    _, ax = plt.subplots(figsize=(5, 7) if vertical else (7, 3))
    ax.boxplot(attribute_values, widths=0.8, vert=vertical)
    if vertical:
        ax.set_xticklabels(names)
        ax.set_xlabel('Datasets')
        ax.set_ylabel(label)
    else:
        ax.set_yticklabels(names)
        ax.set_ylabel('Datasets')
        ax.set_xlabel(label)

    plt.show()

def main():
    ## === TRAINING DATASET INFORMATION ===

    seed = 1234
    recipes_dataset = load_foodcom_review_dataset()
    train_recipes_dataset = recipes_dataset.sample(n=60_000, random_state=seed)
    print('Recipe dataframe')
    print(dataframe_information(train_recipes_dataset))

    plot_bar_categories(train_recipes_dataset, ['orange', 'orangered'])
    plot_hist_instructions(train_recipes_dataset, ['orange', 'orangered'])
    plot_hist_ingredients(train_recipes_dataset, ['orange', 'orangered'])
    plot_box_instructions(train_recipes_dataset)
    plot_box_ingredients(train_recipes_dataset)

    ## === SCORING DATASET INFORMATION ===

    available_recipes = recipes_dataset.drop(train_recipes_dataset.index, axis=0)               # remove recipes used in training
    available_recipes = available_recipes[available_recipes['Instructions'].str.len()  >= 2]    # filter by instructions number
    available_recipes = available_recipes[available_recipes['Instructions'].str.len()  <= 12]   # filter by instructions number
    available_recipes = available_recipes[available_recipes['Ingredients'].str.len() >= 2]      # filter by ingrdients number
    available_recipes = available_recipes[available_recipes['Ingredients'].str.len() <= 13]     # filter by ingrdients number
    reference_dataset = available_recipes.sample(n=100, random_state=seed)
    print(dataframe_information(reference_dataset))

    plot_bar_categories(reference_dataset, ['lightseagreen', 'teal'])
    plot_hist_instructions(reference_dataset, ['lightseagreen', 'teal'])
    plot_hist_ingredients(reference_dataset, ['lightseagreen', 'teal'])
    plot_box_instructions(reference_dataset)
    plot_box_ingredients(reference_dataset)

    ## === COMPARISON INFORMATION ===

    plot_box_instructions_comparison([train_recipes_dataset, reference_dataset], ['Train', 'Reference'], False)
    plot_box_ingredients_comparison([train_recipes_dataset, reference_dataset], ['Train', 'Reference'], False)

if __name__ == '__main__':
    main()