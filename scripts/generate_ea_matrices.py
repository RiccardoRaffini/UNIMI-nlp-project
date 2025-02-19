import sys
sys.path.append('.')

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from commons.data import load_foodcom_review_dataset
from commons.nlp_utils import RecipeProcessor
from commons.recipes import Recipe, RecipeGraph, RecipeMatrices

def main():
    seed = 1234

    ## Sample data
    recipes_dataset = load_foodcom_review_dataset()
    recipes_dataset = recipes_dataset.sample(n=60_000, random_state=seed)
    print(len(recipes_dataset))

    ## Set recipe processor
    recipe_processor = RecipeProcessor(ignored_prepositions={'at', 'for'}, additional_objects={'heat'})
    Recipe.set_recipe_processor(recipe_processor)

    ## Populate recipes matrices
    recipe_matrices = RecipeMatrices()
    recipes_number = len(recipes_dataset)

    for index in tqdm(range(recipes_number), 'Recipes processing'):
        try:
            recipe = recipes_dataset.iloc[index]
            recipe = Recipe.from_dataframe_row(recipe)

            recipe_graph = RecipeGraph.from_recipe(recipe)
            recipe_graph.simplify_graph()

            recipe_matrices.process_recipe_graph(recipe_graph)
        except Exception as e:
            print(e)
            print('>>>', index)

        if index % 100 == 0:
            recipe_matrices.compile()
            recipe_matrices.save('global_recipes_matrices')

    recipe_matrices.compile()
    recipe_matrices.save('global_recipes_matrices')

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`*.')
    warnings.filterwarnings('ignore', '1Torch was not compiled with flash attention.')
    
    main()