import pandas as pd
from functools import reduce
from typing import List

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
        self._step_ingredients = None
        self._step_tools = None
        self._step_actions = None

        ## Process new recipe
        self._process_recipe()

    def _process_recipe(self) -> None:
        """
        Processes recipe's raw fields to obtain its internal representation using
        the recipe processor assigned to ths class.
        """

        assert self.processor is not None, 'cannot process a recipe without a processor'

        processed_instructions = self.processor.process_instructions(self._raw_instructions)

        self._step_ingredients, self._step_tools, self._step_actions = reduce(
            lambda a, b: (a[0] + b[1], a[1] + b[2], a[2] + b[3]),
            processed_instructions,
            ([], [], [])
        )
