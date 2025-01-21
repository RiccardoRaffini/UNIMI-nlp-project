from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Generic, TypeVar, Callable, List, Any, Dict

from commons.recipes import Recipe, RecipeGraph, RecipeMatrices

T = TypeVar('T')

class Population(ABC, Generic[T]):
    @abstractmethod
    def add_individual(self, individual:T) -> None:
        raise NotImplementedError()
    
    def expand(self, population:'Population[T]') -> None:
        for individual in population:
            self.add_individual(individual)
    
    @abstractmethod
    def individual_at(self, index:int) -> T:
        raise NotImplementedError()
    
    @abstractmethod
    def sort(self, key:Callable[[T], Any], reverse:bool = False) -> None:
        raise NotImplementedError()
        
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def copy(self) -> 'Population':
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return self.size()
    
    def __iter__(self) -> 'Population[T]':
        self.__iter_index__ = 0

        return self

    def __next__(self) -> T:
        if self.__iter_index__ >= self.size():
            raise StopIteration()
        
        next_individual = self.individual_at(self.__iter_index__)
        self.__iter_index__ += 1
        
        return next_individual

## ==========

class RecipeIndividual(RecipeGraph):
    @classmethod
    def from_recipe_graph(cls, recipe_graph:RecipeGraph) -> 'RecipeIndividual':
        recipe_individual = cls(recipe_graph._graph_configuration, recipe_graph._full_label, recipe_graph._action_group)
        recipe_individual._graph = recipe_graph._graph.copy()
        recipe_individual._root = recipe_graph._root

        return recipe_individual
    
    @classmethod
    def from_recipe(cls, recipe:Recipe, additional_configuration = None) -> 'RecipeIndividual':
        recipe_graph = super().from_recipe(recipe, additional_configuration)
        recipe_individual = cls.from_recipe_graph(recipe_graph)

        return recipe_individual

    def __init__(self, additional_configuration:Dict[str, Any] = None, show_full_label:bool = True, show_action_group:bool = False):
        super(RecipeIndividual, self).__init__(additional_configuration, show_full_label, show_action_group)

    def mutate(self) -> 'RecipeIndividual':
        pass

    def crossover(self, other:'RecipeIndividual') -> 'RecipeIndividual':
        pass

    def copy(self) -> 'RecipeIndividual':
        return deepcopy(self)

class RecipePopulation(Population[RecipeIndividual]):
    def __init__(self, population:List[RecipeIndividual] = None) -> None:
        super(Population, self).__init__()

        self._population:List[RecipeIndividual] = list()
        if population is not None:
            self._population = population.copy()

    def add_individual(self, individual:RecipeIndividual) -> None:
        self._population.append(individual)

    def expand(self, population:'RecipePopulation') -> None:
        self._population.extend(population)
    
    def individual_at(self, index:int) -> RecipeIndividual:
        if index > self.size():
            raise IndexError('individual index out of range')
        
        individual = self._population[index]
        return individual
    
    def sort(self, key:Callable[[RecipeIndividual], Any], reverse:bool = False) -> None:
        self._population.sort(key=key, reverse=reverse)

    def size(self) -> int:
        return len(self._population)

    def copy(self) -> 'RecipePopulation':
        individuals_copies = [individual.copy() for individual in self._population]
        population_copy = type(self)(individuals_copies)

        return population_copy
