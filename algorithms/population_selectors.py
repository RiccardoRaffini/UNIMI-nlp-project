import numpy as np
import random
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, List, Any, Union, Iterable

from algorithms.populations import Population
from algorithms.fitness_evaluators import FitnessEvaluator

T = TypeVar('T')

class PopulationSelector(ABC, Generic[T]):
    @abstractmethod
    def select(self, population:Population[T], individuals_number:int = 1) -> Population[T]:
        raise NotImplementedError()
    
class RouletteWheelSelector(PopulationSelector[T]):
    def __init__(self, fitness_evaluator:FitnessEvaluator[T], fitness_scaling_factor:float = 1.0) -> None:
        super(PopulationSelector, self).__init__()

        self._fitness_evaluator = fitness_evaluator
        self._fitness_scaling_factor = fitness_scaling_factor

    def select(self, population:Population[T], individuals_number:int = 1)  -> Population[T]:
        ## Compute absolute fitness of indovduals
        individuals_absolute_fitness = [self._fitness_scaling_factor * self._fitness_evaluator.evaluate(individual) for individual in population]

        ## Transform into relative fitness (probability)
        fintess_sum = sum(individuals_absolute_fitness)
        individual_relative_probabilities = [individual_absolute_fitness / fintess_sum for individual_absolute_fitness in individuals_absolute_fitness]

        ## Select individuals
        selected_indicies = np.random.choice(population.size(), individuals_number, replace=False, p=individual_relative_probabilities)
        selected_individuals = [population.individual_at(index) for index in selected_indicies]
        selected_population = type(population)(selected_individuals)

        return selected_population
    
class ExpectedValueSelector(PopulationSelector[T]):
    def __init__(self, fitness_evaluator:FitnessEvaluator[T], fitness_scaling_factor:float = 1.0, markers_number:int = 1) -> None:
        super(PopulationSelector, self).__init__()

        self._fitness_evaluator = fitness_evaluator
        self._fitness_scaling_factor = fitness_scaling_factor
        self._markers_number = markers_number
        self._markers_distance = 1 / self._markers_number

    def select(self, population:Population[T], individuals_number:int = 1) -> Population[T]:
        ## Compute absolute fitness of indovduals
        individuals_absolute_fitness = [self._fitness_scaling_factor * self._fitness_evaluator.evaluate(individual) for individual in population]

        ## Transform into relative fitness (probability)
        fintess_sum = sum(individuals_absolute_fitness)
        individual_relative_probabilities = [individual_absolute_fitness / fintess_sum for individual_absolute_fitness in individuals_absolute_fitness]

        ## Select individuals
        selected_population = type(population)()
        
        while selected_population.size() < individuals_number:
            selected_position = random.uniform(0.0, self._markers_distance)
            cumulative_probability = 0
            current_individual_index = 0

            ## Iterate markers position
            for i in range(self._markers_number):
                current_marker_position = selected_position + i * self._markers_distance

                ## Find individual on marker
                while current_individual_index < population.size() and cumulative_probability < current_marker_position:
                    cumulative_probability += individual_relative_probabilities[current_individual_index]
                    current_individual_index += 1

                selected_population.add_individual(population.individual_at(current_individual_index-1))

                if selected_population.size() == individuals_number:
                    break

        return selected_population
    
class RankBasedSelector(PopulationSelector[T]):
    def __init__(self, fitness_evaluator:FitnessEvaluator[T]) -> None:
        super(PopulationSelector, self).__init__()

        self._fitness_evaluator = fitness_evaluator

    def select(self, population:Population[T], individuals_number:int = 1) -> Population[T]:
        ## Sort individuals by fitness
        new_population = type(population)()
        new_population.expand(population)
        population = new_population
        population.sort(self._fitness_evaluator)

        ## Assign ranks to individuals
        individuals_ranks = list(range(1, population.size()+1))

        ## Assign probabilities to ranks
        ranks_sum = population.size() * (population.size() + 1) // 2
        individual_relative_probabilities = [individual_rank / ranks_sum for individual_rank in individuals_ranks]

        ## Select individuals as in roulette wheel selector
        selected_indicies = np.random.choice(population.size(), individuals_number, replace=False, p=individual_relative_probabilities)
        selected_individuals = [population.individual_at(index) for index in selected_indicies]
        selected_population = type(population)(selected_individuals)

        return selected_population
    
class TournamentSelector(PopulationSelector[T]):
    def __init__(self, fitness_evaluator:FitnessEvaluator[T], tournament_size:int, with_replacement:bool = True) -> None:
        super(PopulationSelector, self).__init__()

        self._fitness_evaluator = fitness_evaluator
        self._tournament_size = tournament_size
        self._with_replacement = with_replacement

    def _perform_tournament(self, individuals:List[T]) -> T:
        selected_individuals = random.choices(individuals, k=self._tournament_size)
        selected_individuals.sort(key=self._fitness_evaluator, reverse=True)
        best_individual = selected_individuals[0]

        return best_individual

    def select(self, population:Population[T], individuals_number:int = 1) -> Population[T]:
        ## Perform tournaments
        selected_population = type(population)()
        individuals = [individual for individual in population]

        for _ in range(individuals_number):
            selected_individual = self._perform_tournament(individuals)
            selected_population.add_individual(selected_individual)

            if not self._with_replacement:
                individuals.remove(selected_individual)

        return selected_population
    
class ElitismSelector(PopulationSelector[T]):
    def __init__(self, fitness_evaluator:FitnessEvaluator[T]) -> None:
        super(PopulationSelector, self).__init__()

        self._fitness_evaluator = fitness_evaluator

    def select(self, population:Population[T], individuals_number:int = 1) -> Population[T]:
        ## Sort individuals by fitness
        new_population = type(population)()
        new_population.expand(population)
        population = new_population
        population.sort(key=self._fitness_evaluator, reverse=True)

        ## Select best individuals
        if individuals_number >= population.size():
            return population

        selected_population = type(population)()
        population_iterator = iter(population)

        for _ in range(individuals_number):
            selected_individual = next(population_iterator)
            selected_population.add_individual(selected_individual)

        return selected_population
