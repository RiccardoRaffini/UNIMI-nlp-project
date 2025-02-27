import numpy as np
import random
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, List, Any, Union, Iterable, Tuple

from algorithms.populations import Population
from algorithms.fitness_evaluators import FitnessEvaluator

T = TypeVar('T')

class PopulationSelector(ABC, Generic[T]):
    @abstractmethod
    def select(self, population:Population[T], individuals_number:int = 1, population_scores:List[float] = None) -> Tuple[Population[T], List[float]]:
        raise NotImplementedError()
    
    def __call__(self, population:Population[T], individuals_number:int = 1, population_scores:List[float] = None) -> Tuple[Population[T], List[float]]:
        selected_population, selected_scores = self.select(population, individuals_number, population_scores)
        return selected_population, selected_scores
    
class RandomSelector(PopulationSelector[T]):
    def __init__(self, with_replacement:bool = False):
        super(RandomSelector, self).__init__()

        self._with_replacement = with_replacement

    def select(self, population:Population[T], individuals_number:int = 1, population_scores:List[float] = None) -> Tuple[Population[T], List[float]]:
        ## Select random indices
        if self._with_replacement:
            selected_indices = random.choices(range(population.size()), k=individuals_number)
        else:
            selected_indices = random.sample(range(population.size()), k=individuals_number)

        ## Select individuals
        selected_population = type(population)()
        for selected_index in selected_indices:
            selected_individual = population.individual_at(selected_index)
            selected_population.add_individual(selected_individual)

        ## Select individuals fitness
        if population_scores is not None:
            selected_scores = [population_scores[index] for index in selected_indices]
        else:
            selected_scores = None

        return selected_population, selected_scores

class RouletteWheelSelector(PopulationSelector[T]):
    def __init__(self, fitness_evaluator:FitnessEvaluator[T], fitness_scaling_factor:float = 1.0) -> None:
        super(PopulationSelector, self).__init__()

        self._fitness_evaluator = fitness_evaluator
        self._fitness_scaling_factor = fitness_scaling_factor

    def select(self, population:Population[T], individuals_number:int = 1, population_scores:List[float] = None) -> Tuple[Population[T], List[float]]:
        ## Compute absolute fitness of indivduals
        if population_scores is None:
            population_scores = [self._fitness_evaluator.evaluate(individual) for individual in population]

        individuals_absolute_fitness = [self._fitness_scaling_factor * fitness_score for fitness_score in population_scores]

        ## Transform into relative fitness (probability)
        fintess_sum = sum(individuals_absolute_fitness)
        individual_relative_probabilities = [individual_absolute_fitness / fintess_sum for individual_absolute_fitness in individuals_absolute_fitness]

        ## Select individuals
        selected_indices = np.random.choice(population.size(), individuals_number, replace=True, p=individual_relative_probabilities)
        selected_individuals = [population.individual_at(index) for index in selected_indices]
        selected_population = type(population)(selected_individuals)

        ## Select individuals fitness
        selected_scores = [population_scores[index] for index in selected_indices]

        return selected_population, selected_scores
    
class ExpectedValueSelector(PopulationSelector[T]):
    def __init__(self, fitness_evaluator:FitnessEvaluator[T], fitness_scaling_factor:float = 1.0, markers_number:int = 1) -> None:
        super(PopulationSelector, self).__init__()

        self._fitness_evaluator = fitness_evaluator
        self._fitness_scaling_factor = fitness_scaling_factor
        self._markers_number = markers_number
        self._markers_distance = 1 / self._markers_number

    def select(self, population:Population[T], individuals_number:int = 1, population_scores:List[float] = None) -> Tuple[Population[T], List[float]]:
        ## Compute absolute fitness of individuals
        if population_scores is None:
            population_scores = [self._fitness_evaluator.evaluate(individual) for individual in population]
        
        individuals_absolute_fitness = [self._fitness_scaling_factor * fitness_score for fitness_score in population_scores]

        ## Transform into relative fitness (probability)
        fintess_sum = sum(individuals_absolute_fitness)
        individual_relative_probabilities = [individual_absolute_fitness / fintess_sum for individual_absolute_fitness in individuals_absolute_fitness]

        ## Select individuals
        selected_population = type(population)()
        selected_scores = []
        
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
                selected_scores.append(population_scores[current_individual_index-1])

                if selected_population.size() == individuals_number:
                    break

        return selected_population, selected_scores
    
class RankBasedSelector(PopulationSelector[T]):
    def __init__(self, fitness_evaluator:FitnessEvaluator[T]) -> None:
        super(PopulationSelector, self).__init__()

        self._fitness_evaluator = fitness_evaluator

    def select(self, population:Population[T], individuals_number:int = 1, population_scores:List[float] = None) -> Tuple[Population[T], List[float]]:
        ## Sort individuals by fitness
        if population_scores is None:
            population_scores = [self._fitness_evaluator.evaluate(individual) for individual in population]

        indices_individuals = sorted(enumerate(population), key=lambda index_individual: population_scores[index_individual[0]])
        new_population = type(population)()
        for _, individual in indices_individuals:
            new_population.add_individual(individual)
        population = new_population

        ## Assign ranks to individuals
        individuals_ranks = list(range(1, population.size()+1))

        ## Assign probabilities to ranks
        ranks_sum = population.size() * (population.size() + 1) // 2
        individual_relative_probabilities = [individual_rank / ranks_sum for individual_rank in individuals_ranks]

        ## Select individuals as in roulette wheel selector
        selected_indices = np.random.choice(population.size(), individuals_number, replace=False, p=individual_relative_probabilities)
        selected_individuals = [population.individual_at(index) for index in selected_indices]
        selected_population = type(population)(selected_individuals)

        ## Select individuals fitness
        selected_scores = [population_scores[index] for index in selected_indices]

        return selected_population, selected_scores
    
class TournamentSelector(PopulationSelector[T]):
    def __init__(self, fitness_evaluator:FitnessEvaluator[T], tournament_size:int, with_replacement:bool = True) -> None:
        super(PopulationSelector, self).__init__()

        self._fitness_evaluator = fitness_evaluator
        self._tournament_size = tournament_size
        self._with_replacement = with_replacement

    def _perform_tournament(self, individuals:List[T], scores:List[float]) -> Tuple[T, float]:
        selected_individuals = random.choices(individuals, k=self._tournament_size)
        selected_individuals.sort(key=self._fitness_evaluator, reverse=True)
        best_individual = selected_individuals[0]

        return best_individual

    def select(self, population:Population[T], individuals_number:int = 1, population_scores:List[float] = None) -> Tuple[Population[T], List[float]]:
        ## Perform tournaments
        selected_population = type(population)()
        selected_scores = []

        individuals = [individual for individual in population]
        if population_scores is None:
            population_scores = [self._fitness_evaluator.evaluate(individual) for individual in population]

        for _ in range(individuals_number):
            selected_individual, selected_score = self._perform_tournament(individuals, population_scores)
            selected_population.add_individual(selected_individual)
            selected_scores.append(selected_score)

            if not self._with_replacement:
                individual_index = individuals.index(selected_individual)
                del individuals[individual_index]
                del population_scores[individual_index]

        return selected_population, selected_scores
    
class ElitismSelector(PopulationSelector[T]):
    def __init__(self, fitness_evaluator:FitnessEvaluator[T]) -> None:
        super(PopulationSelector, self).__init__()

        self._fitness_evaluator = fitness_evaluator

    def select(self, population:Population[T], individuals_number:int = 1, population_scores:List[float] = None) -> Tuple[Population[T], List[float]]:
        ## Sort individuals by fitness
        if population_scores is None:
            population_scores = [self._fitness_evaluator.evaluate(individual) for individual in population]

        indices_individuals = sorted(enumerate(population), key=lambda index_individual: population_scores[index_individual[0]], reverse=True)
        population_scores.sort(reverse=True)
        new_population = type(population)()
        for _, individual in indices_individuals:
            new_population.add_individual(individual)
        population = new_population

        ## Select best individuals
        if individuals_number >= population.size():
            return population

        selected_population = type(population)()
        selected_scores = []

        for i in range(individuals_number):
            selected_individual = population.individual_at(i)
            selected_population.add_individual(selected_individual)
            selected_scores.append(population_scores[i])

        return selected_population, selected_scores
