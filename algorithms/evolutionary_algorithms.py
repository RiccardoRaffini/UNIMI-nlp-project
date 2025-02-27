import random
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Generic, TypeVar, Callable, List, Any, Union, Iterable, Dict, Tuple

from algorithms.crossovers import Crossover, RecipeCrossover
from algorithms.fitness_evaluators import FitnessEvaluator
from algorithms.mutations import Mutation, RecipeNodeMutation
from algorithms.population_selectors import PopulationSelector
from algorithms.populations import Population, RecipePopulation, RecipeIndividual

T = TypeVar('T')

class EvolutionaryAlgorithm(ABC, Generic[T]):
    def __init__(self, configuration:Dict[str, Any]):
        super(EvolutionaryAlgorithm, self).__init__()

        self._population:Population[T] = configuration.get('population', None)
        self._population_size:int = self._population.size() if self._population else None
        self._fitness_evaluator:FitnessEvaluator = configuration.get('fitness evaluator', None)
        self._population_selector:PopulationSelector = configuration.get('population selector')

        ## Configuration checks
        assert self._fitness_evaluator is not None, 'a fitness evaluator must be assigned to this algorithm'
        assert self._population_selector is not None, 'a population selector must be assigned to this algorithm'

    def run(self, evolution_epochs:int, verbose:bool = False) -> Dict[str, Any]:
        assert self._population is not None, 'an initial Population must be assigned to this algorithm before running it'

        epochs_data = dict()
        
        for epoch in tqdm(range(evolution_epochs), 'Algorithm epochs'):
            print(f'>Epoch {epoch}')
            epoch_data = dict()
            epoch_data['initial_population_size'] = self._population.size()

            ## Create next generation population
            next_population = type(self._population)()
            next_population.expand(self._population)

            mutated_individuals = self.mutation(self._population)
            next_population.expand(mutated_individuals)

            crossed_individuals = self.crossover(self._population)
            next_population.expand(crossed_individuals)
            
            epoch_data['new_population_size'] = next_population.size()

            ## Compute next population score
            next_population_score = self.evaluate(next_population)
            epoch_data['new_population_scores'] = next_population_score

            ## Select final population
            print('Individuals selection')
            self._population, final_population_score = self.select_population(next_population, self._population_size, next_population_score)
            epoch_data['final_population_size'] = self._population.size()
            epoch_data['final_population_score'] = final_population_score

            ## Update epochs data
            epochs_data[epoch] = epoch_data
            if verbose:
                print(f'epoch [{epoch}/{evolution_epochs-1}]: {epoch_data}')

        return epochs_data
    
    def set_population(self, population:Population[T]) -> None:
        self._population = population
        self._population_size = self._population.size()

    def get_population(self) -> Population[T]:
        assert self._population is not None, 'no Population set for algorithm'

        return self._population.copy()

    @abstractmethod
    def mutation(self, population:Population[T]) -> Population[T]:
        raise NotImplementedError

    @abstractmethod
    def crossover(self, population:Population[T]) -> Population[T]:
        raise NotImplementedError

    def evaluate(self, population:Population[T]) -> List[float]:
        population_evaluations = []
        for individual in tqdm(population, 'Individuals evaluation'):
            individual_evaluation = self._fitness_evaluator(individual)
            population_evaluations.append(individual_evaluation)

        return population_evaluations

    def select_population(self, population:Population[T], individuals_number:int, population_scores:List[float] = None) -> Tuple[Population[T], List[float]]:
        selected_population, selected_scores = self._population_selector(population, individuals_number, population_scores)

        return selected_population, selected_scores

## ==========

class RecipeEvolutionaryAlgorithm(EvolutionaryAlgorithm):
    def __init__(self, configuration:Dict[str, Any]):
        super(RecipeEvolutionaryAlgorithm, self).__init__(configuration)

        ## Crossover configuration
        self._crossover_methods:List[RecipeCrossover] = configuration.get('crossover methods', []).copy()
        self._crossover_probabilities:List[float] = configuration.get('crossover probabilities', None)
        self._crossover_individuals_selector:List[PopulationSelector] = configuration.get('crossover individuals selector', None)

        ## Mutation configuration
        self._mutation_methods:List[RecipeNodeMutation] = configuration.get('mutation methods', []).copy()
        self._mutation_probabilities:List[float] = configuration.get('mutation probabilities', None)
        self._mutation_nodes_selectors:Callable[[RecipeIndividual], int] = configuration.get('mutation nodes selectors', []).copy()

        ## Configuration checks
        if self._crossover_probabilities is not None:
            assert len(self._crossover_methods) == len(self._crossover_probabilities), 'each crossover method must have exactly one probability value associated to it'
        assert self._crossover_individuals_selector is not None, 'an individual selector for crossover individuals must be specfied'
        if self._mutation_probabilities is not None:
            assert len(self._mutation_methods) == len(self._mutation_probabilities), 'each mutation method must have exactly one probability value associated to it'
        assert len(self._mutation_methods) == len(self._mutation_nodes_selectors), 'each mutation method must have exactly one node selctor associated to it'

    def mutation(self, population:RecipePopulation) -> RecipePopulation:
        ## Create individuals copies
        mutated_individuals = population.copy()

        ## Select application method
        if self._mutation_probabilities is None:
            ## Iterate individual in population
            for individual in tqdm(population, 'Individuals mutation'):
                ## Apply all mutations
                for mutation, node_selector in zip(self._mutation_methods, self._mutation_nodes_selectors):
                    node_index = node_selector(individual)
                    if node_index is not None and mutation.is_valid(individual, node_index):
                        mutation(individual, node_index)
        
        else:
            ## Iterate individuals in population
            for individual in tqdm(population, 'Individuals mutation'):
                ## Select mutation to apply
                selected_index = random.choices(range(len(self._mutation_methods)), weights=self._mutation_probabilities)[0]
                mutation = self._mutation_methods[selected_index]
                node_selector = self._mutation_nodes_selectors[selected_index]

                ## Apply selected mutation
                node_index = node_selector(individual)
                if node_index is not None and mutation.is_valid(individual, node_index):
                    mutation(individual, node_index)

        return mutated_individuals
    
    def crossover(self, population:RecipePopulation) -> RecipePopulation:
        ## Create individuals copies
        crossover_individuals = population.copy()

        ## Select application method
        if self._crossover_probabilities is None:
            ## Iterate individuals pairs
            for _ in tqdm(range(population.size() // 2), 'Individuals crossover'):
                ## Apply all crossovers
                for crossover, individuals_selector in zip(self._crossover_methods, self._crossover_individuals_selector):
                    individuals, _ = individuals_selector(population, 2)
                    individual_1 = individuals.individual_at(0)
                    individual_2 = individuals.individual_at(1)

                    if crossover.is_valid(individual_1, individual_2):
                        crossover(individual_1, individual_2)

        else:
            ## Iterate individuals pairs
            for _ in tqdm(range(population.size() // 2), 'Individuals crossover'):
                ## Select crossover to apply
                selected_index = random.choices(range(len(self._crossover_methods)), weights=self._crossover_probabilities)[0]
                crossover = self._crossover_methods[selected_index]
                individuals_selector = self._crossover_individuals_selector[selected_index]

                ## Apply selected crossover
                individuals, _ = individuals_selector(population, 2)
                individual_1 = individuals.individual_at(0)
                individual_2 = individuals.individual_at(1)

                if crossover.is_valid(individual_1, individual_2):
                    crossover(individual_1, individual_2)

        return crossover_individuals
