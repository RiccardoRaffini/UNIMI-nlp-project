import random
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Tuple

from algorithms.populations import RecipeIndividual
from commons.recipes import Action, Ingredient, Tool, Miscellaneous

T = TypeVar('T')

class Crossover(ABC, Generic[T]):
    def __call__(self, individual_1:T, individual_2:T) -> Tuple[T, T]:
        raise NotImplementedError
    
    @abstractmethod
    def is_valid(self, individual_1:T, individual_2:T) -> bool:
        raise NotImplementedError
    
## ==========

class RecipeCrossover(Crossover):
    def __call__(self, individual_1:RecipeIndividual, individual_2:RecipeIndividual) -> Tuple[RecipeIndividual, RecipeIndividual]:
        ## Check crossover is valid
        assert self.is_valid(individual_1, individual_2), 'cannot apply crossover to these individuals'

        ## Crossover individuals
        self._crossover(individual_1, individual_2)

        return individual_1, individual_2

    @abstractmethod
    def is_valid(self, individual_1:RecipeIndividual, individual_2:RecipeIndividual) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _crossover(self, individual_1:RecipeIndividual, individual_2:RecipeIndividual) -> None:
        raise NotImplementedError

class MixNodeCrossover(RecipeCrossover):
    def is_valid(self, individual_1:RecipeIndividual, individual_2:RecipeIndividual) -> bool:
        ## Check first individual
        individual_1_is_valid = False
        for node_index in individual_1.nodes():
            node = individual_1.get_node(node_index)
            if type(node['object']) == Action and node['object'].group == 'mix':
                individual_1_is_valid = True
                break

        ## Check second individual
        individual_2_is_valid = False
        for node_index in individual_2.nodes():
            node = individual_2.get_node(node_index)
            if type(node['object']) == Action and node['object'].group == 'mix':
                individual_2_is_valid = True
                break

        return individual_1_is_valid and individual_2_is_valid
    
    def _crossover(self, individual_1:RecipeIndividual, individual_2:RecipeIndividual):
        ## Get first individual mix nodes
        individual_1_nodes_indices = individual_1.nodes()
        individual_1_mix_nodes_indices = []
        for node_index in individual_1_nodes_indices:
            node = individual_1.get_node(node_index)
            if type(node['object']) == Action and node['object'].group == 'mix':
                individual_1_mix_nodes_indices.append(node_index)

        ## Get second individual mix nodes
        individual_2_nodes_indices = individual_2.nodes()
        individual_2_mix_nodes_indices = []
        for node_index in individual_2_nodes_indices:
            node = individual_2.get_node(node_index)
            if type(node['object']) == Action and node['object'].group == 'mix':
                individual_2_mix_nodes_indices.append(node_index)

        ## Check if both individuals have mix ndoes
        if not individual_1_mix_nodes_indices or not individual_2_mix_nodes_indices:
            return
        
        ## Select split nodes
        individual_1_split_index = random.choice(individual_1_mix_nodes_indices)
        individual_2_split_index = random.choice(individual_2_mix_nodes_indices)

        ## Select split branches
        branch_1_node_index = random.choice(individual_1.get_children_indices(individual_1_split_index))
        branch_2_node_index = random.choice(individual_2.get_children_indices(individual_2_split_index))

        ## Split branches
        individual_1.remove_edge(individual_1_split_index, branch_1_node_index)
        individual_2.remove_edge(individual_2_split_index, branch_2_node_index)

        ## Move separated branches between graphs
        new_branch_1_node_index = self._move_branch_between_individuals(branch_1_node_index, individual_1, individual_2)
        new_branch_2_node_index = self._move_branch_between_individuals(branch_2_node_index, individual_2, individual_1)

        ## Connect new branches to main graphs
        individual_1.add_generic_edge(individual_1_split_index, new_branch_2_node_index, {'type': 'primary'})
        individual_2.add_generic_edge(individual_2_split_index, new_branch_1_node_index, {'type': 'primary'})

    def _move_branch_between_individuals(self, branch_root_index:int, initial_individual:RecipeIndividual, final_individual:RecipeIndividual) -> int:
        add_functions = {
            Action: final_individual.add_action_node,
            Ingredient: final_individual.add_ingredient_node,
            Tool: final_individual.add_tool_node,
            Miscellaneous: final_individual.add_misc_node
        }

        def dfs_movement(node_index:int) -> Tuple[int, dict]:
            ## Move children
            node_children_indices = initial_individual.get_children_indices(node_index)
            children_indices_edges = []
            for child_index in node_children_indices:
                children_indices_edges.append(dfs_movement(child_index))

            ## Move node from initial graph to final graph
            node = initial_individual.get_node(node_index)
            if type(node['object']) == Action:
                primary_children = [index for index, edge in children_indices_edges if edge['type'] == 'primary']
                secondary_children = [index for index, edge in children_indices_edges if edge['type'] == 'secondary']
                new_node_index = add_functions[Action](node['object'], primary_children, secondary_children)
            else:
                new_node_index = add_functions[type(node['object'])](node['object'])

            ## Remove node from initial graph
            parent_edges = initial_individual.get_edges(node_2=node_index)
            if parent_edges:
                parent_node_index = parent_edges[0][0]
                edge_attributes = initial_individual._graph.edges[parent_node_index, node_index]
                initial_individual.remove_edge(parent_node_index, node_index)
            else:
                edge_attributes = None
            initial_individual.remove_node(node_index)

            return new_node_index, edge_attributes

        ## Move branch
        new_branch_root_index = dfs_movement(branch_root_index)[0]

        return new_branch_root_index
