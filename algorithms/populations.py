from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, List, Any, Dict

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
