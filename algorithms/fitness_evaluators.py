from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')

class FitnessEvaluator(ABC, Generic[T]):
    @abstractmethod
    def evaluate(self, individual:T) -> float:
        raise NotImplementedError()
    
    def __call__(self, individual:T) -> float:
        evaluation = self.evaluate(individual)
        return evaluation
