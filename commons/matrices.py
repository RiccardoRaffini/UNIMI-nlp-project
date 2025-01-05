import numpy as np
from abc import ABC
from scipy.sparse import coo_matrix, csr_matrix
from typing import List, Dict, Tuple, Any, Union

class AdjacencyMatrix(ABC):
    def __init__(self, symmetric:bool) -> None:
        super(AdjacencyMatrix, self).__init__()

        ## Define matrix shape
        self._symmetric = symmetric

        if self._symmetric:
            self._labels:List[str] = []
            self._labels_indices_map:Dict[str, int] = {}

        else: # non-symmetric
            self._row_labels:List[str] = []
            self._row_labels_indices_map:Dict[str, int] = {}
            self._column_labels:List[str] = []
            self._column_labels_indices_map:Dict[str, int] = {}

        ## Define matrix
        self._data:List[Tuple[int, int, Any]] = [] # (row_index, col_index, data_value)

        self._base_matrix:coo_matrix = None

    def compile(self) -> None:
        if self._symmetric:
            row_size = column_size = len(self._labels)
        else:
            row_size = len(self._row_labels)
            column_size = len(self._column_labels)

        data, row_indices, column_indices = zip(*self._data)
        self._base_matrix = coo_matrix((data, (row_indices, column_indices)), shape=(row_size, column_size))

    def get_sparse_matrix(self) -> coo_matrix:
        assert self._base_matrix is not None, 'matrix must be compiled first'

        return self._base_matrix.copy()

    def get_csr_matrix(self) -> csr_matrix:
        assert self._base_matrix is not None, 'matrix must be compiled first'

        return self._base_matrix.tocsr(copy=True)
    
    def get_full_matrix(self) -> np.ndarray:
        assert self._base_matrix is not None, 'matrix must be compiled first'

        return self._base_matrix.toarray()

    def add_entry(self, row_index:int, column_index:int, value:Any) -> None:
        self._data.append((value, row_index, column_index))

    def get_labels(self) -> Union[List[str], Tuple[List[str], List[str]]]:
        if self._symmetric:
            return self._labels.copy()
        else:
            return self._row_labels.copy(), self._column_labels.copy()

class MixedIngredientsMatrix(AdjacencyMatrix):
    def __init__(self):
        super(MixedIngredientsMatrix, self).__init__(symmetric=True)

    def label_to_index(self, label:str) -> int:
        label_index = self._labels_indices_map.get(label, -1)

        if label_index == -1:
            label_index = len(self._labels_indices_map)
            self._labels.append(label)
            self._labels_indices_map[label] = label_index

        return label_index
    
    def add_entry(self, row_label:str, column_label:str, value:Any):
        row_index = self.label_to_index(row_label)
        column_index = self.label_to_index(column_label)

        super().add_entry(row_index, column_index, value)

class ActionsIngredientsMatrix(AdjacencyMatrix):
    def __init__(self):
        super(ActionsIngredientsMatrix, self).__init__(symmetric=False)

    def label_to_row_index(self, label:str) -> int:
        label_index = self._row_labels_indices_map.get(label, -1)

        if label_index == -1:
            label_index = len(self._row_labels_indices_map)
            self._row_labels.append(label)
            self._row_labels_indices_map[label] = label_index

        return label_index
    
    def label_to_column_index(self, label:str) -> int:
        label_index = self._column_labels_indices_map.get(label, -1)

        if label_index == -1:
            label_index = len(self._column_labels_indices_map)
            self._column_labels.append(label)
            self._column_labels_indices_map[label] = label_index

        return label_index
    
    def add_entry(self, row_label:str, column_label:str, value:Any):
        row_index = self.label_to_row_index(row_label)
        column_index = self.label_to_column_index(column_label)

        super().add_entry(row_index, column_index, value)

class ActionsToolsMatrix(AdjacencyMatrix):
    def __init__(self):
        super(ActionsToolsMatrix, self).__init__(symmetric=False)

    def label_to_row_index(self, label:str) -> int:
        label_index = self._row_labels_indices_map.get(label, -1)

        if label_index == -1:
            label_index = len(self._row_labels_indices_map)
            self._row_labels.append(label)
            self._row_labels_indices_map[label] = label_index

        return label_index
    
    def label_to_column_index(self, label:str) -> int:
        label_index = self._column_labels_indices_map.get(label, -1)

        if label_index == -1:
            label_index = len(self._column_labels_indices_map)
            self._column_labels.append(label)
            self._column_labels_indices_map[label] = label_index

        return label_index
    
    def add_entry(self, row_label:str, column_label:str, value:Any):
        row_index = self.label_to_row_index(row_label)
        column_index = self.label_to_column_index(column_label)

        super().add_entry(row_index, column_index, value)
