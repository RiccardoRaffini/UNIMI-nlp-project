import json
import numpy as np
import scipy
from abc import ABC, abstractmethod
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
        self._data:List[Tuple[Any, int, int]] = [] # (data_value, row_index, col_index)

        self._base_matrix:coo_matrix = None

    def compile(self) -> None:
        if self._symmetric:
            row_size = column_size = len(self._labels)
        else:
            row_size = len(self._row_labels)
            column_size = len(self._column_labels)

        if self._data:
            data, row_indices, column_indices = zip(*self._data)
        else:
            data, row_indices, column_indices = [], [], []
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

        if self._symmetric:
            self._data.append((value, column_index, row_index))

    def get_labels(self) -> Union[List[str], Tuple[List[str], List[str]]]:
        if self._symmetric:
            return self._labels.copy()
        else:
            return self._row_labels.copy(), self._column_labels.copy()

    @abstractmethod
    def save_to_files(self, matrix_filename:str, labels_filename:str) -> None:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load_from_files(cls, matrix_filename:str, labels_filename:str) -> 'AdjacencyMatrix':
        raise NotImplementedError()

class MixedIngredientsMatrix(AdjacencyMatrix):
    def __init__(self):
        super(MixedIngredientsMatrix, self).__init__(symmetric=True)

    def label_to_index(self, label:str, add_not_existing:bool = True) -> int:
        label_index = self._labels_indices_map.get(label, -1)

        if label_index == -1 and add_not_existing:
            label_index = len(self._labels_indices_map)
            self._labels.append(label)
            self._labels_indices_map[label] = label_index

        return label_index
    
    def add_entry(self, row_label:str, column_label:str, value:Any):
        row_index = self.label_to_index(row_label)
        column_index = self.label_to_index(column_label)

        super().add_entry(row_index, column_index, value)

    def save_to_files(self, matrix_filename:str, labels_filename:str) -> None:
        ## Compile matrix and obtain sparse matrix
        self.compile()
        matrix = self.get_sparse_matrix()

        ## Write matrix file
        scipy.sparse.save_npz(matrix_filename, matrix)

        ## Write labels file
        with open(labels_filename, 'w') as labels_file:
            json.dump(self._labels_indices_map, labels_file)
    
    @classmethod
    def load_from_files(cls, matrix_filename:str, labels_filename:str) -> 'MixedIngredientsMatrix':
        ## Create new matrix instance
        mixed_ingredients_matrix = cls()

        ## Read matrix file
        matrix:coo_matrix = scipy.sparse.load_npz(matrix_filename)
        matrix_dok = matrix.todok()

        matrix_data = matrix_dok.values()
        matrix_indices = matrix_dok.keys()
        data_list = [(data, ) + indices for data, indices in zip(matrix_data, matrix_indices)]
        mixed_ingredients_matrix._data = data_list

        ## Read label file
        with open(labels_filename, 'r') as labels_file:
            labels_map:Dict[str, int] = json.load(labels_file)
            mixed_ingredients_matrix._labels_indices_map = labels_map
            mixed_ingredients_matrix._labels = list(labels_map.keys())

        return mixed_ingredients_matrix

class ActionsIngredientsMatrix(AdjacencyMatrix):
    def __init__(self):
        super(ActionsIngredientsMatrix, self).__init__(symmetric=False)

    def label_to_row_index(self, label:str, add_not_existing:bool = True) -> int:
        label_index = self._row_labels_indices_map.get(label, -1)

        if label_index == -1 and add_not_existing:
            label_index = len(self._row_labels_indices_map)
            self._row_labels.append(label)
            self._row_labels_indices_map[label] = label_index

        return label_index
    
    def label_to_column_index(self, label:str, add_not_existing:bool = True) -> int:
        label_index = self._column_labels_indices_map.get(label, -1)

        if label_index == -1 and add_not_existing:
            label_index = len(self._column_labels_indices_map)
            self._column_labels.append(label)
            self._column_labels_indices_map[label] = label_index

        return label_index
    
    def add_entry(self, row_label:str, column_label:str, value:Any):
        row_index = self.label_to_row_index(row_label)
        column_index = self.label_to_column_index(column_label)

        super().add_entry(row_index, column_index, value)

    def save_to_files(self, matrix_filename:str, labels_filename:str) -> None:
        ## Compile matrix and obtain sparse matrix
        self.compile()
        matrix = self.get_sparse_matrix()

        ## Write matrix file
        scipy.sparse.save_npz(matrix_filename, matrix)

        ## Write labels file
        with open(labels_filename, 'w') as labels_file:
            data = {'row': self._row_labels_indices_map, 'column': self._column_labels_indices_map}
            json.dump(data, labels_file)

    @classmethod
    def load_from_files(cls, matrix_filename:str, labels_filename:str) -> 'ActionsIngredientsMatrix':
        ## Create new matrix instance
        actions_ingredients_matrix = cls()

        ## Read matrix file
        matrix:coo_matrix = scipy.sparse.load_npz(matrix_filename)
        matrix_dok = matrix.todok()

        matrix_data = matrix_dok.values()
        matrix_indices = matrix_dok.keys()
        data_list = [(data, ) + indices for data, indices in zip(matrix_data, matrix_indices)]
        actions_ingredients_matrix._data = data_list

        ## Read label file
        with open(labels_filename, 'r') as labels_file:
            labels_map:Dict[str, Dict[str, int]] = json.load(labels_file)
            actions_ingredients_matrix._row_labels_indices_map = labels_map['row']
            actions_ingredients_matrix._row_labels = list(labels_map['row'].keys())
            actions_ingredients_matrix._column_labels_indices_map = labels_map['column']
            actions_ingredients_matrix._column_labels = list(labels_map['column'].keys())

        return actions_ingredients_matrix

class ActionsToolsMatrix(AdjacencyMatrix):
    def __init__(self):
        super(ActionsToolsMatrix, self).__init__(symmetric=False)

    def label_to_row_index(self, label:str, add_not_existing:bool = True) -> int:
        label_index = self._row_labels_indices_map.get(label, -1)

        if label_index == -1 and add_not_existing:
            label_index = len(self._row_labels_indices_map)
            self._row_labels.append(label)
            self._row_labels_indices_map[label] = label_index

        return label_index
    
    def label_to_column_index(self, label:str, add_not_existing:bool = True) -> int:
        label_index = self._column_labels_indices_map.get(label, -1)

        if label_index == -1 and add_not_existing:
            label_index = len(self._column_labels_indices_map)
            self._column_labels.append(label)
            self._column_labels_indices_map[label] = label_index

        return label_index
    
    def add_entry(self, row_label:str, column_label:str, value:Any):
        row_index = self.label_to_row_index(row_label)
        column_index = self.label_to_column_index(column_label)

        super().add_entry(row_index, column_index, value)

    def save_to_files(self, matrix_filename:str, labels_filename:str) -> None:
        ## Compile matrix and obtain sparse matrix
        self.compile()
        matrix = self.get_sparse_matrix()

        ## Write matrix file
        scipy.sparse.save_npz(matrix_filename, matrix)

        ## Write labels file
        with open(labels_filename, 'w') as labels_file:
            data = {'row': self._row_labels_indices_map, 'column': self._column_labels_indices_map}
            json.dump(data, labels_file)

    @classmethod
    def load_from_files(cls, matrix_filename:str, labels_filename:str) -> 'ActionsToolsMatrix':
        ## Create new matrix instance
        actions_tools_matrix = cls()

        ## Read matrix file
        #with open(matrix_filename, 'r') as matrix_file:
        matrix:coo_matrix = scipy.sparse.load_npz(matrix_filename)
        matrix_dok = matrix.todok()

        matrix_data = matrix_dok.values()
        matrix_indices = matrix_dok.keys()
        data_list = [(data, ) + indices for data, indices in zip(matrix_data, matrix_indices)]
        actions_tools_matrix._data = data_list

        ## Read label file
        with open(labels_filename, 'r') as labels_file:
            labels_map:Dict[str, Dict[str, int]] = json.load(labels_file)
            actions_tools_matrix._row_labels_indices_map = labels_map['row']
            actions_tools_matrix._row_labels = list(labels_map['row'].keys())
            actions_tools_matrix._column_labels_indices_map = labels_map['column']
            actions_tools_matrix._column_labels = list(labels_map['column'].keys())

        return actions_tools_matrix
