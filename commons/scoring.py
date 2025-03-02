import numpy as np
import pandas as pd
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.spatial import distance
from tqdm import tqdm
from typing import List, Dict, Tuple

stop_words = set(stopwords.words('english'))
lemmer = WordNetLemmatizer()

def tokenize(document:str) -> List[str]:
    ## Clear punctuation
    document = document.translate(str.maketrans('', '', string.punctuation))

    ## Tokenize
    tokens = word_tokenize(document)

    ## Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

def normalize(tokens:List[str]) -> List[str]:
    ## Lemming
    tokens = [lemmer.lemmatize(token) for token in tokens]

    return tokens

def term_frequency(term:str, document_terms:List[str]) -> float:
    term_occurrences = document_terms.count(term)
    document_size = len(document_terms)
    term_frequency_ = term_occurrences / document_size

    return term_frequency_

def inverse_document_frequency(term:str, documents_corpus:List[List[str]]) -> float:
    documents_number = len(documents_corpus) + 1
    documents_word_count = sum([1 for document in documents_corpus if term in document]) + 1
    inverse_document_frequency_ = np.log10(documents_number / documents_word_count) + 1

    return inverse_document_frequency_

def tf_idf_score(term:str, document:List[str], documents_corpus:List[List[str]]) -> float:
    term_frequency_ = term_frequency(term, document)
    inverse_document_frequency_ = inverse_document_frequency(term, documents_corpus)
    tf_idf_score_ = term_frequency_ * inverse_document_frequency_

    return tf_idf_score_

def create_tf_idf_vector_space(documents_corpus:List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    ## Terms mapping
    documents_corpus_terms = [normalize(tokenize(document)) for document in documents_corpus]
    terms_set = list(set(sum(documents_corpus_terms, [])))
    terms_number = len(terms_set)
    terms_to_index = {term: i for i, term in enumerate(terms_set)}

    ## Documents vectors
    documents_vectors = []
    for document_terms in tqdm(documents_corpus_terms):
        document_vector = [0]*terms_number

        for term in document_terms:
            term_index = terms_to_index[term]
            term_tf_idf_score = tf_idf_score(term, document_terms, documents_corpus_terms)
            document_vector[term_index] = term_tf_idf_score

        documents_vectors.append(document_vector)

    ## tfidf vector space
    vector_space = pd.DataFrame(documents_vectors).T

    return vector_space, terms_to_index

def tf_idf_vector(document:str, documents_corpus:List[str]) -> pd.Series:
    ## Terms mapping
    documents_corpus_terms = [normalize(tokenize(document)) for document in documents_corpus]
    terms_set = list(set(sum(documents_corpus_terms, [])))
    terms_number = len(terms_set)
    terms_to_index = {term: i for i, term in enumerate(terms_set)}

    ## Compute document vector
    document_vector = [0]*terms_number
    document_terms = normalize(tokenize(document))
    for term in document_terms:
        if term not in terms_to_index:
            continue

        term_index = terms_to_index[term]
        term_tf_idf_score = tf_idf_score(term, document_terms, documents_corpus_terms)
        document_vector[term_index] = term_tf_idf_score

    document_vector = pd.Series(document_vector)
    return document_vector

def knn_vectors(target_vector:np.ndarray, vectors_space:np.ndarray[np.ndarray], k:int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cosine_distances = np.apply_along_axis(lambda vector: distance.cosine(target_vector, vector), 1, vectors_space)
    top_k_indices = np.argpartition(cosine_distances, k)[:k]
    top_k_vectors = vectors_space[top_k_indices]
    top_k_distances = cosine_distances[top_k_indices]

    return top_k_indices, top_k_vectors, top_k_distances

def average_score(reference_scores:List[float], weights:List[float] = None) -> float:
    return np.average(reference_scores, weights=weights)

def classification(reference_categories:List[str]) -> str:
    categories_count = list(Counter(reference_categories).items())
    categories_count.sort(key=lambda cv: cv[1], reverse=True)
    
    top_category, top_count = categories_count[0]

    return top_category if top_count > 1 else 'uncategorized'
