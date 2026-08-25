"""
Mini Search Engine - Stage 12 & 14
Information Retrieval Evaluation Metrics (Precision, Recall, F1, MAP, MRR, NDCG)
"""

import math
from typing import List, Set, Dict, Any, Union, Optional


def precision(retrieved: List[str], relevant: Union[List[str], Set[str]]) -> float:
    """
    Calculate Precision: proportion of retrieved documents that are relevant.
    Precision = |Retrieved ∩ Relevant| / |Retrieved|
    """
    if not retrieved:
        return 1.0 if not relevant else 0.0
    relevant_set = set(relevant)
    true_positives = len([doc for doc in retrieved if doc in relevant_set])
    return round(true_positives / len(retrieved), 4)


def recall(retrieved: List[str], relevant: Union[List[str], Set[str]]) -> float:
    """
    Calculate Recall: proportion of relevant documents that are retrieved.
    Recall = |Retrieved ∩ Relevant| / |Relevant|
    """
    if not relevant:
        return 1.0
    relevant_set = set(relevant)
    true_positives = len([doc for doc in retrieved if doc in relevant_set])
    return round(true_positives / len(relevant_set), 4)


def f1_score(p: float, r: float) -> float:
    """
    Calculate F1 Score: harmonic mean of Precision and Recall.
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    if (p + r) == 0:
        return 0.0
    return round(2 * (p * r) / (p + r), 4)


def precision_at_k(retrieved: List[str], relevant: Union[List[str], Set[str]], k: int) -> float:
    """
    Calculate Precision@K: precision measured on the top-K retrieved results.
    Precision@K = |Retrieved[:K] ∩ Relevant| / K
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    true_positives = len([doc for doc in top_k if doc in relevant_set])
    return round(true_positives / k, 4)


def recall_at_k(retrieved: List[str], relevant: Union[List[str], Set[str]], k: int) -> float:
    """
    Calculate Recall@K: recall measured on the top-K retrieved results.
    Recall@K = |Retrieved[:K] ∩ Relevant| / |Relevant|
    """
    if not relevant:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    true_positives = len([doc for doc in top_k if doc in relevant_set])
    return round(true_positives / len(relevant_set), 4)


def average_precision(retrieved: List[str], relevant: Union[List[str], Set[str]]) -> float:
    """
    Calculate Average Precision (AP): rewards placing relevant documents near the top.
    AP = sum(Precision@k * rel(k)) / |Relevant|
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0 if not retrieved else 0.0
    
    running_relevant_count = 0
    precision_sum = 0.0
    
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            running_relevant_count += 1
            precision_at_current_rank = running_relevant_count / rank
            precision_sum += precision_at_current_rank
            
    return round(precision_sum / len(relevant_set), 4)


def reciprocal_rank(retrieved: List[str], relevant: Union[List[str], Set[str]]) -> float:
    """
    Calculate Reciprocal Rank (RR): 1 / rank of the first relevant result.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0 if not retrieved else 0.0

    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return round(1.0 / rank, 4)
            
    return 0.0


def mean_average_precision(ap_list: List[float]) -> float:
    """Calculate Mean Average Precision (MAP) across all queries."""
    if not ap_list:
        return 0.0
    return round(sum(ap_list) / len(ap_list), 4)


def mean_reciprocal_rank(rr_list: List[float]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) across all queries."""
    if not rr_list:
        return 0.0
    return round(sum(rr_list) / len(rr_list), 4)


# --- Stage 14: Normalized Discounted Cumulative Gain (NDCG) ---

def discounted_cumulative_gain(
    retrieved: List[str], 
    relevant: Union[List[str], Set[str]], 
    k: Optional[int] = None
) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG@K):
    DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
    """
    cutoff = k if k is not None else len(retrieved)
    top_k = retrieved[:cutoff]
    relevant_set = set(relevant)
    
    dcg = 0.0
    for rank, doc in enumerate(top_k, start=1):
        rel = 1.0 if doc in relevant_set else 0.0
        if rel > 0:
            dcg += (rel / math.log2(rank + 1))
            
    return dcg


def ideal_discounted_cumulative_gain(
    relevant: Union[List[str], Set[str]], 
    k: Optional[int] = None
) -> float:
    """
    Calculate Ideal Discounted Cumulative Gain (IDCG@K):
    IDCG@K = sum_{i=1}^{min(K, |Relevant|)} 1.0 / log2(i + 1)
    """
    num_rel = len(set(relevant))
    if num_rel == 0:
        return 0.0
    
    cutoff = min(k, num_rel) if k is not None else num_rel
    idcg = 0.0
    for rank in range(1, cutoff + 1):
        idcg += (1.0 / math.log2(rank + 1))
        
    return idcg


def ndcg_at_k(
    retrieved: List[str], 
    relevant: Union[List[str], Set[str]], 
    k: int = 5
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K):
    NDCG@K = DCG@K / IDCG@K
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0 if not retrieved else 0.0
        
    dcg = discounted_cumulative_gain(retrieved, relevant, k=k)
    idcg = ideal_discounted_cumulative_gain(relevant, k=k)
    
    if idcg <= 0.0:
        return 0.0
        
    return round(dcg / idcg, 4)


def mean_ndcg_at_k(ndcg_list: List[float]) -> float:
    """Calculate Mean NDCG across all queries."""
    if not ndcg_list:
        return 0.0
    return round(sum(ndcg_list) / len(ndcg_list), 4)


def calculate_confusion_matrix(
    retrieved: List[str], 
    relevant: Union[List[str], Set[str]], 
    all_documents: Union[List[str], Set[str]]
) -> Dict[str, int]:
    """Compute True Positives, False Positives, False Negatives, True Negatives."""
    ret_set = set(retrieved)
    rel_set = set(relevant)
    all_set = set(all_documents)
    
    tp = len(ret_set & rel_set)
    fp = len(ret_set - rel_set)
    fn = len(rel_set - ret_set)
    tn = len(all_set - (ret_set | rel_set))
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    }
