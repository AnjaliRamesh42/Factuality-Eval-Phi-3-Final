import numpy as np

def precision_at_k(ranking_scores, relevant_docs, k):
    retrieved_docs = [doc_id for doc_id, _ in ranking_scores[:k]]
    relevant_retrieved_docs = len(set(retrieved_docs) & relevant_docs)
    return relevant_retrieved_docs / k

def recall_at_k(ranking_scores, relevant_docs, k):
    retrieved_docs = [doc_id for doc_id, _ in ranking_scores[:k]]
    relevant_retrieved_docs = len(set(retrieved_docs) & relevant_docs)
    return relevant_retrieved_docs / len(relevant_docs) if relevant_docs else 0

def ndcg_at_k(ranking_scores, relevant_docs, k):
    # Compute the ideal DCG
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant_docs), k))])
    
    # If idcg is zero, return 0.0 to avoid division by zero
    if idcg == 0:
        return 0.0
    
    # Compute the DCG for the actual ranking
    dcg = 0.0
    for i, (doc_id, score) in enumerate(ranking_scores[:k]):
        if doc_id in relevant_docs:
            dcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg