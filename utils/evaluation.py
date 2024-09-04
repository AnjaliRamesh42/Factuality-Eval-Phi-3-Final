import numpy as np
from utils.metrics import precision_at_k, recall_at_k, ndcg_at_k

def evaluate_results_at_k(results, queries, corpus, top_k, qrels):
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for query_id, ranking_scores in results.items():
        relevant_docs = set(qrels.get(query_id, {}))
        precision_scores.append(precision_at_k(ranking_scores, relevant_docs, top_k))
        recall_scores.append(recall_at_k(ranking_scores, relevant_docs, top_k))
        ndcg_scores.append(ndcg_at_k(ranking_scores, relevant_docs, top_k))

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_ndcg = np.mean(ndcg_scores)

    print(f"Average Precision@{top_k}: {avg_precision}")
    print(f"Average Recall@{top_k}: {avg_recall}")
    print(f"Average NDCG@{top_k}: {avg_ndcg}")

def evaluate_results_multiple_ks(results, queries, corpus, top_ks, qrels):
    for top_k in top_ks:
        evaluate_results_at_k(results, queries, corpus, top_k, qrels)