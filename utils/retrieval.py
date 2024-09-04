import torch
from tqdm import tqdm
import numpy as np

def encode_corpus(corpus, model, batch_size=32):
    corpus_ids = list(corpus.keys())
    corpus_texts = [doc['text'] for doc in corpus.values()]
    corpus_embeddings = []
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Encoding Corpus"):
        batch_texts = corpus_texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)
        corpus_embeddings.append(batch_embeddings)
    corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
    return corpus_ids, corpus_embeddings

def encode_queries(queries, model, batch_size=32):
    query_ids = list(queries.keys())
    query_texts = list(queries.values())
    query_embeddings = []
    for i in tqdm(range(0, len(query_texts), batch_size), desc="Encoding Queries"):
        batch_texts = query_texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)
        query_embeddings.append(batch_embeddings)
    query_embeddings = torch.cat(query_embeddings, dim=0)
    return query_ids, query_embeddings

def save_embeddings(ids, embeddings, file_prefix):
    np.save(f"./embeddings/{file_prefix}_ids.npy", np.array(ids))
    np.save(f"./embeddings/{file_prefix}_embeddings.npy", embeddings.cpu().numpy())

def load_embeddings(file_prefix):
    ids = np.load(f"./embeddings/{file_prefix}_ids.npy", allow_pickle=True).tolist()
    embeddings = torch.tensor(np.load(f"./embeddings/{file_prefix}_embeddings.npy"))
    return ids, embeddings

def load_corpus_embeddings_for_evaluation():
    print("Loading embeddings...")
    corpus_ids, corpus_embeddings = load_embeddings("corpus_embeddings")
    print("Embeddings loaded.")
    return corpus_ids, corpus_embeddings

def load_query_embeddings_for_evaluation():
    print("Loading embeddings...")
    query_ids, query_embeddings = load_embeddings("query_embeddings")
    print("Embeddings loaded.")
    return query_ids, query_embeddings

def retrieve(corpus_ids, corpus_embeddings, query_ids, query_embeddings, top_k):
    results = {}
    for i, query_embedding in tqdm(enumerate(query_embeddings), desc="Retrieving Documents", total=len(query_embeddings)):
        scores = torch.matmul(query_embedding, corpus_embeddings.T)
        sorted_indices = torch.argsort(scores, descending=True)[:top_k]
        results[query_ids[i]] = [(corpus_ids[idx], scores[idx].item()) for idx in sorted_indices]
    return results

# def retrieve_single_query(corpus_ids, corpus_embeddings, query_id, query_embedding, top_k, similarity_threshold=0.5):

#     scores = torch.matmul(query_embedding, corpus_embeddings.T)
#     sorted_indices = torch.argsort(scores, descending=True)

#     top_results = []
#     for idx in sorted_indices:
#         if idx >= len(corpus_ids):
#             print(f"Index {idx} is out of bounds for corpus_ids with length {len(corpus_ids)}")
#             continue

#         score = scores[idx].item()
#         if score >= similarity_threshold:
#             top_results.append((corpus_ids[idx], score))
#             if len(top_results) == top_k:
#                 break
#         else:
#             break

#     return top_results

def retrieve_single_query_complex(corpus_ids, corpus_embeddings, query_id, query_embedding, top_k, similarity_threshold=0.5):

    scores = torch.matmul(query_embedding, corpus_embeddings.T)
    sorted_indices = torch.argsort(scores, descending=True)

    top_results = []
    for idx in sorted_indices:
        if idx >= len(corpus_ids):
            print(f"Index {idx} is out of bounds for corpus_ids with length {len(corpus_ids)}")
            continue

        score = scores[idx].item()
        if score >= similarity_threshold:
            top_results.append((corpus_ids[idx], score))
            if len(top_results) == top_k:
                break
        else:
            break
      
       # Ensure at least one document is retrieved
    if len(top_results) == 0:
        # Retrieve the top-ranked document even if it doesn't meet the similarity threshold
        valid_index = sorted_indices[sorted_indices < len(corpus_ids)][0]
        top_results.append((corpus_ids[valid_index], scores[valid_index].item()))

    return top_results

def retrieve_single_query_simple(corpus_ids, corpus_embeddings, query_id, query_embedding, top_k, similarity_threshold=0.5):

    scores = torch.matmul(query_embedding, corpus_embeddings.T)
    
    sorted_indices = torch.argsort(scores, descending=True)[:top_k]

    top_results = [(corpus_ids[idx], scores[idx].item()) for idx in sorted_indices]

    return top_results