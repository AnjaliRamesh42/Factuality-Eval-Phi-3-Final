import hashlib
def create_corpus(dataset):
    corpus = {}
    for item in dataset:
        search_results = item.get('search_results', {})
        descriptions = search_results.get('description', [])
        titles = search_results.get('title', [])

        for i in range(len(descriptions)):
            doc_id = hashlib.md5(descriptions[i].encode()).hexdigest()  # Create a unique ID
            if doc_id not in corpus:
                corpus[doc_id] = {
                    "title": titles[i] if i < len(titles) else "",  # Handle missing titles
                    "text": descriptions[i]
                }
    return corpus

def create_queries(dataset):
    queries = {}
    for i, item in enumerate(dataset):
        query_id = item['question_id']  # Use the original question ID from the dataset
        queries[query_id] = item['question']
    return queries

def create_qrels(dataset, corpus):
    qrels = {}
    for i, item in enumerate(dataset):
        query_id = f"query_{i}"
        qrels[query_id] = {}
        search_results = item.get('search_results', {})
        descriptions = search_results.get('description', [])

        for desc in descriptions:
            doc_id = hashlib.md5(desc.encode()).hexdigest()
            if doc_id in corpus:
                qrels[query_id][doc_id] = 1
    return qrels