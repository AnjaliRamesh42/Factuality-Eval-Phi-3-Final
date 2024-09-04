import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
import random
import os
import json

def random_sampling(queries, sample_size):
    # Ensure sample size is not larger than the available number of queries
    if sample_size > len(queries):
        sample_size = len(queries)
    
    # Randomly sample query IDs
    sampled_query_ids = random.sample(list(queries.keys()), sample_size)
    
    # Create a new dictionary with the sampled queries
    sampled_queries = {qid: queries[qid] for qid in sampled_query_ids}
    
    return sampled_queries


def compute_query_length(query):
    """Compute the length of the query."""
    return len(query)  # length based on number of characters


def stratified_sampling(queries, sample_size_per_bin, bin_size):
    # Calculate the length of each query (in words)
    # query_lengths = {qid: len(query.split()) for qid, query in queries.items()}
    query_lengths = {qid: len(query) for qid, query in queries.items()}

    
    # Group queries by their length ranges
    bins = defaultdict(list)
    for qid, length in query_lengths.items():
        bin_range = (length - 1) // bin_size * bin_size + 1
        bins[bin_range].append(qid)
    
    # Print the number of bins and the word length ranges they contain
    # print(f"Number of bins: {len(bins)}")
    # print("Word length ranges and their respective number of queries:")
    for bin_range, qids in bins.items():
        bin_end = bin_range + bin_size - 1
        # print(f"Length {bin_range}-{bin_end} words: {len(qids)} queries")
    
    # Sample from each bin
    sampled_queries = {}
    for bin_range, qids in bins.items():
        if len(qids) > sample_size_per_bin:
            sampled_qids = random.sample(qids, sample_size_per_bin)
        else:
            sampled_qids = qids  # Take all if less than the sample size
        sampled_queries.update({qid: queries[qid] for qid in sampled_qids})
    
    return sampled_queries

def cross_validation_sampling(queries, num_folds, sample_size):
    """Generate multiple folds of samples."""
    query_ids = list(queries.keys())
    kf = KFold(n_splits=num_folds)
    
    folds = []
    for train_index, test_index in kf.split(query_ids):
        fold = {qid: queries[qid] for qid in np.array(query_ids)[test_index]}
        folds.append(fold)
    
    return folds


def calculate_metric_variance(metric_files):
    """
    Calculates the variance for each metric across the 5 folds.

    :param metric_files: List of file paths to the metric JSON files.
    :return: A dictionary with the variance for each metric.
    """
    metrics = {"em": [], "f1": [], "prec": [], "recall": []}

    # Load the metrics from each fold
    for file in metric_files:
        with open(file, 'r') as f:
            fold_metrics = json.load(f)
            for key in metrics.keys():
                if key in fold_metrics:
                    metrics[key].append(fold_metrics[key])

    # Calculate the variance for each metric
    metric_variance = {}
    for key, values in metrics.items():
        metric_variance[key] = np.var(values, ddof=1)  # ddof=1 for sample variance

    return metric_variance

def get_metric_files(directory, sampling_method):
    """
    Retrieves the list of metric files for the given sampling method.

    :param directory: Directory where the metric JSON files are stored.
    :param sampling_method: Sampling method name to filter files.
    :return: List of file paths to the metric JSON files.
    """
    return [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if sampling_method in filename
    ]


