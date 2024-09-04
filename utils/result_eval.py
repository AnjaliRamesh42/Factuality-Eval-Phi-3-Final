import sys
import ujson as json
import re
import string
from collections import Counter

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def eval(prediction_file, type):
    with open(prediction_file) as f:
        data = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    N = len(data)

    for cur_id, values in data.items():
        generated_answer = values['generated_answer']
        gold_truth_answer = values['gold_truth_answer']
        
        em, prec, recall = update_answer(metrics, generated_answer, gold_truth_answer)

    for k in metrics.keys():
        metrics[k] /= N

    with open(f'/Users/anjali/Imperial/dissertation/results/metrics_{type}.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    print(metrics)

# if __name__ == '__main__':
#     eval('/Users/anjali/Imperial/dissertation/utils/updated_answers.json', 'base-retrieval-system')