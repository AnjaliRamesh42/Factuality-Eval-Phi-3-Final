import json

# Load JSON data (assuming this is in a file named 'generated_answers.json')
with open('/Users/anjali/Imperial/dissertation/results/base-retrieval-system_generated_answers.json', 'r') as file:
    generated_answers = json.load(file)

# Load JSONL data (assuming this is in a file named 'gold_answers.jsonl')
id_to_gold_answer = {}
with open('/Users/anjali/Imperial/dissertation/datasets/hotpotqa/queries.jsonl', 'r') as file:
    for line in file:
        item = json.loads(line)
        id_to_gold_answer[item["_id"]] = item["metadata"]["answer"]

# Update the first JSON object with gold truth answers
for key in generated_answers:
    if key in id_to_gold_answer:
        generated_answers[key] = {
            "generated_answer": generated_answers[key],
            "gold_truth_answer": id_to_gold_answer[key]
        }

# Save the updated JSON data (assuming you want to save it to 'updated_answers.json')
with open('/results/updated_answers.json', 'w') as file:
    json.dump(generated_answers, file, indent=4)

# Print the updated results
# print(json.dumps(generated_answers, indent=4))