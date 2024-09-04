import ollama
from ollama import generate
import requests 
import json
import os
from groq import Groq

os.environ["GROQ_API_KEY"] = "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"


def format_input(passages_texts, question):
    formatted_passages = ""
    for i, passage in enumerate(passages_texts, 1):
        formatted_passages += f"Document [{i}]: {passage}\n"
    formatted_input = f"{formatted_passages}\nQuestion: {question}\nAnswer:"
    return formatted_input

def generate_answer(question, passages_texts):
    with open('/Users/anjali/Imperial/dissertation/prompts/hotpotqa_prompt.txt', 'r') as file:
        prompt_template = file.read()
    
    formatted_passages = ""
    for i, passage in enumerate(passages_texts, 1):
        formatted_passages += f"Document [{i}] {passage}\n"
    
    input_text = prompt_template.format(search_results=formatted_passages, question=question)

    client = Groq(api_key="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")

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": input_text,
        }
    ],
    model="llama3-8b-8192",
    )

    answer = chat_completion.choices[0].message.content
    return answer

# def save_answers_to_json(generated_answers, filename, update_existing=True):
#     if update_existing and os.path.exists(filename):
#         with open(filename, 'r') as file:
#             existing_data = json.load(file)
#         existing_data.update(generated_answers)
#         with open(filename, 'w') as file:
#             json.dump(existing_data, file, indent=4)
#     else:
#         with open(filename, 'w') as file:
#             json.dump(generated_answers, file, indent=4)

def save_answers_to_json(generated_answers, filename, append=True):
    if append:
        try:
            with open(filename, 'r') as f:
                existing_answers = json.load(f)
        except FileNotFoundError:
            existing_answers = {}
        
        existing_answers.update(generated_answers)

        with open(filename, 'w') as f:
            json.dump(existing_answers, f, indent=4)
    else:
        with open(filename, 'w') as f:
            json.dump(generated_answers, f, indent=4)

def map_jsons(generated_answers_filename, gold_truth_answers_filename, title):
    # Load JSON data (assuming this is in a file named 'generated_answers.json')
    with open(generated_answers_filename, 'r') as file:
        generated_answers = json.load(file)

    # Load JSONL data (assuming this is in a file named 'gold_answers.jsonl')
    id_to_gold_answer = {}
    with open(gold_truth_answers_filename, 'r') as file:
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
    with open(f'./{title}_updated_answers.json', 'w') as file:
        json.dump(generated_answers, file, indent=4)



def evaluate_answers(filename):

    # Load the JSON data from file
    with open(filename, 'r') as file:
        data = json.load(file)

    # Initialize counters
    total_answers = 0
    matched_answers = 0

    # Iterate through the dictionary and check if gold_truth_answer is in generated_answer
    for key, value in data.items():
        total_answers += 1
        # Normalize both answers by stripping leading/trailing spaces and converting to lowercase
        generated_answer = value['generated_answer'].strip().lower()
        gold_truth_answer = value['gold_truth_answer'].strip().lower()
        
        # Check if the gold_truth_answer is a substring of the generated_answer
        if gold_truth_answer in generated_answer:
            matched_answers += 1

    # Calculate accuracy
    accuracy = (matched_answers / total_answers) * 100

    # Print results
    print(f"Total Answers: {total_answers}")
    print(f"Matched Answers: {matched_answers}")
    print(f"Accuracy: {accuracy:.2f}%")


# generate_answer("What is the capital of France?", ["France is a country in Europe.", "Paris is the capital of France.", "France is known for its wine and cheese."])