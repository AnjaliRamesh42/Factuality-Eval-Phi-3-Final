# Project Name

## Description

A brief description of the project.

## Directory Structure

- `data/`: Contains data loading modules.
- `models/`: Contains model loading and initialization modules.
- `utils/`: Contains utility functions for retrieval, evaluation, and metrics.
- `hotpot_evaluation.ipynb`: Notebook to run the evaluation pipeline on HotpotQA.
- `triviaqa_evaluation.ipynb`: Notebook to run the evaluation pipeline TriviaQA.

## Requirements

List of required packages can be found in `requirements.txt`

## Usage

1. Clone the repository.
2. Install the required packages: `pip install -r requirements.txt`.
3. In either hotpot_evaluation.ipynb or triviaqa_evaluation.ipynb, replace the filepaths in the indicated cells with yours for the following variables: (files can be found in a Google Drive folder, linked here: https://drive.google.com/drive/folders/1libLcirWGIarxHtCYRQt9NpgOViZ9Gfw?usp=sharing)
      - `data_path` = 
      - `corpus_embeddings_path`
      - `corpus_ids_path `
      - `query_embedding_path`
      - `gold_truth_path`
      - `classifier_model_path`
      - `full_wiki_path`
      - `hotpot_prompt_path`
