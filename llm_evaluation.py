"""
LLM Evaluation of Explanation Quality for Fake News Detection

This script evaluates explanation quality using an OpenAI LLM. It processes an input CSV (e.g., Examples.csv)
and, for each evaluation level, generates quality assessments based on the following criteria:
  • Fluency
  • Informativeness
  • Persuasiveness
  • Soundness

The evaluations are performed at three levels:
  - Level 1: Uses the column 'lime_features'
  - Level 2: Uses the column 'gpt4_response_level_2'
  - Level 3: Uses the column 'gpt4_response_level_3'

The LLM response is parsed via regular expressions to extract a score (1-5) and an explanation for each metric.
"""

import os
import re
import json
import pandas as pd
import numpy as np
from openai import OpenAI

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'Replace with your actual OpenAI API key'
client = OpenAI()

# Global variables for caching and control
gpt4o_json = {}
REPLACE = False

# Load the input CSV file
df = pd.read_csv('Examples.csv')

# LLM Evaluation Prompt and Question Template
LLM_EVALUATION_PROMPT_TEMPLATE = """
You will be provided with an explanation for a detected fake news article.
Your task is to rate the explanation based on multiple metrics.

Evaluation Criteria:

Fluency (1-5): Assesses whether the explanation follows proper grammar and structural rules.
Informativeness (1-5): Measures how well the explanation provides new information and context.
Persuasiveness (1-5): Evaluates whether the explanation is convincing.
Soundness (1-5): Describes whether the explanation is valid and logically sound.

Evaluation Steps:
1. Read the provided explanation.
2. For each criterion, assign a score from 1 to 5 and briefly explain your rating.
"""

question_template = "explanation: {explanation}"

# LLM Query and Response Handling Functions

def query_gpt_evaluation(prompt, question, model="gpt-4o", n=3, temperature=0.8, max_tokens=400, json_storage=None, replace=False):
    """Query the LLM evaluation using a prompt and question."""
    if json_storage is not None and not replace and prompt in json_storage and question in json_storage[prompt]:
        print("SKIPPING GENERATION BECAUSE replace=False\n")
        return get_first_response(json_storage, prompt, question)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        n=n
    )
    if json_storage is not None:
        if prompt not in json_storage:
            json_storage[prompt] = {}
        json_storage[prompt][question] = get_responses(response)
    return get_first_response(json_storage, prompt, question)

def get_responses(response):
    """Extract responses from the LLM API output."""
    responses = []
    for choice in response.choices:
        responses.append(choice.message.content)
    return responses

def get_first_response(response_storage, prompt, question):
    """Return the first response stored for a given prompt and question."""
    if prompt in response_storage and question in response_storage[prompt]:
        return response_storage[prompt][question][0]
    return None

# Response Parsing Function

def parse_gpt_response(response):
    """
    Parse the LLM response to extract scores and explanations for each metric.
    Expected format in the response:
      Fluency: <score> <explanation>
      Informativeness: <score> <explanation>
      Persuasiveness: <score> <explanation>
      Soundness: <score> <explanation>
    """
    fluency_match = re.search(r"Fluency:\s*(\d)\s*(.*?)(?=Informativeness|$)", response, re.DOTALL)
    informativeness_match = re.search(r"Informativeness:\s*(\d)\s*(.*?)(?=Persuasiveness|$)", response, re.DOTALL)
    persuasiveness_match = re.search(r"Persuasiveness:\s*(\d)\s*(.*?)(?=Soundness|$)", response, re.DOTALL)
    soundness_match = re.search(r"Soundness:\s*(\d)\s*(.*?)(?=$)", response, re.DOTALL)

    metrics = {
        'Fluency_score': int(fluency_match.group(1)) if fluency_match else None,
        'Fluency_explanation': fluency_match.group(2).strip() if fluency_match else None,
        'Informativeness_score': int(informativeness_match.group(1)) if informativeness_match else None,
        'Informativeness_explanation': informativeness_match.group(2).strip() if informativeness_match else None,
        'Persuasiveness_score': int(persuasiveness_match.group(1)) if persuasiveness_match else None,
        'Persuasiveness_explanation': persuasiveness_match.group(2).strip() if persuasiveness_match else None,
        'Soundness_score': int(soundness_match.group(1)) if soundness_match else None,
        'Soundness_explanation': soundness_match.group(2).strip() if soundness_match else None
    }
    return metrics

# Processing Routine for Evaluation Levels

def process_level(df, explanation_column, output_csv):
    """
    Process a given evaluation level.
    For each row in the DataFrame, retrieve the explanation from the specified column,
    query the LLM for evaluation, parse the response, and save the results.
    """
    parsed_results = []
    for index, row in df.iterrows():
        explanation = row[explanation_column]
        question = question_template.format(explanation=explanation)
        response = query_gpt_evaluation(LLM_EVALUATION_PROMPT_TEMPLATE, question, json_storage=gpt4o_json, replace=REPLACE)
        parsed_results.append(parse_gpt_response(response))
    parsed_df = pd.DataFrame(parsed_results)
    parsed_df.to_csv(output_csv, index=False)
    print(f"Sample results from {explanation_column}:\n", parsed_df.head())
    return parsed_df

def main():
    # Level 1 Evaluation: Using 'lime_features' column
    process_level(df, 'lime_features', 'gpt_evaluation_responses_quality-criteria_level_1.csv')
    # Level 2 Evaluation: Using 'gpt4_response_level_2' column
    process_level(df, 'gpt4_response_level_2', 'gpt_evaluation_responses_quality-criteria_level_2.csv')
    # Level 3 Evaluation: Using 'gpt4_response_level_3' column
    process_level(df, 'gpt4_response_level_3', 'gpt_evaluation_responses_quality-criteria_level_3.csv')

if __name__ == "__main__":
    main()
