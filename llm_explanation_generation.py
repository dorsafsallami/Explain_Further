"""
Explanation Generation for Fake News Detection using LLMs

This script generates two levels of explanations for fake news predictions:
  
  Level 2: Converts LIME feature contribution strings (from the 'lime_features' column)
           into simple narrative explanations.
  
  Level 3: Provides a step-by-step explanation of why a news item (from the 'Title' column)
           is classified as Fake or Real (using the 'BERT_Label' column).

Both levels are generated using GPT-3.5 and GPT-4 via the OpenAI API.
  
Usage:
  python explanation_generation.py --input_file Examples.csv --output_file Examples.csv
  
Make sure to set your OpenAI API key by replacing the placeholder below or by setting the
environment variable OPENAI_API_KEY.
"""

import os
import pandas as pd
import json
import argparse
from openai import OpenAI

# Set your OpenAI API key here or set the environment variable externally.
os.environ['OPENAI_API_KEY'] = 'Replace with your actual OpenAI API key'

client = OpenAI()


def query_gpt(prompt, question, model, n=3, temperature=0.8, max_tokens=400, json_storage=None, replace=False):
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
    responses = []
    for choice in response.choices:
        responses.append(choice.message.content)
    return responses

def get_first_response(response_storage, prompt, question):
    if prompt in response_storage and question in response_storage[prompt]:
        return response_storage[prompt][question][0]
    return None


# Level 2 Explanation (LIME-based Narrative)

LEVEL2_PROMPT = (
    "You are assisting users with no ML experience to understand an ML model's predictions regarding fake news detection. "
    "I will provide you with LIME feature contribution explanations in the format (feature_name, contribution). "
    "Convert the explanation into a simple narrative. Please provide a clear, concise, and understandable narrative that "
    "clearly explains the model's decision. Ensure the explanation is easy to understand and uses the fewest tokens possible."
)

def generate_level2_explanation(row, model_name):
    """
    Generate a level 2 explanation for a given row using the value from the 'lime_features' column.
    """
    storage = {} 
    return query_gpt(LEVEL2_PROMPT, row['lime_features'], model=model_name, json_storage=storage, replace=False)


# Level 3 Explanation (Step-by-Step Reasoning)

LEVEL3_PROMPT = (
    "You are assisting users with no ML experience to understand why a news item is classified as Fake or Real. "
    "I will provide you with the news and its classification. Please provide a step-by-step explanation of why "
    "the news is classified that way. Keep it simple, concise, and easy to understand."
)

QUESTION_TEMPLATE_LEVEL3 = "News: {}\nLabel: {}\n\nAnswer: Let's think step by step."

def generate_level3_explanation(row, model_name):
    """
    Generate a level 3 explanation for a given row using the 'Title' and 'BERT_Label' columns.
    """
    tweet = row['Title']
    label = row['BERT_Label'] 
    question = QUESTION_TEMPLATE_LEVEL3.format(tweet, label)
    storage = {}
    return query_gpt(LEVEL3_PROMPT, question, model=model_name, json_storage=storage, replace=False)


# Processing and File I/O

def process_explanations(input_file, output_file):
    """
    Reads the input CSV file, generates level 2 and level 3 explanations using both GPT-3.5 and GPT-4,
    and writes the results back to the CSV file.
    """
    df = pd.read_csv(input_file)

    # Generate Level 2 explanations
    df['gpt3.5_response_level_2'] = df.apply(lambda row: generate_level2_explanation(row, model_name="gpt-3.5-turbo"), axis=1)
    df['gpt4_response_level_2']   = df.apply(lambda row: generate_level2_explanation(row, model_name="gpt-4"), axis=1)

    # Generate Level 3 explanations
    df['gpt3.5_response_level_3'] = df.apply(lambda row: generate_level3_explanation(row, model_name="gpt-3.5-turbo"), axis=1)
    df['gpt4_response_level_3']   = df.apply(lambda row: generate_level3_explanation(row, model_name="gpt-4"), axis=1)

    df.to_csv(output_file, index=False)
    print("Explanations generated and saved to", output_file)
    print(df)

def main():
    parser = argparse.ArgumentParser(description="LLM Explanation Generation for Fake News Detection")
    parser.add_argument("--input_file", type=str, default="Examples.csv", help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, default="Examples.csv", help="Path to the output CSV file")
    args = parser.parse_args()
    process_explanations(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
