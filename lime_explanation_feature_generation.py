import numpy as np
import torch
import pandas as pd
import argparse
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer


class ModelWrapper:
    """Wrapper class for loading and using the BERT model"""
    def __init__(self, model_name, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict_label(self, text):
        """Predict whether text is fake or real"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1).detach().numpy()
        return "Fake" if probabilities[0][1] > 0.5 else "Real"

    def predict_probabilities(self, texts):
        """Get prediction probabilities for a batch of texts"""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=1).detach().numpy()

    

def generate_lime_explanation(model_wrapper, text, num_features=5, include_average=False):
    """Generate LIME explanation for a given text"""
    explainer = LimeTextExplainer(class_names=["Fake", "Real"])
    
    explanation = explainer.explain_instance(
        text,
        classifier_fn=model_wrapper.predict_probabilities,
        num_samples=1000,
        num_features=num_features
    )

    features = []
    exp_list = explanation.as_list()
    num_features = num_features or len(exp_list)
    
    for i in range(num_features):
        feature_name, contribution = exp_list[i]
        if include_average:
            features.append(f"({feature_name.strip()}, {contribution}, {exp_list[i][-1]})")
        else:
            features.append(f"({feature_name.strip()}, {contribution})")
    
    return ", ".join(features)


def process_data(data_file, output_file, model_wrapper):
    """Process data and save results with predictions and explanations"""
    df = pd.read_excel(data_file)
    df['BERT_Label'] = df['Title'].apply(model_wrapper.predict_label)
    df['lime_features'] = df['Title'].apply(
        lambda x: generate_lime_explanation(model_wrapper, x, num_features=5)
    )
    df.to_csv(output_file, index=False)


def demonstrate_example(model_wrapper, example_text):
    """Demonstrate LIME explanation for an example text"""
    explainer = LimeTextExplainer(class_names=["Fake", "Real"])
    
    explanation = explainer.explain_instance(
        example_text,
        classifier_fn=model_wrapper.predict_probabilities,
        num_samples=1000,
        num_features=5
    )
    
    print("Prediction probabilities: 0 -> Real, 1 -> Fake")
    explanation.show_in_notebook(text=True)

    

def get_args():
    parser = argparse.ArgumentParser(description="Fake News Detection with BERT and LIME")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Name of the model (default: bert-base-uncased)")
    parser.add_argument("--model_path", type=str, default="Fake_news_detction/saved_models/bert/pytorch_model.bin",
                        help="Path to the model weights")
    parser.add_argument("--data_file", type=str, default="Snopes.xlsx",
                        help="Path to the Excel data file")
    parser.add_argument("--output_file", type=str, default="Examples.csv",
                        help="Path to the output CSV file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Initialize model with provided inputs
    news_detector = ModelWrapper(args.model_name, args.model_path)
    
    # Process dataset and save output
    process_data(args.data_file, args.output_file, news_detector)
    
    # Generate LIME explanation for each example from the output CSV
    df_examples = pd.read_csv(args.output_file)
    for idx, row in df_examples.iterrows():
        example_text = row['Title']  # Adjust the column name if necessary
        print(f"\n=== LIME Explanation for example {idx} ===")
        demonstrate_example(news_detector, example_text)
