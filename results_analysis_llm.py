"""
Results Analysis for LLM Evaluation Responses

This script processes LLM evaluation response CSV files (one per explanation level)
and computes summary statistics (mean scores) for the quality criteria:
  • Fluency_score
  • Informativeness_score
  • Persuasiveness_score
  • Soundness_score
"""

import pandas as pd

# Helper Functions

def load_explanations(level):
    """
    Load the evaluation responses CSV for a given level.
    
    Parameters:
      level (int): The explanation level (1, 2, or 3).
      
    Returns:
      pd.DataFrame: Loaded DataFrame.
    """
    filename = f"ResultsAnalysis/gpt_evaluation_responses_quality-criteria_level_{level}.csv"
    return pd.read_csv(filename)

def compute_summary(exp1, exp2, exp3, criteria_columns):
    """
    Compute the mean scores for each quality criterion for the three explanation levels.
    
    Parameters:
      exp1, exp2, exp3 (pd.DataFrame): DataFrames for Explanation 1, 2, and 3.
      criteria_columns (list): List of columns to summarize.
      
    Returns:
      pd.DataFrame: Summary DataFrame.
    """
    means1 = exp1[criteria_columns].mean()
    means2 = exp2[criteria_columns].mean()
    means3 = exp3[criteria_columns].mean()
    summary = pd.DataFrame({
         'Explanation Type': ['Explanation 1', 'Explanation 2', 'Explanation 3'],
         'Fluency': [means1['Fluency_score'], means2['Fluency_score'], means3['Fluency_score']],
         'Informativeness': [means1['Informativeness_score'], means2['Informativeness_score'], means3['Informativeness_score']],
         'Persuasiveness': [means1['Persuasiveness_score'], means2['Persuasiveness_score'], means3['Persuasiveness_score']],
         'Soundness': [means1['Soundness_score'], means2['Soundness_score'], means3['Soundness_score']]
    })
    return summary

def filter_explanations(df, rows_to_keep):
    """
    Filter a DataFrame to keep only the specified rows.
    
    Parameters:
      df (pd.DataFrame): The DataFrame to filter.
      rows_to_keep (list): List of row indices to keep.
      
    Returns:
      pd.DataFrame: Filtered DataFrame.
    """
    return df.iloc[rows_to_keep]

# Main Processing Function

def main():
    criteria_columns = ['Fluency_score', 'Informativeness_score', 'Persuasiveness_score', 'Soundness_score']
    
    # Load data for each explanation level (using all rows)
    exp1_all = load_explanations(1)
    exp2_all = load_explanations(2)
    exp3_all = load_explanations(3)
    
    # Overall summary (all rows)
    print("Overall Summary:")
    summary_all = compute_summary(exp1_all, exp2_all, exp3_all, criteria_columns)
    print(summary_all, "\n")

    
    # Summary for "Real" condition
    rows_real = ['presice the indexes']   # list of real news identifiers
    exp1_real = filter_explanations(exp1_all, rows_real)
    exp2_real = filter_explanations(exp2_all, rows_real)
    exp3_real = filter_explanations(exp3_all, rows_real)
    print("Real News Summary:")
    summary_real = compute_summary(exp1_real, exp2_real, exp3_real, criteria_columns)
    print(summary_real, "\n")
    
    # Summary for "Fake" condition
    rows_fake = ['presice the indexes']   # list of fake news identifiers
    exp1_fake = filter_explanations(exp1_all, rows_fake)
    exp2_fake = filter_explanations(exp2_all, rows_fake)
    exp3_fake = filter_explanations(exp3_all, rows_fake)
    print("Fake News Summary:")
    summary_fake = compute_summary(exp1_fake, exp2_fake, exp3_fake, criteria_columns)
    print(summary_fake)

if __name__ == "__main__":
    main()
