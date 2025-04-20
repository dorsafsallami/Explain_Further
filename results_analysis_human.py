"""
This script processes evaluation CSV files to:
 • Normalize and rename columns based on news articles and evaluation questions.
 • Split the data into separate files (No Explanation, Explanation 1, 2, and 3).
 • Correct numbering of news ID columns.
 • Split each explanation file into fake and real news subsets.
 • Compute quality metrics (Fluency, Informativeness, Persuasiveness, Soundness) for overall, fake, and real news.
 • Calculate truthfulness accuracy and perform paired t-tests.
 • Analyze misclassifications and plot misclassification counts.
 • Measure the effect of explanation length via word counts.
 """

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel



# Utility Functions

def normalize_text(text):
    """Convert text to lowercase, replace non-alphanumeric characters with spaces, and collapse spaces."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text.lower())).strip()

def create_news_mapping(news_articles, news_labels):
    """Creates a mapping from news ID to article and label."""
    news_mapping = {}
    for idx, (article, label) in enumerate(zip(news_articles, news_labels), start=1):
        news_mapping[idx] = {'article': article, 'label': label}
    return news_mapping

def rename_columns(df, news_articles, news_labels, column_mapping):
    """Rename DataFrame columns using normalized news articles and question-to-category mapping."""
    # Normalize DataFrame column names.
    df_columns_normalized = {col: normalize_text(col) for col in df.columns}
    normalized_article_to_id = {
        normalize_text(article): f'news_id_{idx}' 
        for idx, (article, _) in enumerate(zip(news_articles, news_labels), start=1)
    }
    column_rename_mapping = {
        original: normalized_article_to_id[norm] 
        for original, norm in df_columns_normalized.items() if norm in normalized_article_to_id
    }
    df.rename(columns=column_rename_mapping, inplace=True)
    # Rename columns based on evaluation question mapping.
    new_columns = {}
    for col in df.columns:
        for key, value in column_mapping.items():
            if col.startswith(key):
                new_columns[col] = value
    df.rename(columns=new_columns, inplace=True)
    return df

def split_dataframe_by_columns(df):
    """Split the main DataFrame into No Explanation and Explanation parts; save intermediate files."""
    df_NoExp = df.iloc[:, 0:10]
    explanation1 = df.iloc[:, 10:61]
    explanation2 = df.iloc[:, 61:112]
    explanation3 = df.iloc[:, 112:]
    
    df_NoExp.to_csv(f'{RESULTS_ANALYSIS_DIR}/df_NoExp.csv', index=False)
    explanation1.to_csv(f'{RESULTS_ANALYSIS_DIR}/explanation1.csv', index=False)
    explanation2.to_csv(f'{RESULTS_ANALYSIS_DIR}/explanation2.csv', index=False)
    explanation3.to_csv(f'{RESULTS_ANALYSIS_DIR}/explanation3.csv', index=False)
    return df_NoExp, explanation1, explanation2, explanation3

def fix_news_id_columns(df, total_ids=10):
    """Fix numbering of news_id columns to range from news_id_1 to news_id_total_ids."""
    news_id_columns = [f'news_id_{i+1}' for i in range(total_ids)]
    df.columns = df.columns.str.replace('Do you believe this news is:', 'news_id_')
    counter = 0
    new_column_names = []
    for col in df.columns:
        if 'news_id_' in col:
            if counter < total_ids:
                new_column_names.append(news_id_columns[counter])
                counter += 1
            else:
                new_column_names.append(col)
        else:
            new_column_names.append(col)
    df.columns = new_column_names
    return df

def process_explanation_file(file_path, output_path, total_ids=10):
    """Load an explanation CSV, fix news_id columns, and save the updated file."""
    df = pd.read_csv(file_path)
    df = fix_news_id_columns(df, total_ids)
    df.to_csv(output_path, index=False)
    return df

def get_news_with_columns(df, news_id):
    """Return the list of columns associated with a given news_id (four preceding columns plus the news_id)."""
    idx = df.columns.get_loc(news_id)
    start_idx = max(idx - 4, 0)
    return list(df.columns[start_idx:idx+1])

def split_fake_real_data(df, fake_news_ids, real_news_ids):
    """Split the DataFrame into fake and real news data using specified news_ids."""
    fake_columns = []
    real_columns = []
    for news_id in fake_news_ids:
        fake_columns.extend(get_news_with_columns(df, news_id))
    for news_id in real_news_ids:
        real_columns.extend(get_news_with_columns(df, news_id))
    fake_data = df[fake_columns]
    real_data = df[real_columns]
    return fake_data, real_data

def calculate_means_for_criteria(df, criteria):
    """Calculate the mean value for each criterion based on columns that start with the criterion name."""
    mean_values = {}
    for criterion in criteria:
        criterion_columns = [col for col in df.columns if col.startswith(criterion)]
        mean_values[criterion] = df[criterion_columns].mean().mean() if criterion_columns else np.nan
    return mean_values

def calculate_accuracy(df, fake_cols, real_cols):
    """Calculate classification accuracy for fake and real news."""
    df['correct_fake'] = df[fake_cols].apply(lambda x: (x == 'Fake').mean(), axis=1)
    df['correct_real'] = df[real_cols].apply(lambda x: (x == 'Real').mean(), axis=1)
    fake_accuracy = df['correct_fake'].mean()
    real_accuracy = df['correct_real'].mean()
    return fake_accuracy, real_accuracy

def check_correct_classifications(df, fake_cols, real_cols):
    """Determine correct classifications (True if the column contains the correct label)."""
    correct_fake = df[fake_cols].apply(lambda x: x.str.contains('Fake', case=False).any(), axis=1)
    correct_real = df[real_cols].apply(lambda x: x.str.contains('Real', case=False).any(), axis=1)
    return correct_fake, correct_real

def count_misclassifications(noexp_df, exp_df, fake_cols, real_cols):
    """Count misclassifications comparing No Explanation vs Explanation conditions."""
    correct_fake, correct_real = check_correct_classifications(noexp_df, fake_cols, real_cols)
    exp_correct_fake, exp_correct_real = check_correct_classifications(exp_df, fake_cols, real_cols)
    misclassified_as_fake = correct_fake & ~exp_correct_fake
    misclassified_as_real = correct_real & ~exp_correct_real
    return misclassified_as_fake.sum(), misclassified_as_real.sum()

def analyze_explanation_length(df, response_column):
    """Calculate the average word count for the specified explanation response column."""
    df['word_count'] = df[response_column].apply(lambda x: len(str(x).split()))
    return df['word_count'].mean()

def plot_misclassifications(labels, misclassified_fake, misclassified_real):
    """Plot a horizontal bar chart for misclassification counts."""
    bar_width = 0.4
    index = np.arange(len(labels))
    plt.barh(index, misclassified_fake, bar_width, color='red', label='Misclassified as Fake')
    plt.barh(index + bar_width, misclassified_real, bar_width, color='pink', label='Misclassified as Real')
    plt.xlabel('Number of News Items')
    plt.ylabel('Explanation Type')
    plt.title('Misclassification of Fake/Real News Based on Explanations')
    plt.yticks(index + bar_width / 2, labels)
    plt.legend()
    plt.show()

def perform_ttests(noexp_df, exp_dfs, fake_cols, real_cols):
    """Perform paired t-tests comparing 'No Explanation' to each Explanation level for fake and real accuracies."""
    fake_accuracies = {"No Explanation": noexp_df[fake_cols].apply(lambda x: (x == 'Fake').mean(), axis=1)}
    real_accuracies = {"No Explanation": noexp_df[real_cols].apply(lambda x: (x == 'Real').mean(), axis=1)}
    
    for key, exp_df in exp_dfs.items():
        fake_accuracies[key] = exp_df[fake_cols].apply(lambda x: (x == 'Fake').mean(), axis=1)
        real_accuracies[key] = exp_df[real_cols].apply(lambda x: (x == 'Real').mean(), axis=1)
    
    p_values_fake = {}
    p_values_real = {}
    for key in exp_dfs.keys():
        _, p_value_fake = ttest_rel(fake_accuracies["No Explanation"], fake_accuracies[key])
        _, p_value_real = ttest_rel(real_accuracies["No Explanation"], real_accuracies[key])
        p_values_fake[key] = p_value_fake
        p_values_real[key] = p_value_real
    return p_values_fake, p_values_real


# Global Parameters (update these placeholders)
NEWS_ARTICLES = ['presice the news list']
NEWS_LABELS = ['presice the ground truth labels']
FAKE_NEWS_INDICES = ['presice the indexes']   # list of fake news column identifiers
REAL_NEWS_INDICES = ['presice the indexes']   # list of real news column identifiers

COLUMN_MAPPING = {
    " How would you rate the fluency of the explanation?": "Fluency",
    "How would you rate the informativeness of the explanation?": "Informativeness",
    "How would you rate the persuasiveness of the explanation?": "Persuasiveness",
    "How would you rate the soundness of the explanation?": "Soundness"
}

RESULTS_ANALYSIS_DIR = 'ResultsAnalysis'


# Main Processing Function

def main():
    # Load and preprocess the main CSV file.
    input_csv = f'{RESULTS_ANALYSIS_DIR}/News_Authenticity_and_Explanation-Quality_Assessment.csv'
    df = pd.read_csv(input_csv)
    df = df.drop(columns=['What is 7*3?', 'Horodateur'])
    df = rename_columns(df, NEWS_ARTICLES, NEWS_LABELS, COLUMN_MAPPING)
    
    # Split into No Explanation and Explanation parts.
    df_NoExp, exp1, exp2, exp3 = split_dataframe_by_columns(df)
    
    # Fix news_id numbering for explanation files.
    exp1 = process_explanation_file(f'{RESULTS_ANALYSIS_DIR}/explanation1.csv', f'{RESULTS_ANALYSIS_DIR}/explanation1.csv')
    exp2 = process_explanation_file(f'{RESULTS_ANALYSIS_DIR}/explanation2.csv', f'{RESULTS_ANALYSIS_DIR}/explanation2.csv')
    exp3 = process_explanation_file(f'{RESULTS_ANALYSIS_DIR}/explanation3.csv', f'{RESULTS_ANALYSIS_DIR}/explanation3.csv')
    
    # Split explanation files into fake and real news subsets.
    fake_news_ids = FAKE_NEWS_INDICES
    real_news_ids = REAL_NEWS_INDICES
    fake_exp1, real_exp1 = split_fake_real_data(exp1, fake_news_ids, real_news_ids)
    fake_exp2, real_exp2 = split_fake_real_data(exp2, fake_news_ids, real_news_ids)
    fake_exp3, real_exp3 = split_fake_real_data(exp3, fake_news_ids, real_news_ids)
    
    fake_exp1.to_csv(f'{RESULTS_ANALYSIS_DIR}/explanation1_fake_news_data.csv', index=False)
    real_exp1.to_csv(f'{RESULTS_ANALYSIS_DIR}/explanation1_real_news_data.csv', index=False)
    fake_exp2.to_csv(f'{RESULTS_ANALYSIS_DIR}/explanation2_fake_news_data.csv', index=False)
    real_exp2.to_csv(f'{RESULTS_ANALYSIS_DIR}/explanation2_real_news_data.csv', index=False)
    fake_exp3.to_csv(f'{RESULTS_ANALYSIS_DIR}/explanation3_fake_news_data.csv', index=False)
    real_exp3.to_csv(f'{RESULTS_ANALYSIS_DIR}/explanation3_real_news_data.csv', index=False)
    
    # Quality Analysis
    criteria = ['Fluency', 'Informativeness', 'Persuasiveness', 'Soundness']
    exp1_means = calculate_means_for_criteria(pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/explanation1.csv'), criteria)
    exp2_means = calculate_means_for_criteria(pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/explanation2.csv'), criteria)
    exp3_means = calculate_means_for_criteria(pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/explanation3.csv'), criteria)
    summary = pd.DataFrame({
        'Explanation Type': ['Explanation 1', 'Explanation 2', 'Explanation 3'],
        'Fluency': [exp1_means['Fluency'], exp2_means['Fluency'], exp3_means['Fluency']],
        'Informativeness': [exp1_means['Informativeness'], exp2_means['Informativeness'], exp3_means['Informativeness']],
        'Persuasiveness': [exp1_means['Persuasiveness'], exp2_means['Persuasiveness'], exp3_means['Persuasiveness']],
        'Soundness': [exp1_means['Soundness'], exp2_means['Soundness'], exp3_means['Soundness']]
    })
    print("Quality Summary (Both):\n", summary)
    
    # Quality Analysis for Fake News.
    exp1_fake_means = calculate_means_for_criteria(pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/explanation1_fake_news_data.csv'), criteria)
    exp2_fake_means = calculate_means_for_criteria(pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/explanation2_fake_news_data.csv'), criteria)
    exp3_fake_means = calculate_means_for_criteria(pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/explanation3_fake_news_data.csv'), criteria)
    summary_fake = pd.DataFrame({
        'Explanation Type': ['Explanation 1', 'Explanation 2', 'Explanation 3'],
        'Fluency': [exp1_fake_means['Fluency'], exp2_fake_means['Fluency'], exp3_fake_means['Fluency']],
        'Informativeness': [exp1_fake_means['Informativeness'], exp2_fake_means['Informativeness'], exp3_fake_means['Informativeness']],
        'Persuasiveness': [exp1_fake_means['Persuasiveness'], exp2_fake_means['Persuasiveness'], exp3_fake_means['Persuasiveness']],
        'Soundness': [exp1_fake_means['Soundness'], exp2_fake_means['Soundness'], exp3_fake_means['Soundness']]
    })
    print("Quality Summary (Fake):\n", summary_fake)
    
    # Quality Analysis for Real News.
    exp1_real_means = calculate_means_for_criteria(pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/explanation1_real_news_data.csv'), criteria)
    exp2_real_means = calculate_means_for_criteria(pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/explanation2_real_news_data.csv'), criteria)
    exp3_real_means = calculate_means_for_criteria(pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/explanation3_real_news_data.csv'), criteria)
    summary_real = pd.DataFrame({
        'Explanation Type': ['Explanation 1', 'Explanation 2', 'Explanation 3'],
        'Fluency': [exp1_real_means['Fluency'], exp2_real_means['Fluency'], exp3_real_means['Fluency']],
        'Informativeness': [exp1_real_means['Informativeness'], exp2_real_means['Informativeness'], exp3_real_means['Informativeness']],
        'Persuasiveness': [exp1_real_means['Persuasiveness'], exp2_real_means['Persuasiveness'], exp3_real_means['Persuasiveness']],
        'Soundness': [exp1_real_means['Soundness'], exp2_real_means['Soundness'], exp3_real_means['Soundness']]
    })
    print("Quality Summary (Real):\n", summary_real)
    
    # Truthfulness Analysis (Accuracy)
    noexp_df = pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/df_NoExp.csv')
    exp1_df = pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/explanation1.csv')
    exp2_df = pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/explanation2.csv')
    exp3_df = pd.read_csv(f'{RESULTS_ANALYSIS_DIR}/explanation3.csv')
    noexp_fake_acc, noexp_real_acc = calculate_accuracy(noexp_df, FAKE_NEWS_INDICES, REAL_NEWS_INDICES)
    exp1_fake_acc, exp1_real_acc = calculate_accuracy(exp1_df, FAKE_NEWS_INDICES, REAL_NEWS_INDICES)
    exp2_fake_acc, exp2_real_acc = calculate_accuracy(exp2_df, FAKE_NEWS_INDICES, REAL_NEWS_INDICES)
    exp3_fake_acc, exp3_real_acc = calculate_accuracy(exp3_df, FAKE_NEWS_INDICES, REAL_NEWS_INDICES)
    print(f'No Explanation - Fake: {noexp_fake_acc:.2f}, Real: {noexp_real_acc:.2f}')
    print(f'Explanation 1 - Fake: {exp1_fake_acc:.2f}, Real: {exp1_real_acc:.2f}')
    print(f'Explanation 2 - Fake: {exp2_fake_acc:.2f}, Real: {exp2_real_acc:.2f}')
    print(f'Explanation 3 - Fake: {exp3_fake_acc:.2f}, Real: {exp3_real_acc:.2f}')
    summary_df = pd.DataFrame({
        'No Explanation': [noexp_fake_acc, noexp_real_acc],
        'Explanation 1': [exp1_fake_acc, exp1_real_acc],
        'Explanation 2': [exp2_fake_acc, exp2_real_acc],
        'Explanation 3': [exp3_fake_acc, exp3_real_acc]
    }, index=['Fake News Accuracy', 'Real News Accuracy'])
    print("Truthfulness Summary:\n", summary_df)
    
    # Paired t-tests comparing No Explanation vs each explanation level.
    exp_dfs = {"Explanation 1": exp1_df, "Explanation 2": exp2_df, "Explanation 3": exp3_df}
    p_values_fake, p_values_real = perform_ttests(noexp_df, exp_dfs, FAKE_NEWS_INDICES, REAL_NEWS_INDICES)
    print("Paired t-test p-values for Fake News:", p_values_fake)
    print("Paired t-test p-values for Real News:", p_values_real)
    
    # Misclassification Analysis 
    exp1_mis_fake, exp1_mis_real = count_misclassifications(noexp_df, exp1_df, FAKE_NEWS_INDICES, REAL_NEWS_INDICES)
    exp2_mis_fake, exp2_mis_real = count_misclassifications(noexp_df, exp2_df, FAKE_NEWS_INDICES, REAL_NEWS_INDICES)
    exp3_mis_fake, exp3_mis_real = count_misclassifications(noexp_df, exp3_df, FAKE_NEWS_INDICES, REAL_NEWS_INDICES)
    misclassification_summary = pd.DataFrame({
        'Explanation 1': [exp1_mis_fake, exp1_mis_real],
        'Explanation 2': [exp2_mis_fake, exp2_mis_real],
        'Explanation 3': [exp3_mis_fake, exp3_mis_real]
    }, index=['Fake News Misclassification', 'Real News Misclassification'])
    print("Misclassification Summary:\n", misclassification_summary)
    plot_misclassifications(['Explanation 1', 'Explanation 2', 'Explanation 3'],
                              [exp1_mis_fake, exp2_mis_fake, exp3_mis_fake],
                              [exp1_mis_real, exp2_mis_real, exp3_mis_real])
    
    # Effects of Explanation Length
    examples_df = pd.read_csv('Examples.csv')
    avg_words_level2 = analyze_explanation_length(examples_df, 'gpt4_response_level_2')
    avg_words_level3 = analyze_explanation_length(examples_df, 'gpt4_response_level_3')
    print(f'Average words per text (Level 2): {avg_words_level2}')
    print(f'Average words per text (Level 3): {avg_words_level3}')

if __name__ == "__main__":
    main()
