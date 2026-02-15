import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Set the style for all plots
sns.set_theme(style="whitegrid") 

# Define consistent colors for metrics
METRIC_COLORS = {
    'f1_score': '#4c72b0', # '#2ecc71',     # green for F1-score
    'accuracy': '#dd8452', # '#3498db',      # blue for accuracy
    'confidence_median': '#55a868', #'#e74c3c'  # red for confidence
}

INPUT_ROOT_DIR = os.path.join('..','transcripts_summary_reports')
OUTPUT_ROOT_DIR = os.path.join('..','transcripts_summary_reports')

INPUT_FILEPATH = os.path.join(INPUT_ROOT_DIR, 'transcripts_stat_summary_final_20250128_013836.csv')

def load_and_preprocess_data(csv_path):
    """
    Load and preprocess the CSV data for visualization.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame without any specific sorting
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create truncated model names by removing specified suffixes
    df['model_name_truncated'] = df['model_name'].apply(lambda x: 
        x.replace('_instruct_q4_k_m', '').replace('_q4_k_m', '')
    )
    
    return df

def plot_f1_confidence_comparison(df):
    """
    Create a grouped bar plot comparing F1-score and normalized confidence median.
    Data is ordered by F1-score in descending order.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame
    """
    # Sort by F1-score in descending order and reset index
    df_sorted = df.sort_values('f1_score', ascending=False).reset_index(drop=True)
    
    # Keep the original model order based on F1-score
    model_order = df_sorted['model_name_truncated'].tolist()
    
    # Create a copy of the sorted dataframe
    plot_df = df_sorted.copy()
    
    # Normalize confidence from 0-100 to 0.0-1.0 scale
    plot_df['confidence_median'] = plot_df['confidence_median'] / 100.0
    
    # Prepare data for plotting with normalized confidence
    plot_data = plot_df.melt(
        id_vars=['model_name_truncated'],
        value_vars=['f1_score', 'confidence_median'],
        var_name='Metric',
        value_name='Score'
    )
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot_data,
        x='model_name_truncated',
        y='Score',
        hue='Metric',
        palette=[METRIC_COLORS['f1_score'], METRIC_COLORS['confidence_median']],
        order=model_order
    )
    
    # Customize the plot
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model (ordered by F1-score)')
    plt.ylabel('Score (0-1 scale)')
    plt.title('F1-Score vs Normalized Confidence Median by Model (AgenticSimLLM)')
    plt.legend(title='')
    plt.tight_layout()
    
    # Save the plot
    output_plot_filepath = os.path.join(OUTPUT_ROOT_DIR, 'f1_confidence_comparison.png')
    plt.savefig(output_plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_f1_comparison(df):
    """
    Create a grouped bar plot comparing accuracy and F1-score with a random baseline.
    Data is ordered by accuracy in descending order.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame
    """
    # Sort by accuracy in descending order and reset index
    df_sorted = df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    
    # Keep the original model order based on accuracy
    model_order = df_sorted['model_name_truncated'].tolist()
    
    # Prepare data for plotting
    plot_data = df_sorted.melt(
        id_vars=['model_name_truncated'],
        value_vars=['accuracy', 'f1_score'],
        var_name='Metric',
        value_name='Score'
    )
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Create the main bar plot with explicit ordering
    sns.barplot(
        data=plot_data,
        x='model_name_truncated',
        y='Score',
        hue='Metric',
        palette=[METRIC_COLORS['accuracy'], METRIC_COLORS['f1_score']],
        order=model_order
    )
    
    # Add random baseline line
    plt.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5)
    
    # Add baseline label
    plt.text(plt.xlim()[1] * 0.7, 0.51, 'random baseline',
             fontsize=10, color='gray', alpha=0.7)
    
    # Customize the plot
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model (ordered by accuracy)')
    plt.ylabel('Score')
    plt.title('Accuracy vs F1-Score by Model (AgenticSimLLM)')
    plt.legend(title='')
    plt.tight_layout()
    
    # Save the plot
    output_plot_filepath = os.path.join(OUTPUT_ROOT_DIR, 'accuracy_f1_comparison.png')
    plt.savefig(output_plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_f1_accuracy_comparison(df):
    """
    Create a grouped bar plot comparing F1-score and accuracy with baselines.
    Data is ordered by F1-score in descending order.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame
    """
    # Sort by F1-score in descending order and reset index
    df_sorted = df.sort_values('f1_score', ascending=False).reset_index(drop=True)
    
    # Keep the original model order based on F1-score
    model_order = df_sorted['model_name_truncated'].tolist()
    
    # Prepare data for plotting
    plot_data = df_sorted.melt(
        id_vars=['model_name_truncated'],
        value_vars=['f1_score', 'accuracy'],
        var_name='Metric',
        value_name='Score'
    )
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Create the main bar plot with explicit ordering
    sns.barplot(
        data=plot_data,
        x='model_name_truncated',
        y='Score',
        hue='Metric',
        palette=[METRIC_COLORS['f1_score'], METRIC_COLORS['accuracy']],
        order=model_order
    )
    
    # Add baseline lines
    plt.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5)
    plt.axhline(y=0.36, linestyle='--', color='gray', alpha=0.5)
    
    # Add baseline labels
    plt.text(plt.xlim()[1] * 0.7, 0.51, '50/50 baseline',
             fontsize=10, color='gray', alpha=0.7)
    plt.text(plt.xlim()[1] * 0.7, 0.37, '28/72 baseline',
             fontsize=10, color='gray', alpha=0.7)
    
    # Customize the plot
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model (ordered by F1-score)')
    plt.ylabel('Score')
    plt.title('F1-Score vs Accuracy by Model (AgenticSimLLM)')
    plt.legend(title='')
    plt.tight_layout()
    
    # Save the plot
    output_plot_filepath = os.path.join(OUTPUT_ROOT_DIR, 'f1_accuracy_comparison.png')
    plt.savefig(output_plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_f1_vs_duration(df):
    """
    Create a scatter plot of F1-score vs total duration with directly labeled points
    using distinct shapes and dark colors.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame containing model performance data
    """
    # Create figure with proportional size
    plt.figure(figsize=(12, 8))
    
    # Sort data by F1-score for consistent ordering
    df_sorted = df.sort_values('f1_score', ascending=False)
    
    # Get marker/color combinations
    markers, colors = get_marker_color_combinations()
    
    # Create scatter plot with unique marker/color combinations
    for idx, row in df_sorted.iterrows():
        marker_idx = idx % len(markers)
        
        plt.scatter(
            row['speaker_all_total_duration_sec_median'],
            row['f1_score'],
            marker=markers[marker_idx],
            color=colors[marker_idx // len(markers)],
            s=150,
            alpha=1.0,
            edgecolors='black',
            linewidth=1.5
        )
        
        plt.text(
            row['speaker_all_total_duration_sec_median'],
            row['f1_score'],
            row['model_name_truncated'],
            fontsize=10,
            ha='right',
            va='bottom'
        )
    
    plt.xlabel('Total Duration (seconds)', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Score vs Total Duration (AgenticSimLLM)', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.3, color='grey')
    
    output_plot_filepath = os.path.join(OUTPUT_ROOT_DIR, 'f1_vs_duration.png')
    plt.savefig(output_plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_f1_vs_tokens(df):
    """
    Create a scatter plot of F1-score vs total tokens with directly labeled points
    using distinct shapes and dark colors.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame containing model performance data
    """
    plt.figure(figsize=(12, 8))
    
    # Sort data by F1-score for consistent ordering
    df_sorted = df.sort_values('f1_score', ascending=False)
    
    # Get marker/color combinations
    markers, colors = get_marker_color_combinations()
    
    for idx, row in df_sorted.iterrows():
        marker_idx = idx % len(markers)
        
        plt.scatter(
            row['speaker_all_token_median'],
            row['f1_score'],
            marker=markers[marker_idx],
            color=colors[marker_idx // len(markers)],
            s=150,
            alpha=1.0,
            edgecolors='black',
            linewidth=1.5
        )
        
        plt.text(
            row['speaker_all_token_median'],
            row['f1_score'],
            row['model_name_truncated'],
            fontsize=10,
            ha='right',
            va='bottom',
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.8,
                pad=1
            )
        )
    
    plt.xlabel('Total Tokens', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Score vs Total Tokens (AgenticSimLLM)', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.3, color='grey')
    
    output_plot_filepath = os.path.join(OUTPUT_ROOT_DIR, 'f1_vs_tokens.png')
    plt.savefig(output_plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()

def get_marker_color_combinations():
    """
    Create 16 unique combinations using 8 distinct shapes and 2 colors from seaborn's
    dark palette, with black edges for better visibility.
    
    Returns:
        tuple: (markers_list, colors_list) containing the visual elements for plot points
    """
    markers = ['o', 's', '^', 'D', 'p', 'h', 'v', '8']
    colors = sns.color_palette("muted", n_colors=4)[:2]
    
    markers_list = []
    colors_list = []
    
    for marker in markers:
        for color in colors:
            markers_list.append(marker)
            colors_list.append(color)
    
    return markers_list, colors_list

def main():
    """
    Main function to execute all visualizations.
    """
    # Load and preprocess data
    df = load_and_preprocess_data(INPUT_FILEPATH)
    
    # Generate all plots
    plot_f1_accuracy_comparison(df)
    plot_accuracy_f1_comparison(df)
    plot_f1_confidence_comparison(df)
    plot_f1_vs_duration(df)
    plot_f1_vs_tokens(df)

if __name__ == "__main__":
    main()