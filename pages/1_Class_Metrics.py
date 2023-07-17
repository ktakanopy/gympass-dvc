import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def read_metrics(model):
    # Reads the metrics and confusion matrix dataframes for a given model
    class_metrics = pd.read_csv(f'reports/test/{model}_class_metrics.csv')
    confusion_matrix = pd.read_csv(f'reports/test/{model}_confusion_matrix.csv')
    return class_metrics, confusion_matrix

def main():
    st.title("Model Metrics Comparison")

    # Model names
    model_names = [model.split('_')[0] for model in os.listdir('reports/test/') if 'class_metrics' in model]

    # Display each model's metrics side by side
    cols = st.columns(len(model_names))
    metrics_dfs = []
    for i, model in enumerate(model_names):
        class_metrics, _ = read_metrics(model)
        class_metrics.set_index("Unnamed: 0", inplace=True)
        metrics_dfs.append((model, class_metrics))
    # Specify the rows that you want to include
    rows = ['non-churn', 'churn', 'macro avg']
    
    all_metrics_rows = []

    for model_name, model_df in metrics_dfs:
        for row in rows:
            metrics = model_df.loc[row]
            # Add the model name as a new 'Model' column
            all_metrics_rows.append([model_name, row, metrics['precision'], metrics['recall'], metrics['f1-score']])

    all_metrics = pd.DataFrame(all_metrics_rows, columns=["Model Name", "Type", "Precision", "Recall", "F1-score"])

    st.header("Model Comparison")

    type_metrics = ['churn', 'non-churn', 'macro avg']
    
    for type_metric in type_metrics:
        st.subheader(f"Plot for {type_metric}")
        metric_df = all_metrics[all_metrics.Type == type_metric]
        df_melt = metric_df.melt(id_vars=['Model Name', 'Type'], value_vars=['Precision', 'Recall', 'F1-score'], var_name='Metric', value_name='Value')
 
        plt.figure(figsize=(10,6))
        ax =sns.barplot(x='Model Name', y='Value', hue='Metric', data=df_melt)
        plt.title(f'Model Comparison for {type_metric}')
            
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2, p.get_height(), '{0:.2f}'.format(p.get_height()), 
                    color='black', ha='center', va='bottom')

        st.pyplot(plt)

    print(all_metrics)

if __name__ == "__main__":
    main()
