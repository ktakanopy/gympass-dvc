import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def read_metrics(model):
    # Reads the metrics and confusion matrix dataframes for a given model
    class_metrics = pd.read_csv(f'reports/test/{model}_class_metrics.csv')
    confusion_matrix = pd.read_csv(f'reports/test/{model}_confusion_matrix.csv', index_col=0)
    return class_metrics, confusion_matrix

def plot_confusion_matrix(confusion_matrix, model_name):
    # Normalizes the confusion matrix
    confusion_matrix_norm = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_norm, annot=True, cmap='Blues', fmt=".2f")
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    st.pyplot(plt)

def main():
    st.title("Confusion Matrix Comparison and Model Interpretation with Shap Values")

    st.header("Confusion Matrix for each model")
    # Model names
    model_names = [model.split('_')[0] for model in os.listdir('reports/test/') if 'class_metrics' in model]

    # Display each model's metrics and confusion matrix side by side
    metrics_dfs = []
    for i, model in enumerate(model_names):
        st.subheader(model)
        class_metrics, confusion_matrix = read_metrics(model)
        class_metrics.set_index("Unnamed: 0", inplace=True)
        metrics_dfs.append((model, class_metrics))
            # Plot the confusion matrix for the current model
        print(confusion_matrix)
        plot_confusion_matrix(confusion_matrix, model)
        
    # Define the path to the image file
    image_path = 'reports/summary_shap_values.png'

    st.header("SHAP Values")
    # Check if the image file exists
    if os.path.exists(image_path):
        # Load and display the image
        st.image(image_path, caption='Summary SHAP Values', use_column_width=True)
    else:
        # Display a message if the file doesn't exist
        st.write("You need to run DVC pipeline!")


if __name__ == "__main__":
    main()
