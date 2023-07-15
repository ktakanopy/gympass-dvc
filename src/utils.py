from collections import Counter
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy
import seaborn as sns
from pandas import DataFrame
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

def get_features(df, target_features, index_features):
    return list(set(list(df)).difference(target_features + index_features))

def plot_histogram(df, column, bins=50, figsize=(10,6), title=None, color='blue', kde=False, hue=None):
    plt.figure(figsize=figsize)
    sns.histplot(data=df, x=column, bins=bins, color=color, kde=kde, hue=hue, stat='density')
    plt.title(title if title else f'Histogram of {column}')
    plt.show()

def plot_avg(df, row, column, hue=None, figsize=(10,6), title='Average by Categories', xlabel=None, ylabel='Average', palette='viridis', sort=False):
    plt.figure(figsize=figsize)
    order = df.groupby(row)[column].mean().sort_values().index if sort else None
    ax = sns.barplot(data=df, x=row, y=column, palette=palette, order=order, hue=hue, errorbar=None, estimator=np.mean)
    plt.title(f"Average {column} by {row}")
    plt.xlabel(xlabel if xlabel else row)
    plt.ylabel(ylabel)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
    plt.xticks(rotation=90)
    plt.show()

def plot_count(df, column, figsize=(10,6), title='Distribution of Target Classes', xlabel=None, ylabel='Count', palette='viridis', hue=None, sort=False, log_scale=False):
    plt.figure(figsize=figsize)
    order = df[column].value_counts().index if sort else None
    ax = sns.countplot(data=df, x=column, palette=palette, order=order, hue=hue)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel)
    if log_scale:
        ax.set_yscale('log')
    plt.xticks(rotation=90)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center',
                    xytext = (0, 9),
                    textcoords = 'offset points')
    plt.show()

def get_numerical_features(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_features(df):
    return df.select_dtypes(include=['object']).columns.tolist()


def filter_list(base_list, filter_list):
    return [item for item in base_list if item not in filter_list]

def find_nulls(df):
    null_values = df.isnull().sum()
    return null_values[null_values > 0]

def show_null_values_df(df):
    null_cols = df.columns[df.isnull().any()]
    null_rows = df[df.isnull().any(axis=1)] 
    return null_rows[null_cols]

def swap_inf_to_none(df):
    df.replace([np.inf, -np.inf], None, inplace=True)
    return df

def evaluate_metrics(y_test, y_pred, thres=None, labels=None, normalize='all'):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    print(f'Precision-Recall AUC: {pr_auc}')

    f1_scores = 2*precision*recall / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores, nan=-np.inf)
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index] if thres is None else thres
    best_score = f1_scores[best_index]

    print("classification_report")
    print(classification_report(y_test, y_pred > best_threshold))
    print()

    print(f'Best F1-score: {best_score} at threshold {best_threshold}')
    
    plt.plot(recall, precision)
    plt.scatter(recall[best_index], precision[best_index], color='red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.show()
    
    plot_confusion_matrix(y_test, y_pred > best_threshold, labels, normalize)
    return best_threshold


def plot_confusion_matrix(y_true, y_pred_class, labels, normalize=False):
    cm = confusion_matrix(y_true, y_pred_class, normalize=normalize)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
