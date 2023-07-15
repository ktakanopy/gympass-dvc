from pandas import DataFrame
from typing import List
import pandas as pd
from typing import Optional
from collections import Counter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import f1_score, classification_report, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score
def get_features(df, target_features, index_features):
    return list(set(list(df)).difference(target_features + index_features))

def plot_histogram(df: pd.DataFrame, column: str, bins: Optional[int] = 50, figsize: Tuple[int, int] = (10,6), 
                   title: Optional[str] = None, color: Optional[str] = 'blue', kde: Optional[bool] = False, hue: str = None) -> None:
    """
    Plot a histogram of a specified column in a DataFrame using seaborn's displot function.

    Args:
    df (pd.DataFrame): The DataFrame containing the column to plot.
    column (str): The name of the column to plot.
    bins (Optional[int], default=50): Number of histogram bins.
    figsize (Tuple[int, int], default=(10,6)): The size of the figure to display.
    title (Optional[str], default=None): The title of the plot.
    color (Optional[str], default='blue'): The color of the histogram.
    kde (Optional[bool], default=False): Whether or not to plot a gaussian kernel density estimate.

    Returns:
    None: This function doesn't return anything; it only produces a plot.
    """
    plt.figure(figsize=figsize)
    sns.histplot(data=df, x=column, bins=bins, color=color, kde=kde, hue=hue, stat='density')
    plt.title(title if title else f'Histogram of {column}')
    plt.show()

def plot_avg(df: pd.DataFrame, row: str, column: str, hue=None, figsize: Tuple[int, int] = (10,6), 
             title: Optional[str] = 'Average by Categories', xlabel: Optional[str] = None, 
             ylabel: Optional[str] = 'Average', palette: Optional[str] = 'viridis',
             sort: Optional[bool] = False) -> None:

    plt.figure(figsize=figsize)
    if sort:
        order = df.groupby(row)[column].mean().sort_values().index
    else:
        order = None
    ax = sns.barplot(data=df, x=row, y=column, palette=palette, order=order, hue=hue, errorbar=None, estimator=np.mean)
    plt.title(f"Average {column} by {row}")

    # If xlabel is not provided, use the column name
    if xlabel is None:
        xlabel = row
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Annotating the count on top of each bar
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    # Rotate x-axis labels
    plt.xticks(rotation=90)

    plt.show()

def plot_count(df: pd.DataFrame, column: str, figsize: Tuple[int, int] = (10,6), title: Optional[str] = 'Distribution of Target Classes', 
               xlabel: Optional[str] = None, ylabel: Optional[str] = 'Count', palette: Optional[str] = 'viridis', hue = None,
               sort: Optional[bool] = False, log_scale: Optional[bool] = False) -> None:
    """
    Plot a countplot of a specified column in a DataFrame. 

    Args:
    df (pd.DataFrame): The DataFrame containing the column to plot.
    column (str): The name of the column to plot.
    figsize (Tuple[int, int], default=(10,6)): The size of the figure to display.
    title (Optional[str], default='Distribution of Target Classes'): The title of the plot. 
    xlabel (Optional[str], default=None): The label of the x-axis. If not provided, the column name is used.
    ylabel (Optional[str], default='Count'): The label of the y-axis.
    palette (Optional[str], default='viridis'): The color palette to use.
    sort (Optional[bool], default=False): Whether to sort the categories by count.
    log_scale (Optional[bool], default=False): Whether to apply a logarithmic scale to the y-axis.

    Returns:
    None: This function doesn't return anything; it only produces a plot.
    """
    plt.figure(figsize=figsize)
    if sort:
        order = df[column].value_counts().index
    else:
        order = None
    ax = sns.countplot(data=df, x=column, palette=palette, order=order, hue=hue)
    plt.title(title)

    # If xlabel is not provided, use the column name
    if xlabel is None:
        xlabel = column
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Apply logarithmic scale if log_scale is True
    if log_scale:
        ax.set_yscale('log')

    # Rotate x-axis labels
    plt.xticks(rotation=90)

    # Annotating the count on top of each bar
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center',
                    xytext = (0, 9),
                    textcoords = 'offset points')

    plt.show()


def get_numerical_features(df: pd.DataFrame) -> List[str]:
    """
    Get the list of numerical features in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame.

    Returns:
    List[str]: The list of numerical features.
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_features(df: pd.DataFrame) -> List[str]:
    """
    Get the list of categorical features in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame.

    Returns:
    List[str]: The list of categorical features.
    """
    return df.select_dtypes(include=['object']).columns.tolist()

def add_labels(df: DataFrame) -> DataFrame:
    """
    Assigns a label to each row in a DataFrame based on the conditions of 'is_churn', 'is_keep', and 'is_upgrade'.

    Parameters:
    df (DataFrame): The DataFrame that needs labels. 
    It should have 'is_churn', 'is_keep', and 'is_upgrade' columns which are expected to be boolean.

    Returns:
    DataFrame: The original DataFrame with an additional 'multiclass_target' column which contains the labels.
    """

    df['multiclass_target'] = df['is_churn']*1 + df['is_keep']*0 + df['is_upgrade']*2
    df.rename(columns={'is_churn': 'churn_target'}, inplace=True)
    df = df.drop(['is_keep', 'is_upgrade'], axis=1)
    return df

def filter_list(base_list, filter_list):
    return [item for item in base_list if item not in filter_list]

def find_nulls(df):
  """Finds all the null values in a Pandas DataFrame."""
  null_values = df.isnull().sum()
  return null_values[null_values > 0]

def show_null_values_df(df):
    null_cols = df.columns[df.isnull().any()]
    null_rows = df[df.isnull().any(axis=1)] 
    return null_rows[null_cols]

def swap_inf_to_none(df):
  """Swaps np.inf to None in a Pandas DataFrame."""
  df.replace([np.inf, -np.inf], None, inplace=True)
  return df

def preprocessing(df, use_user_group_text=False, test=False):
    # Features
    df['years_usage'] = df['user_billings'] / 12
    df['user_engagement'] = df['user_lifetime_visits'] / df['user_billings']
    df['gym_visit_frequency'] = df['gym_last_60_days_visits'] / 60
    df.loc[df['user_last_60_days_visits'] > 60, 'user_last_60_days_visits'] = None # none to use in median inputer
    df['user_visit_frequency'] = df['user_last_60_days_visits'] / 60

    # User Age Group
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = [1, 2, 3, 4, 5, 6] if not use_user_group_text else ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
    df['user_age_group'] = pd.cut(df['user_age'], bins=bins, labels=labels, right=False)

    if not use_user_group_text:
        df['user_age_group'] = df['user_age_group'].astype(int)

    df.rename(columns={'user_billings': 'months_usage'}, inplace=True)

    # Filter noise
    if not test:
        df = add_labels(df)  # Assuming 'add_labels' is a function defined elsewhere

    return df



def evaluate_metrics(y_test, y_pred, thres=None, labels=None, normalize='all'):

    # Compute and print the ROC AUC score
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    # Compute and print the AUC for the precision-recall curve
    pr_auc = auc(recall, precision)
    print(f'Precision-Recall AUC: {pr_auc}')

    # Compute F1 scores for different thresholds
    f1_scores = 2*precision*recall / (precision + recall)
    
    f1_scores = np.nan_to_num(f1_scores, nan=-np.inf)
    
    # Get the best threshold
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index] if thres is None else thres
    best_score = f1_scores[best_index]

    print("classification_report")
    print(classification_report(y_test, y_pred > best_threshold))
    print()

    print(f'Best F1-score: {best_score} at threshold {best_threshold}')
    
    # Plot the precision-recall curve
    plt.plot(recall, precision)
    plt.scatter(recall[best_index], precision[best_index], color='red')  # mark best point
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.show()
    
    # Confusion matrix at the best threshold
    plot_confusion_matrix(y_test, y_pred > best_threshold, labels, normalize)
    return best_threshold


def plot_confusion_matrix(y_true, y_pred_class, labels, normalize=False):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred_class, normalize=normalize)
    
    # Create a confusion matrix display
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Plot the confusion matrix
    cm_display.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def feat_engineering(df, small_categories_train, numeric_features, user_features):
    
    #TODO: criar feature de valor do usuário do plano
    # Features
    df['user_visit_share_months_usage_ratio'] = df['user_lifetime_visit_share'] / df['months_usage']
    df['last_60_user_visit_months_usage_ratio'] = df['user_last_60_days_visits'] / df['months_usage']
    df['last_60_user_visit_share_months_usage_ratio'] = df['user_last_60_days_visits'] / df['months_usage']
    df['last_60_user_visit_frequency'] = df['user_last_60_days_visits'] / 60

    # transformation features
    numeric_features = filter_list(numeric_features, ['user_age_group'])
    for num_feat in numeric_features:
        df[num_feat + '_log'] = np.log(df[num_feat]+1)

    df['user_visit_share_months_log_usage_ratio'] = df['user_lifetime_visit_share'] / df['months_usage_log']
    df['last_60_user_visit_months_log_usage_ratio'] = df['user_last_60_days_visits'] / df['months_usage_log']
    df['last_60_user_visit_share_months_log_usage_ratio'] = df['user_last_60_days_visits'] / df['months_usage_log']

    # Calculate aggregated features
    gym_features = df.groupby('gym')[user_features].agg(['sum', 'mean', 'max', np.median, np.average, np.std, np.var, np.ptp, scipy.stats.skew, scipy.stats.kurtosis])
    # Flatten the columns
    gym_features.columns = ['gym_' + '_'.join(col).strip() for col in gym_features.columns.values]

    df['gym_user_count'] = df.groupby('gym')['user'].transform('nunique')

    # Then merge
    df = df.merge(gym_features, on='gym', how='left')
    
    plan_price_dict = {
        'Silver': 99.9,
        'Basic I': 39.9,
        'Basic II': 59.9,
        'Silver +': 140.9,
        'Free': 0}
    
    df['user_plan_price'] = df.user_plan.map(plan_price_dict)
    
    agg_group = df.groupby(["gym"]).user_plan_price.sum().reset_index()
    # Rename the column to reflect total revenue
    agg_group = agg_group.rename(columns={'user_plan_price': 'total_revenue'})

    # Reset the index to flatten the dataframe
    df = df.merge(agg_group, on='gym', how='left')
    # Map these categories to 'Others'
    df['gym_category'].replace(small_categories_train, 'Others').value_counts(normalize=True)
    df = swap_inf_to_none(df)
    # add more gym features
    return df


def optmize_model(X, y, preprocessor, model, params_func, use_predict_proba=True, n_trials=20, balance_feat=None, n_splits=5):

    def objective(trial):
        avg_roc = 0
        avg_f1 = 0
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for tr, ts in kf.split(X, y):
            Xtr, Xvl = X.iloc[tr], X.iloc[ts]
            ytr, yvl = y.iloc[tr], y.iloc[ts]
            
            params = params_func(trial)
            if balance_feat:
                scale_pos_weight = ytr.value_counts()[0] / ytr.value_counts()[1]
                params[balance_feat] = scale_pos_weight

            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model(**params))])
            pipe.fit(Xtr, ytr)
            if use_predict_proba:
                p = pipe.predict_proba(Xvl)[:, 1]
            else:
                p = pipe.predict(Xvl)
            avg_roc += roc_auc_score(yvl, p)
            avg_f1 += f1_score(yvl, p > 0.5, average='macro')


        print('avg roc:', avg_roc / n_splits)
        print('avg f1:', avg_f1 / n_splits)
        return avg_roc / n_splits

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    return study