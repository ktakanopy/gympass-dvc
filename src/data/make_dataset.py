import hydra
import pandas as pd
from omegaconf import DictConfig
import numpy as np
from src.utils import filter_list
from sklearn.model_selection import train_test_split


def add_labels(df):
    df.rename(columns={'is_churn': 'churn_target'}, inplace=True)
    df = df.drop(['is_keep', 'is_upgrade'], axis=1)
    return df

def preprocessing(df, use_user_group_text=False, test=False):
    df['years_usage'] = df['user_billings'] / 12
    df['user_engagement'] = df['user_lifetime_visits'] / df['user_billings']
    df['gym_visit_frequency'] = df['gym_last_60_days_visits'] / 60
    df.loc[df['user_last_60_days_visits'] > 60, 'user_last_60_days_visits'] = None
    df['user_visit_frequency'] = df['user_last_60_days_visits'] / 60

    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = [1, 2, 3, 4, 5, 6] if not use_user_group_text else ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
    df['user_age_group'] = pd.cut(df['user_age'], bins=bins, labels=labels, right=False)

    if not use_user_group_text:
        df['user_age_group'] = df['user_age_group'].astype(int)

    df.rename(columns={'user_billings': 'months_usage'}, inplace=True)

    if not test:
        df = add_labels(df)  

    return df

def feat_engineering(df, small_categories_train, numeric_features, user_features):
    df['user_visit_share_months_usage_ratio'] = df['user_lifetime_visit_share'] / df['months_usage']
    df['last_60_user_visit_months_usage_ratio'] = df['user_last_60_days_visits'] / df['months_usage']
    df['last_60_user_visit_share_months_usage_ratio'] = df['user_last_60_days_visits'] / df['months_usage']
    df['last_60_user_visit_frequency'] = df['user_last_60_days_visits'] / 60

    numeric_features = filter_list(numeric_features, ['user_age_group'])
    for num_feat in numeric_features:
        df[num_feat + '_log'] = np.log(df[num_feat]+1)

    df['user_visit_share_months_log_usage_ratio'] = df['user_lifetime_visit_share'] / df['months_usage_log']
    df['last_60_user_visit_months_log_usage_ratio'] = df['user_last_60_days_visits'] / df['months_usage_log']
    df['last_60_user_visit_share_months_log_usage_ratio'] = df['user_last_60_days_visits'] / df['months_usage_log']

    gym_features = df.groupby('gym')[user_features].agg(['sum', 'mean', 'max', np.median, np.average, np.std, np.var, np.ptp, scipy.stats.skew, scipy.stats.kurtosis])
    gym_features.columns = ['gym_' + '_'.join(col).strip() for col in gym_features.columns.values]
    df['gym_user_count'] = df.groupby('gym')['user'].transform('nunique')
    df = df.merge(gym_features, on='gym', how='left')

    return df

@hydra.main(config_path="config", config_name="main", version_base=None)
def make_dataset(config: DictConfig):
    """Function to process the data"""

    print(f"Process data submission {config.paths.raw.submission}")
    print(f"Process data train {config.paths.raw.train}")

    TRAIN_RAW_PATH = config.paths.raw.train
    SUBMISSION_RAW_PATH = config.paths.raw.submission

    SPLIT_SIZE = config.make_dataset.split_size
    RANDOM_STATE = config.make_dataset.random_state

    TARGET = config.make_dataset.target
    FEATURES = config.make_dataset.features
    
    train_data = pd.read_excel(TRAIN_RAW_PATH)
    submission_data = pd.read_excel(SUBMISSION_RAW_PATH)
    
    train_data = preprocessing(train_data) 
    submission_data = preprocessing(submission_data) 
    
    gym_indexes = train_data.gym.unique()
    train_index, test_index = train_test_split(gym_indexes, test_size=SPLIT_SIZE, random_state=RANDOM_STATE)

    train = train_data[train_data.gym.isin(train_index)]
    test = train_data[train_data.gym.isin(test_index)]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # Print the shapes of the training and testing sets
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape) 


if __name__ == "__main__":
    make_dataset()