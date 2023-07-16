import hydra
import pandas as pd
from omegaconf import DictConfig
import numpy as np
from src.utils import filter_list, get_numerical_features, swap_inf_to_none, save_feature_list
from sklearn.model_selection import train_test_split
import scipy

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

def feat_engineering(df, small_categories_train, numeric_features, user_features, feature_list):
    new_features = [
        'user_visit_share_months_usage_ratio',
        'last_60_user_visit_months_usage_ratio',
        'last_60_user_visit_share_months_usage_ratio',
        'last_60_user_visit_frequency',
        'user_visit_share_months_log_usage_ratio',
        'last_60_user_visit_months_log_usage_ratio',
        'last_60_user_visit_share_months_log_usage_ratio',
        'gym_user_count'
    ]

    df['user_visit_share_months_usage_ratio'] = df['user_lifetime_visit_share'] / df['months_usage']
    df['last_60_user_visit_months_usage_ratio'] = df['user_last_60_days_visits'] / df['months_usage']
    df['last_60_user_visit_share_months_usage_ratio'] = df['user_last_60_days_visits'] / df['months_usage']
    df['last_60_user_visit_frequency'] = df['user_last_60_days_visits'] / 60

    numeric_features = filter_list(numeric_features, ['user_age_group'])
    for num_feat in numeric_features:
        new_feature = num_feat + '_log'
        df[new_feature] = np.log(df[num_feat]+1)
        new_features.append(new_feature)

    df['user_visit_share_months_log_usage_ratio'] = df['user_lifetime_visit_share'] / df['months_usage_log']
    df['last_60_user_visit_months_log_usage_ratio'] = df['user_last_60_days_visits'] / df['months_usage_log']
    df['last_60_user_visit_share_months_log_usage_ratio'] = df['user_last_60_days_visits'] / df['months_usage_log']

    gym_features = df.groupby('gym')[user_features].agg(['sum', 'mean', 'max', np.median, np.average, np.std, np.var, np.ptp, scipy.stats.skew, scipy.stats.kurtosis])
    gym_features.columns = ['gym_' + '_'.join(col).strip() for col in gym_features.columns.values]
    new_features.extend(gym_features.columns.values.tolist())
    
    df['gym_user_count'] = df.groupby('gym')['user'].transform('nunique')
    df = df.merge(gym_features, on='gym', how='left')

    df['gym_category'].replace(small_categories_train, 'Others').value_counts(normalize=True)
    df = swap_inf_to_none(df)

    feature_list.extend(new_features)
    return df, feature_list


def get_small_categories(df):
    counts = df['gym_category'].value_counts(normalize=True)
    return counts[counts < 0.01].index

def get_user_features(df, features):
    numeric_features = sorted(get_numerical_features(df[features]))
    user_features = [k for k in numeric_features if "user" in k]
    return filter_list(user_features, ['user_age_group'])

@hydra.main(config_path="../../config", config_name="main", version_base=None)
def make_dataset(config: DictConfig):
    """Function to process the data"""

    print(f"Process data submission {config.paths.raw.submission}")
    print(f"Process data train {config.paths.raw.train}")

    train_data = pd.read_excel(config.paths.raw.train)
    submission_data = pd.read_excel(config.paths.raw.submission)
    
    train_data = preprocessing(train_data) 
    submission_data = preprocessing(submission_data, test=True) 

    gym_indexes = train_data.gym.unique()
    train_index, test_index = train_test_split(gym_indexes, test_size=config.make_dataset.split_size, random_state=config.make_dataset.random_state)

    train = train_data[train_data.gym.isin(train_index)]
    test = train_data[train_data.gym.isin(test_index)]

    small_categories = get_small_categories(train_data)

    train.to_parquet(config.paths.processed_baseline.train)
    test.to_parquet(config.paths.processed_baseline.test)

    user_features = get_user_features(train, config.make_dataset.features)
    numeric_features = sorted(get_numerical_features(train[config.make_dataset.features]))
    
    feature_list = config.make_dataset.features

    train_data, feature_list = feat_engineering(train_data, small_categories, numeric_features, user_features, feature_list) 
    submission_data, _ = feat_engineering(submission_data, small_categories, numeric_features, user_features, feature_list.copy())
    
    train.to_parquet(config.paths.processed.train)
    test.to_parquet(config.paths.processed.test)
    submission_data.to_parquet(config.paths.processed.submission)
    
    save_feature_list(feature_list, config)


if __name__ == "__main__":
    make_dataset()