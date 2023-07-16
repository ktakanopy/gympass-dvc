import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from src.utils import load_feature_list, load_xgb_params, save_predictions
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def get_model_pipeline(params, cols_to_input, categorical_features, estimate):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), cols_to_input),
            ('cat', TargetEncoder(), categorical_features)],remainder='passthrough')

    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Preprocessing (imputation + one-hot encoding)
        ('model', XGBClassifier(**params, scale_pos_weight=estimate, objective='binary:logistic', use_label_encoder=False, eval_metric='logloss'))  # Model
    ])
    return pipeline

def get_weight_df(y):
    # count examples in each class
    counter = Counter(y)
    # estimate scale_pos_weight value
    estimate = counter[0] / counter[1]
    return estimate

def model_fit(X, y, pipeline, test_size=0.2, random_state=42, early_stopping_rounds=20):
    # split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # instantiate the model
    # specify validation set
    eval_set = [(X_val, y_val)]

    # fit the model
    pipeline.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)
    return pipeline

def generate_heuristic_baseline(train, test, user_engagement_quantile_thres=0.20):
    threshold_user_engagement = train.user_engagement.quantile(user_engagement_quantile_thres)
    heuristic_predict = (test.user_engagement < threshold_user_engagement).astype(bool)
    return heuristic_predict

def generate_decision_tree_predictions(train, test, config):
    numeric_features = config.train_model.baselines.decision_tree.numeric_features
    categorical_features = config.train_model.baselines.decision_tree.categorical_features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)])

    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Preprocessing (imputation + one-hot encoding)
        ('model', DecisionTreeClassifier(class_weight='balanced'))  # Model
    ])
    
    features = numeric_features + categorical_features
    
    X_train = train[features]
    y_train =  train[config.train_model.target]

    X_test = test[features]

    pipeline.fit(X_train, y_train) 
    
    y_pred_proba = pipeline.predict_proba(X_test)[:,1]
    return y_pred_proba

def generate_xgb_classifier(train, test, submission, config):
    xgb_params = load_xgb_params(config)

    FEATURE_LIST = load_feature_list(config)
    COLS_TO_INPUT = train.columns[train.isnull().any()]
    CATEGORICAL_FEATURES = config.train_model.categorical_features
    
    y_train = train[config.train_model.is_churn]

    pipeline = get_model_pipeline(xgb_params, COLS_TO_INPUT, CATEGORICAL_FEATURES, get_weight_df(y_train))

    X_train = train[FEATURE_LIST]
    X_test = test[FEATURE_LIST]
    submission = submission[FEATURE_LIST]
    # Fit the pipeline to the training data
    pipeline = model_fit(X_train, y_train, pipeline, config.train_model.eval_size, config.train_model.early_stopping_rounds)
    
    test_predictions = pipeline.predict_proba(X_test)[:,1]
    submission_predictions = pipeline.predict_proba(submission)[:,1]

    return test_predictions, submission_predictions


@hydra.main(config_path="../../config", config_name="main", version_base=None)
def train_model(config: DictConfig):
    """Function to process the data"""
    train = pd.read_parquet(config.paths.processed.train)
    test = pd.read_parquet(config.paths.processed.test)
    submission = pd.read_parquet(config.paths.processed.submission)
    
    train_baseline_df = pd.read_parquet(config.paths.processed_baseline.train)
    test_baseline_df = pd.read_parquet(config.paths.processed_baseline.test)

    heuristic_test_predictions = generate_heuristic_baseline(train_baseline_df, test_baseline_df, config.train_model.baselines.heuristic.threshold)
    decision_test_tree_predictions = generate_decision_tree_predictions(train_baseline_df, test_baseline_df, config)
    xgb_classifier_test_predictions, xgb_classifier_submission_predictions = generate_xgb_classifier(train, test, submission, config)

    save_predictions(heuristic_test_predictions, decision_test_tree_predictions, xgb_classifier_test_predictions, xgb_classifier_submission_predictions, config)

if __name__ == "__main__":
    train_model()