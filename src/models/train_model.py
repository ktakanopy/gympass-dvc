import hydra
import pandas as pd
import shap
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from src.utils import load_feature_list, load_xgb_params, save_predictions, \
    generate_split, get_categorical_features
from collections import Counter
from src.models.baselines import generate_decision_tree_predictions, generate_heuristic_baseline
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import json


def get_weight_df(y):
    # count examples in each class
    counter = Counter(y)
    # estimate scale_pos_weight value
    estimate = counter[0] / counter[1]
    return estimate

def generate_model_fit_inputs(train, random_state, feature_list, target_name, test_size):
    train, valid = generate_split(train, test_size, 42)

    X_val = valid[feature_list]
    y_val = valid[target_name]
    X_train = train[feature_list]
    y_train = train[target_name]
    return X_train, y_train, X_val, y_val

def get_preprocessor(train):
    nan_cols_mask = train.isna().any()
    cols_to_input = train.columns[nan_cols_mask]
    categorical_features = get_categorical_features(train)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), cols_to_input),
            ('cat', TargetEncoder(), categorical_features)
            ],
        remainder='passthrough')

    return preprocessor


def model_fit(train, feature_list, params, target_name, test_size=0.2, random_state=42, early_stopping_rounds=20):
    preprocessor = get_preprocessor(train)
    

    X_train, y_train, X_val, y_val = generate_model_fit_inputs(train, random_state, feature_list, target_name, test_size)
    X_train = preprocessor.fit_transform(X_train, y_train)

    params['early_stopping_rounds'] = early_stopping_rounds # for validation and reduce overfit
    params['gpu_id'] = 0
    params['tree_method'] = 'gpu_hist'
    
    model = XGBClassifier(**params, scale_pos_weight=get_weight_df(y_train), objective='binary:logistic', eval_metric='logloss')
    eval_set = [(preprocessor.transform(X_val), y_val)]
    # fit the model
    model.fit(X_train, y_train, eval_set=eval_set)
    
    y_pred = model.predict(preprocessor.transform(X_val))
    report = classification_report(y_val, y_pred, output_dict=True)
    
    # Saving the classification report
    with open('reports/validation_report.json', 'w') as file:
        file.write(json.dumps(report))

    shap_values = shap.TreeExplainer(model).shap_values(X_train)

    f = plt.figure()
    shap.summary_plot(shap_values, X_train)
    f.savefig("reports/summary_shap_values.png", bbox_inches='tight', dpi=600)
 
    return model, preprocessor


def generate_xgb_classifier(train, test, submission, config):
    xgb_params = load_xgb_params(config)

    FEATURE_LIST = load_feature_list(config)
    
    X_test = test[FEATURE_LIST]
    submission = submission[FEATURE_LIST]
    
    # Fit the pipeline to the training data
    model, preprocessor = model_fit(train, FEATURE_LIST, xgb_params, config.train_model.target, \
        config.train_model.eval_size, 42, config.train_model.early_stopping_rounds)
    
    test_predictions = model.predict_proba(preprocessor.transform(X_test))[:,1]
    submission_predictions = model.predict_proba(preprocessor.transform(submission))[:,1]

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