
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from src.utils import get_column_indices


def generate_heuristic_baseline(train, test, user_engagement_quantile_thres=0.20):
    threshold_user_engagement = train.user_engagement.quantile(user_engagement_quantile_thres)
    heuristic_predict = (test.user_engagement < threshold_user_engagement).astype(bool)
    return heuristic_predict

def generate_decision_tree_predictions(train, test, config):
    numeric_features = config.train_model.baselines.decision_tree.numeric_features
    categorical_features = config.train_model.baselines.decision_tree.categorical_features
    
    features = numeric_features + categorical_features
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), get_column_indices(train[features], numeric_features)),
            ('cat', OneHotEncoder(), get_column_indices(train[features], categorical_features))])

    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Preprocessing (imputation + one-hot encoding)
        ('model', DecisionTreeClassifier(class_weight='balanced'))  # Model
    ])
    
    
    X_train = train[features]
    y_train =  train[config.train_model.target]

    X_test = test[features]

    pipeline.fit(X_train, y_train) 
    
    y_pred_proba = pipeline.predict_proba(X_test)[:,1]
    return y_pred_proba