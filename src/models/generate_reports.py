import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from src.utils import load_predictions, get_best_threshold_f1
import hydra
from omegaconf import DictConfig

next_upgrade_gym = {
    'Silver': "Silver +",
    'Basic I': "Basic II",
    'Basic II': "Silver",
    'Silver +': "Gold",
    "Free":  "Basic I"
}

plan_price_dict = {
    'Silver': 99.9,
    'Basic I': 39.9,
    'Basic II': 59.9,
    'Silver +': 140.9,
    'Gold': 210, # assumed
    'Free': 0
}

def generate_class_metrics_df(y_test, y_pred_class, labels):
    y_test_mapped = y_test.map({0: 'non-churn', 1: 'churn'})
    y_pred_class_mapped = y_pred_class.map({0: 'non-churn', 1: 'churn'})
    report_dict = classification_report(y_test_mapped, y_pred_class_mapped, output_dict=True, labels=labels)
    metrics_df = pd.DataFrame(report_dict).transpose()
    return metrics_df

def generate_confusion_matrix(y_test, y_pred_class):
    y_test_mapped = y_test.map({0: 'non-churn', 1: 'churn'})
    y_pred_class_mapped = y_pred_class.map({0: 'non-churn', 1: 'churn'})
    
    conf_matrix = confusion_matrix(y_test_mapped, y_pred_class_mapped)

    # Transform to dataframe for easier visualization
    conf_matrix_df = pd.DataFrame(conf_matrix, 
                                index=['actual_negative', 'actual_positive'], 
                                columns=['predicted_negative', 'predicted_positive'])
    return conf_matrix_df

def generate_dataframe_report(X_test, y_test, y_test_xgb, threshold_user):

    X_test['user_plan_price'] = X_test.user_plan.map(plan_price_dict)
    X_test['upgrade_plan'] = X_test.user_plan.map(next_upgrade_gym)
    X_test['upgrade_user_plan_price'] = X_test["upgrade_plan"].map(plan_price_dict)
        
    X_test['predict_churn'] = (y_test_xgb).astype(int)

    new_revenue = X_test.groupby("gym").upgrade_user_plan_price.sum().reset_index(name="upgrade_revenue")
    # new_revenue = new_revenue.rename(columns={'upgrade_user_': 'total_revenue'})
    X_test['is_churn'] = y_test
    total_users = X_test.groupby("gym").user.size().reset_index(name="total_users")
    total_churns = X_test.groupby("gym").is_churn.sum().reset_index(name="total_churns")
    total_pred_churns = X_test.groupby("gym").predict_churn.sum().reset_index(name="total_pred_churns")
    gym_plan = X_test.groupby("gym").upgrade_plan.first().reset_index(name="upgrade_plan")

    old_revenue = X_test.groupby("gym").user_plan_price.sum().reset_index(name="current_plan_revenue")

    gym_profits = new_revenue.merge(old_revenue,on="gym")
    gym_profits = gym_profits.merge(total_users, on="gym")
    gym_profits = gym_profits.merge(total_churns, on="gym")
    gym_profits = gym_profits.merge(gym_plan, on="gym")
    gym_profits = gym_profits.merge(total_pred_churns, on="gym")

    gym_profits['final_total_users'] = (gym_profits['total_users'] - gym_profits['total_churns'])
    gym_profits['percent_lost_users'] = round(gym_profits['total_churns']/gym_profits['total_users'],2)*100

    gym_profits['final_predicted_total_users'] = (gym_profits['total_users'] - gym_profits['total_pred_churns'])
    gym_profits['percent_predicted_lost_users'] = round(gym_profits['total_pred_churns']/gym_profits['total_users'],2)*100


    gym_profits['total_revenue_real_churn'] = gym_profits.apply(lambda x: plan_price_dict[x['upgrade_plan']]*x['final_total_users'] ,axis=1)
    gym_profits['total_revenue_pred_churn'] = gym_profits.apply(lambda x: plan_price_dict[x['upgrade_plan']]*x['final_predicted_total_users'] ,axis=1)

    gym_profits['real_profit'] = gym_profits['total_revenue_real_churn'] - gym_profits['current_plan_revenue']
    gym_profits['predicted_profit'] = gym_profits['total_revenue_pred_churn'] - gym_profits['current_plan_revenue']

    # gym_profits['profit'] = gym_profits['upgrade_revenue'] - gym_profits['current_plan_revenue']
    gym_profits['percent_profit_real_churn'] = round(gym_profits['real_profit']/gym_profits['current_plan_revenue'],2)*100
    gym_profits['percent_profit_pred_churn'] = round(gym_profits['predicted_profit']/gym_profits['current_plan_revenue'],2)*100

    # gym_profits = gym_profits[gym_profits.total_users > 10] # does not make sense to upgrade gyms with a small quantity of users

    gym_profits['indicate_upgrade'] = (gym_profits.percent_lost_users < threshold_user).astype(float)
    gym_profits['predicted_indicate_upgrade'] = (gym_profits.percent_predicted_lost_users <threshold_user).astype(float)
    return gym_profits

def generate_dataframe_report_submission(submission_data, y_test_xgb, threshold_user):
    
    submission_data['user_plan_price'] = submission_data.user_plan.map(plan_price_dict)
    submission_data['upgrade_plan'] = submission_data.user_plan.map(next_upgrade_gym)
    submission_data['upgrade_user_plan_price'] = submission_data["upgrade_plan"].map(plan_price_dict)
    submission_data['predict_churn'] = y_test_xgb

    new_revenue = submission_data.groupby("gym").upgrade_user_plan_price.sum().reset_index(name="upgrade_revenue")
    # new_revenue = new_revenue.rename(columns={'upgrade_user_': 'total_revenue'})
    total_users = submission_data.groupby("gym").user.size().reset_index(name="total_users")
    total_pred_churns = submission_data.groupby("gym").predict_churn.sum().reset_index(name="total_pred_churns")
    gym_plan = submission_data.groupby("gym").upgrade_plan.first().reset_index(name="upgrade_plan")

    old_revenue = submission_data.groupby("gym").user_plan_price.sum().reset_index(name="current_plan_revenue")

    gym_profits = new_revenue.merge(old_revenue,on="gym")
    gym_profits = gym_profits.merge(total_users, on="gym")
    gym_profits = gym_profits.merge(gym_plan, on="gym")
    gym_profits = gym_profits.merge(total_pred_churns, on="gym")

    gym_profits['final_predicted_total_users'] = (gym_profits['total_users'] - gym_profits['total_pred_churns'])
    gym_profits['percent_predicted_lost_users'] = round(gym_profits['total_pred_churns']/gym_profits['total_users'],2)*100


    gym_profits['total_revenue_pred_churn'] = gym_profits.apply(lambda x: plan_price_dict[x['upgrade_plan']]*x['final_predicted_total_users'] ,axis=1)

    gym_profits['predicted_profit'] = gym_profits['total_revenue_pred_churn'] - gym_profits['current_plan_revenue']

    gym_profits['percent_profit_pred_churn'] = round(gym_profits['predicted_profit']/gym_profits['current_plan_revenue'],2)*100

    # gym_profits = gym_profits[gym_profits.total_users > 10] # does not make sense to upgrade gyms with a small quantity of users

    gym_profits['predicted_indicate_upgrade_profit'] = (gym_profits.percent_profit_pred_churn > 40.0).astype(float)
    gym_profits['predicted_indicate_upgrade_users'] = (gym_profits.percent_predicted_lost_users < threshold_user).astype(float)
    return gym_profits 

@hydra.main(config_path="../../config", config_name="main", version_base=None)
def generate_reports(config: DictConfig):
    """Function to process the data"""
    heuristic_test_predictions, decision_test_tree_predictions, xgb_classifier_test_predictions, xgb_classifier_submission_predictions = load_predictions(config)
    
    test_baseline_df = pd.read_parquet(config.paths.processed_baseline.test)
    submission_data = pd.read_parquet(config.paths.processed.submission)
    y_test = test_baseline_df[config.generate_reports.target].reset_index(drop=True)
   
    labels = ["non-churn", "churn"]
    
    test_names_predictions = zip(["Heuristic-Baseline", "Decision-Tree-Baseline", "XGBClassifier"], \
        [heuristic_test_predictions, decision_test_tree_predictions, xgb_classifier_test_predictions])


    for name, predictions in test_names_predictions:
        print(f"--- {name} -----") 
        best_threshold_f1 = get_best_threshold_f1(y_test , predictions)
        predictions = pd.Series((predictions > best_threshold_f1).astype(int))
        if name == "XGBClassifier":
            y_test_xgb = (predictions > best_threshold_f1).astype(int)
            best_threshold_xgb = best_threshold_f1
        
        metrics_df = generate_class_metrics_df(y_test, predictions, labels)
        cf_df = generate_confusion_matrix(y_test, predictions)
        print("** class metrics:")
        print(metrics_df)
        print()
        print("** confusion matrix:")
        print(cf_df)   
        print()
        name = name.lower()
        metrics_df.to_csv(f"reports/test/{name}_class_metrics.csv")
        cf_df.to_csv(f"reports/test/{name}_confusion_matrix.csv")

    threshold_user = config.generate_reports.threshold_user
    dataframe_report = generate_dataframe_report(test_baseline_df, y_test, y_test_xgb, threshold_user)        
    dataframe_report.to_csv(f"reports/test_dataframe_report.csv", index=False)

    threshold_user_sub = config.generate_reports.threshold_user_submission
    
    y_test_xgb_sub = xgb_classifier_submission_predictions > best_threshold_xgb
    submission_report = generate_dataframe_report_submission(submission_data, y_test_xgb_sub, threshold_user_sub) 
    submission_report.to_csv(f"reports/submission_dataframe_report.csv", index=False)


if __name__ == "__main__":
    generate_reports()