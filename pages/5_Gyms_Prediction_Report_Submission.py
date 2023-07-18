import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(config: DictConfig):
    st.title("Gym prediction report for (submission data)")
    
    test_report = pd.read_csv("reports/submission_dataframe_report.csv")
    st.header("For test dataframe (with label)")
    
    st.subheader("Predictions total user churns")
    
    st.dataframe(test_report[['gym', 'total_users', 'total_pred_churns', 'percent_predicted_lost_users']])
    
    
    st.subheader("Predictions of gym profit (when changing user plan)")
    
    st.dataframe(test_report[['gym', 'predicted_profit', 'percent_profit_pred_churn']])

    st.subheader("Predicted churn gyms")

    st.markdown(f"- We indicate upgrade if percent_predicted_lost_users < {config.generate_reports.threshold_user_submission}") 

    st.dataframe(test_report[['gym', 'percent_predicted_lost_users', 'percent_profit_pred_churn',  'predicted_indicate_upgrade_users']])
    
    plt.figure(figsize=(10,6))
    ax = sns.histplot(x='percent_predicted_lost_users',data=test_report)
    plt.title(f'Histplot percent_predicted_lost_users')        
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height(), '{0:.2f}'.format(p.get_height()), 
                color='black', ha='center', va='bottom')
    
    
    plt.figure(figsize=(10,6))
    ax = sns.countplot(x='predicted_indicate_upgrade_users',data=test_report)
    plt.title(f'Predicted indicate upgrade count')        
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height(), '{0:.2f}'.format(p.get_height()), 
                color='black', ha='center', va='bottom')

    st.pyplot(plt)

    # st.header("Submission dataframe pandas report")
    
    # st.dataframe(sub_report)
    
if __name__ == "__main__":
    main()
