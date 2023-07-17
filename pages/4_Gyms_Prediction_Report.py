import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
from sklearn.metrics import classification_report
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="main", version_base=None)
def main(config: DictConfig):
    st.title("Gym prediction report for test data")
    
    test_report = pd.read_csv("reports/test_dataframe_report.csv")
    st.header("For test dataframe (with label)")
    
    st.subheader("Predictions X Real Churn Users")
    
    st.dataframe(test_report[['gym', 'total_users', 'total_churns', 'total_pred_churns']])
    
    st.subheader("Predictions X Real Churn Users (Percent)")
    
    st.dataframe(test_report[['gym', 'total_users', 'percent_lost_users', 'percent_predicted_lost_users']])
    
    st.subheader("Predictions X Real Churn in Revenue")
    
    st.dataframe(test_report[['gym', 'total_revenue_real_churn', 'total_revenue_pred_churn']])

    st.subheader("Predictions X Real Churn in Profit")
    
    st.dataframe(test_report[['gym', 'real_profit', 'predicted_profit', 'percent_profit_real_churn', 'percent_profit_pred_churn']])

    st.subheader("Real Churn vs Predicted Churn")

    st.markdown(f"- We indicate upgrade if percent_predicted_lost_users < {config.generate_reports.threshold_user}") 
    st.dataframe(test_report[['gym', 'percent_lost_users', 'percent_predicted_lost_users', 'percent_profit_real_churn', 'percent_profit_pred_churn', 'indicate_upgrade', 'predicted_indicate_upgrade']])

 
    plt.figure(figsize=(10,6))
    ax = sns.countplot(x='indicate_upgrade',data=test_report)
    plt.title(f'Indicate upgrade count')        
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height(), '{0:.2f}'.format(p.get_height()), 
                color='black', ha='center', va='bottom')

    st.pyplot(plt)
    
    plt.figure(figsize=(10,6))
    ax = sns.countplot(x='predicted_indicate_upgrade',data=test_report)
    plt.title(f'Predicted indicate upgrade count')        
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height(), '{0:.2f}'.format(p.get_height()), 
                color='black', ha='center', va='bottom')

    st.pyplot(plt)

    st.subheader("Classification report")

    y_test = test_report['indicate_upgrade'] 
    y_pred = test_report['predicted_indicate_upgrade'] 
     

    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())


    # sub_report = pd.read_csv("reports/submission_dataframe_report.csv")
    # st.header("Submission dataframe pandas report")
    
    # st.dataframe(sub_report)
    
if __name__ == "__main__":
    main()
