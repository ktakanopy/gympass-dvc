# Gympass churn prediction test
this project was designed with the goal of solving a churn prediction challenge for a database of gym memberships. This project was done with machine learning classification, DVC pipelines and streamlit.

The full description of the task can be found in the task folder.

You can find in this README:
- Strategy
- How to run the project
- Assumptions to do the project.
- Questions for data analysis
- Improvements that could be done.

# Strategy
The project was initiated using this [template](https://github.com/gerbeldo/cookiecutter-dvc/tree/main). 

the following steps were undertaken to address the problem:

- data cleaning: removed or treated any anomalies and noise in the data.
- data analysis: identified patterns and trends within the data.
- baseline creation: developed two baselines. the first is a heuristic-based on user behavior, and the second involves the use of decision trees with raw features.
- feature engineering: developed features assuming that user recency and frequency would contribute towards predicting churn. applied some transformations to support the machine learning classifier.
- model optimization: optimized an xgboost model's parameters using optuna for best results.
- result analysis: predicted and analyzed the results.
- refactoring: transformed jupyter notebooks into a git project with dvc and streamlit.
- model interpretation: interpreted the model with shap values.

# Getting started
1. Create a virtual environment:
```
make venv 
```
2. Activate the environment.
```
source .venv/bin/activate
```
3. Install the required dependencies:
```
make requirements
```
4. Create directories ignored by git.
```
make dirs
```
5. Run dvc pipelines. if you wish to run them manually, you'll need to add the current directory to the pythonpath variable:
```
make repro
```
6. Run streamlit!
```
streamlit run gympass_test_report.py
```

# Questions for the data analysis.
- Main questions for the exploratory data analysis (eda):
 - number of data points in train data and submission data.
- how many users? how many users in train data and in submission?
- how many gyms?
- how many churns, upgrades, and keep?
- do we have the same gyms/users in the application (submission) file?
- what are the gyms with the highest churn rate?
- how different is the months_usage between "top churn gyms" with others?
- how is the "top churn gyms" distributions? how is the distribution in relation to user plan?
- average user's age by gyms with top churns?
- average usage in months by gyms with top churns?
- how are the days since the first visit different in each class/target (boxplot, histogram)?
- how is the target gym visits in the last 60 days different in each class/target (boxplot, histogram)?
- how is the number of gyms different in each class/target (boxplot, histogram)?
- how is the user age different in each class/target (boxplot, histogram)?
- how is the user billing (user usage months) different in each class/target (boxplot, histogram)?
- how is the user days since the first billing different in each class/target (boxplot, histogram)?
- how is the user days since the first visiting in the target gym different in each class/target (boxplot, histogram)?
- how is the user lifetime visits different in each class/target (boxplot, histogram)?
- how does the user plan affect the target? how is the distribution of churned by plans?
- how does the gym category affect the target? how is the distribution of churned by plans?
- who are the most loyal users of gympass? and which gym will undergo the uptier?
- assumptions
- i assumed that the "application" file contains all the information about the gym and that we are trying to predict churn for new gyms.
- "visited the product 60 days before the communication" → the product here refers to the gym.
- "all the data passed is just for the gyms that will suffer an uptier."
- as the task is to focus more on churn than upgrade, i prioritized the business metrics for churn. however, in real life, it could be beneficial to also predict revenue with upgrades.
- loyalty affects churn.
- recency/time affects churn.
- user behavior and characteristics affect the gym, such as user age and user location (in this case, the number of gyms within a 5km radius).
## The way i understood some concepts and variables
 
- variable x description
 - gym_days_since_first_visit: **the amount** of	days elapsed since first visit of **any** user
 - gym_last_60_days_visits: **the amount** of visits of **any** user in last 60 days
 - user_days_since_first_billing: **the amount** of days elapsed since first billing
 - user_days_since_first_gym_visit:	**the amount** of days elapsed since first visit in gym that will suffer uptier
 - user_days_since_first_visit: **amount** of days elapsed since first visit to any gympass gym
 - user_last_60_days_visit_share: **amount** of visits in gym that will suffer uptier /**amount** of visits in gympass network, considering last 60 before communication
 - user_last_60_days_visits:	**total visits** in gympass network, considering last 60 before communication
 - user_lifetime_visit_share:	during user lifetime, **total visits** in gym that will suffer uptier / **total visits** in gympass network

# Improvements
- Utilize features based on different time windows instead of just 60 days. to do this, we would need the timestamp of each visit to the gym.
- Incorporate user app interactions, such as user search tokens, time spent using the app, and time spent using other gympass partnership apps (e.g., zenklub).
- Include gym location details such as address, state, city, and region. consider joining with public data, such as the financial health of the location or local safety information.
- Retrieve public information about the gym, such as the number of stars on google maps or the daily number of visits using google maps. note that each gym has its own membership in addition to the partnership with gympass.
- Incorporate rfm (recency, frequency, monetary) and other loyalty metrics for each customer.
- Analyze the distance between the gym visited and the user's home location.
- Synthesize data using a time-variant autoencoder (tvae).
- Capture the number of past upgrades and downgrades.
- Provide model interpretation for specific cases.
- Retrain the model using the entire database.
- Utilize object-oriented programming for cleaner code organization.
- Diagnose and address any warnings encountered.