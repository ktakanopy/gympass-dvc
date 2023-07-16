
This project was created using this (template)[https://github.com/gerbeldo/cookiecutter-dvc/tree/main].

# Notebook objective
- This notebook is the first step before the modelling phase. We are going to check noise data, nulls. And try to investigate patterns. 
- Main questions for the Exploratory Data Analysis (EDA):
    - Number of data points in train data and submission data.
    - How many users? How many users in train data and in submission? 
    - How many gyms?
    - How many churns, upgrades, and keep?
    - Do we have the same gyms/users in the application (submission) file?
    - What are the gyms with the highest churn rate?
    - How different is the months_usage between "top churn gyms" with others?
    - How is the "top churn gyms" distributions ? How is the distribution in relation of user plan?
    - Average user's age by gyms with top churns?
    - Average usage in months by gyms with top churns?
    - How the days since the first visit is different in each class/target (boxplot, histogram)?
    - How the target gym visits in lat 60 days is different in each class/target (boxplot, histogram)?
    - How the number of gyms is different in each class/target (boxplot, histogram)?
    - How the user age is different in each class/target (boxplot, histogram)?
    - How the user billing (user usage months) is different in each class/target (boxplot, histogram)?
    - How the user days since the first billing is different in each class/target (boxplot, histogram)?
    - How the user days since first visiting in target gym is different in each class/target (boxplot, histogram)?
    - How the user lifetime visits is different in each class/target (boxplot, histogram)?
    - How the user plan will affect the target? How is the distribution of churned by plans?
    - How the gym category affect the target? How is the distribution of churned by plans?
    - Who are the most loyal users of Gympass? And which gym will undergo the uptier?

# Assumptions
- I assumed the "application" file contain all information about the gym. And we are trying to predict churn for new gyms. 
- “Visited the product 60 days before the communication “ → The product here is the gym
- “All the data passed is just for the gyms that will suffer an uptier”.
- As the task is to focus more in churn than upgrade. I focused more the business metrics in for churn. But in real life could be better also to predict the revenue with upgrade. 
- Loyalty affects churn
- Recency/time affects churn
- User behavior and characteristics affects the gym. Example: user age, user location (in this case, the number of gyms in 5km)

## The way I understood some concepts and variables

"Gympass network": any gym registered in gympass.


VARIABLE x DESCRIPTION
- gym_days_since_first_visit: **The amount** of	days elapsed since first visit of **ANY** user
- gym_last_60_days_visits: **The amount** of visits of **ANY** user in last 60 days
- user_days_since_first_billing: **The amount** of days elapsed since first billing
- user_days_since_first_gym_visit:	**The amount** of days elapsed since first visit in gym that will suffer uptier
- user_days_since_first_visit: **Amount** of days elapsed since first visit to any gympass gym
- user_last_60_days_visit_share: **Amount** of visits in gym that will suffer uptier /**amount** of visits in gympass network, considering last 60 before communication
- user_last_60_days_visits:	**Total visits** in gympass network, considering last 60 before communication
- user_lifetime_visit_share:	During user lifetime, **total visits** in gym that will suffer uptier / **total visits** in gympass network


# Feature Engineering
- Multiple stats metrics (average, std deviation) of gyms features

# Improvements
- Use the features by different windows instead of just 60. To do this I would need the timestamp of each visit in gym
- Use user app interactions: user search tokens, time using the app, time using other gympass partnership apps (zenklub, etc)
- Use gym location, address, state, city, region. Maybe try to join with public data (ex: financial health of the location, if its local is dangerous)
- Get public information about the gym. Example: get the number of stars of the gym in google maps. Or the number of visits by day using google maps (Don't forget that each gym has its own membership in addition to the partnership with Gympass.)
- Get RFM and other loyalty metrics for each customer
- See how far the visited is from user home
- Synthesize data with TVAE
- Get Number of Upgrades in Past
- get Number of Downgrades in Past
- Model interpretation by sample cases
- Retrain in all database
- Use Orient Object Programming for to have a cleaner code
- Diagnose the warnings


# Questions
- Does the 'gym network' contains the target gym?
- Why the "_share" variables can be so high? 