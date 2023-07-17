# EDA

## Number of data points in train data and submission data.


```python
train_data_size = train_data.shape[0]
submission_data_size = submission_data.shape[0]
print(f"Trainining data size {train_data_size}")
print(f"Submission data size {submission_data_size}")
```

    Trainining data size 31261
    Submission data size 5556


## How many users in train data and in submission? 


```python
number_users = train_data.user.unique().shape[0]

print(f" Ratio of users in submission and train size {number_users/train_data_size}")
```

     Ratio of users in submission and train size 1.0



```python
number_users_sub = submission_data.user.unique().shape[0]

print(f" Ratio of users in submission and submission size {number_users_sub/submission_data_size}")
```

     Ratio of users in submission and submission size 1.0


## Do we have the same gyms and users in the submission?


```python
gym_ids_train = set(train_data.gym.unique())
user_ids_train = set(train_data.user.unique())

gym_ids_sub = set(submission_data.gym.unique())
user_ids_sub = set(submission_data.user.unique())

print(f"Total of gym ids in intersection {len(gym_ids_train.intersection(gym_ids_sub))}")
```

    Total of gym ids in intersection 0



```python
print(f"Total of gym ids in intersection {len(user_ids_train.intersection(user_ids_sub))}")
```

    Total of gym ids in intersection 0


## How many churns, upgrades, and keep?


```python
plot_count(train_data, column='multiclass_target_names', title='Multi class distribution', xlabel = "Target classes", sort=True)
```


    
![png](pages/assets/output_10_0.png)
    



```python
plot_count(train_data, column='churn_target_names', title='Churn distribution', xlabel = "Churn classes", sort=True)
```


    
![png](pages/assets/output_11_0.png)
    


## How distributed is the user_engagement created feature value?

- Non-churn users has lower user_engagement. This means they have more user_billings (more payments)


```python
plot_histogram(train_data, 'user_engagement', hue='churn_target_names')
```


    
![png](pages/assets/output_14_0.png)
    



```python
sns.boxplot(data=train_data[FEATURES+ ['churn_target_names']], x='churn_target_names', y='user_engagement')
```




    <Axes: xlabel='churn_target_names', ylabel='user_engagement'>




    
![png](pages/assets/output_15_1.png)
    


## How distributed is the gym_visit_frequency and user_visit_frequency created feature value?


```python
sns.boxplot(data=train_data[FEATURES+ ['churn_target_names']], x='churn_target_names', y='gym_visit_frequency')
```




    <Axes: xlabel='churn_target_names', ylabel='gym_visit_frequency'>




    
![png](pages/assets/output_17_1.png)
    



```python
sns.boxplot(data=train_data[FEATURES+ ['churn_target_names']], x='churn_target_names', y='user_visit_frequency')
```




    <Axes: xlabel='churn_target_names', ylabel='user_visit_frequency'>




    
![png](pages/assets/output_18_1.png)
    


## What are the gyms with the highest churn rate?


```python
churn_rate = train_data.groupby("gym")['churn_target'].mean().sort_values()
churn_rate.hist()
```




    <Axes: >




    
![png](pages/assets/output_20_1.png)
    



```python
top_churn_rate_gyms = list(churn_rate[churn_rate > 0.5].index)
print("Total of top churn gyms", len(top_churn_rate_gyms))
```

    Total of top churn gyms 16



```python
# train_data.groupby('churn_target_names').months_usage.mean().plot(kind='bar')
train_data['is_top_churn_gym'] = False
train_data.loc[train_data.gym.isin(top_churn_rate_gyms), 'is_top_churn_gym'] = True
```

## How is the "top churn gyms" distributions ? How is the distribution in relation of user plan?


```python
plot_count(train_data, 'is_top_churn_gym', title="Distribution of top churn gyms")
```


    
![png](pages/assets/output_24_0.png)
    



```python
plot_count(train_data, 'is_top_churn_gym', hue='user_plan', log_scale=True, title="Distribution of top churns ")
```


    
![png](pages/assets/output_25_0.png)
    


## How different is the gyms_5km by churn and non-churn groups?


```python
plot_avg(train_data, 'churn_target_names', 'gyms_5km', sort=True)
```


    
![png](pages/assets/output_27_0.png)
    


## How different is the gym_days_since_first_visit by churn and non-churn groups?


```python
plot_avg(train_data, 'churn_target_names', 'gym_days_since_first_visit', sort=True)
```


    
![png](pages/assets/output_29_0.png)
    


## How different is the gym_last_60_days_visits by churn and non-churn groups?


```python
plot_avg(train_data, 'churn_target_names', 'gym_last_60_days_visits', sort=True)
```


    
![png](pages/assets/output_31_0.png)
    


## How different is the months_usage by churn and non-churn groups?


```python
plot_avg(train_data, 'churn_target_names', 'months_usage', sort=True)
```


    
![png](pages/assets/output_33_0.png)
    


## How different is the user_age between by churn and non-churn groups?


```python
plot_avg(train_data, 'churn_target_names', 'user_age', sort=True)
```


    
![png](pages/assets/output_35_0.png)
    


- Old people give up less


```python
plot_count(train_data, 'churn_target_names', hue='user_age_group', log_scale=False, title="Distribution age group by churn and non-churn users")
```


    
![png](pages/assets/output_37_0.png)
    


## How different is the user_days_since_first_billing by churn and non-churn groups?

- Churn users are more likely to be younger on the platform.


```python
plot_avg(train_data, 'churn_target_names', 'user_days_since_first_billing', sort=True)
```


    
![png](pages/assets/output_40_0.png)
    


## How different is the user_days_since_first_visit by churn and non-churn groups?


```python
plot_avg(train_data, 'churn_target_names', 'user_days_since_first_visit', sort=True)
```


    
![png](pages/assets/output_42_0.png)
    


# Bivariate pairplot


```python
!pip install -U imbalanced-learn
```

    Requirement already satisfied: imbalanced-learn in /home/takano/miniconda3/envs/lightgbm/lib/python3.9/site-packages (0.10.1)
    Collecting imbalanced-learn
      Downloading imbalanced_learn-0.11.0-py3-none-any.whl (235 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m235.6/235.6 kB[0m [31m44.1 kB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: joblib>=1.1.1 in /home/takano/miniconda3/envs/lightgbm/lib/python3.9/site-packages (from imbalanced-learn) (1.2.0)
    Requirement already satisfied: numpy>=1.17.3 in /home/takano/miniconda3/envs/lightgbm/lib/python3.9/site-packages (from imbalanced-learn) (1.24.3)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /home/takano/miniconda3/envs/lightgbm/lib/python3.9/site-packages (from imbalanced-learn) (3.1.0)
    Requirement already satisfied: scipy>=1.5.0 in /home/takano/miniconda3/envs/lightgbm/lib/python3.9/site-packages (from imbalanced-learn) (1.10.1)
    Requirement already satisfied: scikit-learn>=1.0.2 in /home/takano/miniconda3/envs/lightgbm/lib/python3.9/site-packages (from imbalanced-learn) (1.2.2)
    Installing collected packages: imbalanced-learn
      Attempting uninstall: imbalanced-learn
        Found existing installation: imbalanced-learn 0.10.1
        Uninstalling imbalanced-learn-0.10.1:
          Successfully uninstalled imbalanced-learn-0.10.1
    Successfully installed imbalanced-learn-0.11.0



```python
from imblearn.under_sampling import RandomUnderSampler

# Assume df is your dataframe and 'target' is your target column
X = train_data[FEATURES]
y = train_data['churn_target_names']

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# Concatenate our training data back together
df_res = pd.concat([pd.DataFrame(y_res), pd.DataFrame(X_res)], axis=1)
df_res.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>churn_target_names</th>
      <th>months_usage</th>
      <th>years_usage</th>
      <th>gyms_5km</th>
      <th>user_age</th>
      <th>gym_visit_frequency</th>
      <th>gym_category</th>
      <th>user_visit_frequency</th>
      <th>user_lifetime_visit_share</th>
      <th>gym_days_since_first_visit</th>
      <th>user_lifetime_visits</th>
      <th>user_last_60_days_visit_share</th>
      <th>user_plan</th>
      <th>user_age_group</th>
      <th>user_days_since_first_visit</th>
      <th>user_days_since_first_billing</th>
      <th>user_last_60_days_visits</th>
      <th>user_days_since_first_gym_visit</th>
      <th>user_engagement</th>
      <th>gym_last_60_days_visits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>is_churn</td>
      <td>48</td>
      <td>4.000000</td>
      <td>11</td>
      <td>27</td>
      <td>16.666667</td>
      <td>bodybuilding</td>
      <td>0.566667</td>
      <td>9.892473e+14</td>
      <td>2129</td>
      <td>558.0</td>
      <td>1.000000e+00</td>
      <td>Silver</td>
      <td>2</td>
      <td>1456</td>
      <td>2539.0</td>
      <td>34.0</td>
      <td>1456</td>
      <td>11.625000</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>is_churn</td>
      <td>22</td>
      <td>1.833333</td>
      <td>0</td>
      <td>70</td>
      <td>17.316667</td>
      <td>not found</td>
      <td>0.016667</td>
      <td>6.600000e-01</td>
      <td>1131</td>
      <td>200.0</td>
      <td>1.000000e+00</td>
      <td>Basic II</td>
      <td>6</td>
      <td>1316</td>
      <td>1345.0</td>
      <td>1.0</td>
      <td>871</td>
      <td>9.090909</td>
      <td>1039</td>
    </tr>
    <tr>
      <th>26</th>
      <td>is_churn</td>
      <td>15</td>
      <td>1.250000</td>
      <td>35</td>
      <td>19</td>
      <td>31.000000</td>
      <td>bodybuilding</td>
      <td>0.250000</td>
      <td>1.463415e+16</td>
      <td>1168</td>
      <td>41.0</td>
      <td>6.666667e+15</td>
      <td>Basic II</td>
      <td>1</td>
      <td>1682</td>
      <td>1867.0</td>
      <td>15.0</td>
      <td>522</td>
      <td>2.733333</td>
      <td>1860</td>
    </tr>
    <tr>
      <th>42</th>
      <td>is_churn</td>
      <td>11</td>
      <td>0.916667</td>
      <td>29</td>
      <td>35</td>
      <td>5.583333</td>
      <td>functional</td>
      <td>0.750000</td>
      <td>4.455959e+16</td>
      <td>275</td>
      <td>193.0</td>
      <td>1.111111e+15</td>
      <td>Basic I</td>
      <td>3</td>
      <td>273</td>
      <td>273.0</td>
      <td>45.0</td>
      <td>273</td>
      <td>17.545455</td>
      <td>335</td>
    </tr>
    <tr>
      <th>43</th>
      <td>is_churn</td>
      <td>6</td>
      <td>0.500000</td>
      <td>44</td>
      <td>53</td>
      <td>159.016667</td>
      <td>bodybuilding</td>
      <td>0.100000</td>
      <td>1.000000e+00</td>
      <td>747</td>
      <td>10.0</td>
      <td>1.000000e+00</td>
      <td>Basic I</td>
      <td>4</td>
      <td>138</td>
      <td>139.0</td>
      <td>6.0</td>
      <td>138</td>
      <td>1.666667</td>
      <td>9541</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_res.churn_target_names.value_counts()
```




    churn_target_names
    is_churn        5361
    is_not_churn    5361
    Name: count, dtype: int64




```python
NUMERIC_FEATURES
```




    ['gym_days_since_first_visit',
     'gym_last_60_days_visits',
     'gym_visit_frequency',
     'gyms_5km',
     'months_usage',
     'user_age',
     'user_age_group',
     'user_days_since_first_billing',
     'user_days_since_first_gym_visit',
     'user_days_since_first_visit',
     'user_engagement',
     'user_last_60_days_visit_share',
     'user_last_60_days_visits',
     'user_lifetime_visit_share',
     'user_lifetime_visits',
     'user_visit_frequency',
     'years_usage']




```python
sns.pairplot(df_res[['user_lifetime_visits', 'months_usage', 'user_engagement', 'gym_visit_frequency'] + ['churn_target_names']].sample(2000), hue='churn_target_names', plot_kws={'alpha':0.2})
```




    <seaborn.axisgrid.PairGrid at 0x7f040615a6d0>




    
![png](pages/assets/output_48_1.png)
    


- trying to create 'loyalty' features


```python
NUMERIC_FEATURES
```




    ['gym_days_since_first_visit',
     'gym_last_60_days_visits',
     'gym_visit_frequency',
     'gyms_5km',
     'months_usage',
     'user_age',
     'user_age_group',
     'user_days_since_first_billing',
     'user_days_since_first_gym_visit',
     'user_days_since_first_visit',
     'user_engagement',
     'user_last_60_days_visit_share',
     'user_last_60_days_visits',
     'user_lifetime_visit_share',
     'user_lifetime_visits',
     'user_visit_frequency',
     'years_usage']




```python
# remove _share features as it hard to understand
FEATURES_ANALYZE = [k for k in NUMERIC_FEATURES if '_share' not in k]
```


```python
FEATURES_ANALYZE
```




    ['gym_days_since_first_visit',
     'gym_last_60_days_visits',
     'gym_visit_frequency',
     'gyms_5km',
     'months_usage',
     'user_age',
     'user_age_group',
     'user_days_since_first_billing',
     'user_days_since_first_gym_visit',
     'user_days_since_first_visit',
     'user_engagement',
     'user_last_60_days_visits',
     'user_lifetime_visits',
     'user_visit_frequency',
     'years_usage']



# Plotting features vs target

### BoxPlot


```python
import math

# calculate the number of rows and columns needed for the subplots
num_vars = len(FEATURES_ANALYZE)
num_cols = 2  # adjust as needed
num_rows = math.ceil(num_vars / num_cols)

fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows*6))

for i, numeric_var in enumerate(FEATURES_ANALYZE):
    row = i // num_cols
    col = i % num_cols
    ax = axs[row, col]
    sns.boxplot(data=train_data[FEATURES_ANALYZE + ['churn_target_names']], x='churn_target_names', y=numeric_var, ax=ax)
    ax.set_title(f'{numeric_var} by Target Class')
    ax.set_xlabel('Target Classes')
    ax.set_ylabel(f"{numeric_var}")

# Remove unused subplots
if num_rows * num_cols > num_vars:
    for j in range(i+1, num_rows * num_cols):
        fig.delaxes(axs.flatten()[j])

plt.tight_layout()
# Save the figure
plt.savefig('viz/numeric_features_binary_box_plot.png')

plt.show()
```


    
![png](pages/assets/output_55_0.png)
    


### Histograms


```python
import math

# calculate the number of rows and columns needed for the subplots
num_vars = len(FEATURES_ANALYZE)
num_cols = 3  # adjust as needed
num_rows = math.ceil(num_vars / num_cols)

fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows*6))

for i, numeric_var in enumerate(FEATURES_ANALYZE):
    row = i // num_cols
    col = i % num_cols
    ax = axs[row, col]
    sns.histplot(data=df_res[FEATURES_ANALYZE + ['churn_target_names']], x=numeric_var, hue='churn_target_names', ax=ax, stat='density')
    ax.set_title(f'{numeric_var} Distribution by Target Class')
    ax.set_xlabel(f"{numeric_var}")
    ax.set_ylabel('Count')

# Remove unused subplots
if num_rows * num_cols > num_vars:
    for j in range(i+1, num_rows * num_cols):
        fig.delaxes(axs.flatten()[j])


plt.tight_layout()
# Save the figure
plt.savefig('viz/numeric_features_histogram.png')
plt.show()
```


    
![png](pages/assets/output_57_0.png)
    


# Conclusions
- Imbalaced dataset
- Skewed features
- People tend to not churn
- Top churn gyms have more basic II and silver plans 
- Old people (66+) is less likely too churn
- The user frequency and recency may affects the churn
- We can combine features (user engagement) to see some separation between classes.
- Some features don't seem to help (e.g age, gyms_5k)


```python

```
