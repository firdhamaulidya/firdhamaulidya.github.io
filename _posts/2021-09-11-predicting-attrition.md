---
layout: post
title: "Predicting Employee Attrition with Gradient Boosting"
subtitle: "Final Project for Data Science Bootcamp Rakamin Academy Batch 11"
background: '/img/posts/predicting-attrition/bg1.jpg'
---

# Predicting Employee Attrition with Gradient Boosting

This project has been created in order to complete the final project at the Rakamin Data Science bootcamp Batch 11 Group 6.
We use IBM HR Analytics Employee Attrition & Performance dataset from **[kaggle](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)**.

For complete explanation, please come to this **[GoogleDrive](https://drive.google.com/file/d/19Wc1l5SVXhYpi2mfiq-SsFMtecyNJN5B/view?usp=sharing)**.

## Import Packages and Files


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import plotly.express as px
from scipy.stats import pearsonr
import warnings
warnings.simplefilter('ignore')

print(np.__version__)
print(pd.__version__)
```

    C:\Users\Firdha\AppData\Roaming\Python\Python37\site-packages\pandas\compat\_optional.py:138: UserWarning: Pandas requires version '2.7.0' or newer of 'numexpr' (version '2.6.8' currently installed).
      warnings.warn(msg, UserWarning)
    

    1.19.2
    1.3.1
    

## Load File


```python
# copy path dari dataset di drive masing-masing
raw = pd.read_csv('Employee Attrition.csv')
```

## Data Pre-Processing


```python
raw.head()
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
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
raw.isnull().sum().sum()
```




    0




```python
raw.duplicated().sum()
```




    0



This dataset consists of 35 columns and 1,470 rows. No *missing value* and *duplicate* data in this dataset.

There are 2 datatypes, *integer* and *object*.


```python
categorical_features = raw.select_dtypes(include=[np.object]).columns
print(f"Total categorical features: {len(categorical_features)}")

numerical_features = raw.select_dtypes(include=[np.int64]).columns
print(f"Total numerical features: {len(numerical_features)}")
```

    Total categorical features: 9
    Total numerical features: 26
    


```python
overview = [[column, raw[column].dtypes, raw[column].nunique(), raw[column].unique()] for column in raw.columns]
overview = pd.DataFrame(overview, columns = ['Column', 'Data Type', 'Num of Unique Values', 'Values'])
overview
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
      <th>Column</th>
      <th>Data Type</th>
      <th>Num of Unique Values</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>int64</td>
      <td>43</td>
      <td>[41, 49, 37, 33, 27, 32, 59, 30, 38, 36, 35, 2...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Attrition</td>
      <td>object</td>
      <td>2</td>
      <td>[Yes, No]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BusinessTravel</td>
      <td>object</td>
      <td>3</td>
      <td>[Travel_Rarely, Travel_Frequently, Non-Travel]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DailyRate</td>
      <td>int64</td>
      <td>886</td>
      <td>[1102, 279, 1373, 1392, 591, 1005, 1324, 1358,...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Department</td>
      <td>object</td>
      <td>3</td>
      <td>[Sales, Research &amp; Development, Human Resources]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DistanceFromHome</td>
      <td>int64</td>
      <td>29</td>
      <td>[1, 8, 2, 3, 24, 23, 27, 16, 15, 26, 19, 21, 5...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Education</td>
      <td>int64</td>
      <td>5</td>
      <td>[2, 1, 4, 3, 5]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>EducationField</td>
      <td>object</td>
      <td>6</td>
      <td>[Life Sciences, Other, Medical, Marketing, Tec...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>EmployeeCount</td>
      <td>int64</td>
      <td>1</td>
      <td>[1]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>EmployeeNumber</td>
      <td>int64</td>
      <td>1470</td>
      <td>[1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16,...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>EnvironmentSatisfaction</td>
      <td>int64</td>
      <td>4</td>
      <td>[2, 3, 4, 1]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Gender</td>
      <td>object</td>
      <td>2</td>
      <td>[Female, Male]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>HourlyRate</td>
      <td>int64</td>
      <td>71</td>
      <td>[94, 61, 92, 56, 40, 79, 81, 67, 44, 84, 49, 3...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>JobInvolvement</td>
      <td>int64</td>
      <td>4</td>
      <td>[3, 2, 4, 1]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>JobLevel</td>
      <td>int64</td>
      <td>5</td>
      <td>[2, 1, 3, 4, 5]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>JobRole</td>
      <td>object</td>
      <td>9</td>
      <td>[Sales Executive, Research Scientist, Laborato...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>JobSatisfaction</td>
      <td>int64</td>
      <td>4</td>
      <td>[4, 2, 3, 1]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MaritalStatus</td>
      <td>object</td>
      <td>3</td>
      <td>[Single, Married, Divorced]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MonthlyIncome</td>
      <td>int64</td>
      <td>1349</td>
      <td>[5993, 5130, 2090, 2909, 3468, 3068, 2670, 269...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>MonthlyRate</td>
      <td>int64</td>
      <td>1427</td>
      <td>[19479, 24907, 2396, 23159, 16632, 11864, 9964...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>NumCompaniesWorked</td>
      <td>int64</td>
      <td>10</td>
      <td>[8, 1, 6, 9, 0, 4, 5, 2, 7, 3]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Over18</td>
      <td>object</td>
      <td>1</td>
      <td>[Y]</td>
    </tr>
    <tr>
      <th>22</th>
      <td>OverTime</td>
      <td>object</td>
      <td>2</td>
      <td>[Yes, No]</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PercentSalaryHike</td>
      <td>int64</td>
      <td>15</td>
      <td>[11, 23, 15, 12, 13, 20, 22, 21, 17, 14, 16, 1...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PerformanceRating</td>
      <td>int64</td>
      <td>2</td>
      <td>[3, 4]</td>
    </tr>
    <tr>
      <th>25</th>
      <td>RelationshipSatisfaction</td>
      <td>int64</td>
      <td>4</td>
      <td>[1, 4, 2, 3]</td>
    </tr>
    <tr>
      <th>26</th>
      <td>StandardHours</td>
      <td>int64</td>
      <td>1</td>
      <td>[80]</td>
    </tr>
    <tr>
      <th>27</th>
      <td>StockOptionLevel</td>
      <td>int64</td>
      <td>4</td>
      <td>[0, 1, 3, 2]</td>
    </tr>
    <tr>
      <th>28</th>
      <td>TotalWorkingYears</td>
      <td>int64</td>
      <td>40</td>
      <td>[8, 10, 7, 6, 12, 1, 17, 5, 3, 31, 13, 0, 26, ...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>TrainingTimesLastYear</td>
      <td>int64</td>
      <td>7</td>
      <td>[0, 3, 2, 5, 1, 4, 6]</td>
    </tr>
    <tr>
      <th>30</th>
      <td>WorkLifeBalance</td>
      <td>int64</td>
      <td>4</td>
      <td>[1, 3, 2, 4]</td>
    </tr>
    <tr>
      <th>31</th>
      <td>YearsAtCompany</td>
      <td>int64</td>
      <td>37</td>
      <td>[6, 10, 0, 8, 2, 7, 1, 9, 5, 4, 25, 3, 12, 14,...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>YearsInCurrentRole</td>
      <td>int64</td>
      <td>19</td>
      <td>[4, 7, 0, 2, 5, 9, 8, 3, 6, 13, 1, 15, 14, 16,...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>YearsSinceLastPromotion</td>
      <td>int64</td>
      <td>16</td>
      <td>[0, 1, 3, 2, 7, 4, 8, 6, 5, 15, 9, 13, 12, 10,...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>YearsWithCurrManager</td>
      <td>int64</td>
      <td>18</td>
      <td>[5, 7, 0, 2, 6, 8, 3, 11, 17, 1, 4, 12, 9, 10,...</td>
    </tr>
  </tbody>
</table>
</div>



The table above provides an overall overview of the values contained in each column. There are some important **insights**:
- Some columns only have 1 unique value, i.e EmployeeCount, Over18, and StandardHours columns. These columns are not informative so they need to be **take out**.
- The EmployeeNumber column has unique value as many as the number of rows, so it can be used as an **identifier**.
- DailyRate, HourlyRate, MonthlyRate, and MonthlyIncome at a glance refer to the same thing, employee salary. The relationship between them needs to be investigated further.
- PerformanceRating has two unique values, 3 and 4. The scale to measure it seems unclear, but it can still be used to perceive order because 4 is greater than 3
- There are three columns oriented to *satisfaction*, i.e EnvironmentSatisfaction, JobSatisfaction, and RelationshipSatisfaction. The three of them might be able to merge into General Satisfaction

## The Follow-Ups


```python
# make a copy of the dataset
df = raw.copy()

# remove columns with only single unique value
df = df.drop(['EmployeeCount','Over18','StandardHours'], axis=1)

# rename EmployeeNumber to id, then make it the index
df.rename({'EmployeeNumber':'id'}, axis=1, inplace=True)
df.set_index('id', inplace=True)

# select integer columns based on whether it is continous or categorical
df_cat = df[['Education','EnvironmentSatisfaction','JobInvolvement','JobLevel',
             'JobSatisfaction','PerformanceRating','RelationshipSatisfaction',
             'StockOptionLevel','WorkLifeBalance']]
df_cont = df.select_dtypes(include=[np.int64]).drop(df_cat, axis=1)
```

## Exploratory Data Analysis (EDA)

EDA has **[three purposes](https://www.ibm.com/cloud/learn/exploratory-data-analysis)**:
1. Data Understanding
2. Data Cleaning
3. Multivariate Analysis

### Data Understanding

### Detecting outliers


```python
# create box plots out of each continuous column to detect outliers visually
fig, ax = plt.subplots(3,5, figsize=(20,12))

for num in np.arange(0,5):
    sns.boxplot(x=df_cont.columns[num], data=df_cont, ax=ax[0,num])

for num in np.arange(0,5):
    sns.boxplot(x=df_cont.columns[num + 5], data=df_cont, ax=ax[1,num])

for num in np.arange(0,5):
    try:
        sns.boxplot(x=df_cont.columns[num + 10], data=df_cont, ax=ax[2,num])
    except:
        pass
```


![png](/img/posts/predicting-attrition/output_19_0.png)



```python
# create a function to calculate proportion of outliers
def outliers_perc(kolom):
    Q1 = df[kolom].describe()['25%']
    Q3 = df[kolom].describe()['75%']
    iqr = Q3-Q1
    low_lim = Q1 - iqr*1.5
    up_lim = Q3 + iqr*1.5

    x = df[kolom][(df[kolom] < low_lim) | (df[kolom] > up_lim)].value_counts().sum()
    return round(x/df[kolom].shape[0],4)*100

print('MonthlyIncome: {}'.format(outliers_perc('MonthlyIncome')))
print('TotalWorkingYears: {}'.format(outliers_perc('TotalWorkingYears')))
print('YearsAtCompany: {}'.format(outliers_perc('YearsAtCompany')))
print('YearsSinceLastPromotion: {}'.format(outliers_perc('YearsSinceLastPromotion')))
```

    MonthlyIncome: 7.76
    TotalWorkingYears: 4.29
    YearsAtCompany: 7.07
    YearsSinceLastPromotion: 7.28
    

Several *continuous* columns appear to have *outliers* values. Columns that appear to have many *outliers* values are:
- MonthlyIncome (7,76%)
- TotalWorkingYears (4,29%)
- YearsAtCompany (7,07%)
- YearsSinceLastPromotion (7,28%)


It is not known whether these many *outliers* will significantly reduce the model's performance. So, in the first iteration, these columns will be preserved along with the *outliers* in them.

### Testing normality assumption


```python
# import norm from scipy to test normality
from scipy.stats import norm

# create distribution plot out of each continuous column to view normality
fig, ax = plt.subplots(3,5, figsize=(25,12))

for num in np.arange(0,5):
    sns.distplot(df[df_cont.columns[num]], hist=False, fit=norm, ax=ax[0,num])

for num in np.arange(0,5):
    sns.distplot(df[df_cont.columns[num + 5]], hist=False, fit=norm, ax=ax[1,num])

for num in np.arange(0,5):
    try:
        sns.distplot(df[df_cont.columns[num + 10]], hist=False, fit=norm, ax=ax[2,num])
    except:
        pass
```


![png](/img/posts/predicting-attrition/output_23_0.png)


Only a few columns in dataset which approach normality visually:
- Age
- MonthlyIncome
- PercentSalaryHike
- TotalWorkingYears
- YearsAtCompany

### Data Cleaning

*Target variable* in this dataset is Attrition, but the datatype is *object*. In order to make model, the Attrition data type needs to be converted into integer form with the notation 0 for "No" and 1 for "Yes".

In addition, all columns that have a number of unique values ​​of 2 also need to be converted to 0 and 1 notation so that the modeling is not disturbed. Categorical columns with unique values ​​more than 2 will be processed in another way to avoid the impression of *order* which is actually an intrinsic property of integers.


```python
# filter any values from column with dual category
overview[overview['Num of Unique Values'] == 2]
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
      <th>Column</th>
      <th>Data Type</th>
      <th>Num of Unique Values</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Attrition</td>
      <td>object</td>
      <td>2</td>
      <td>[Yes, No]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Gender</td>
      <td>object</td>
      <td>2</td>
      <td>[Female, Male]</td>
    </tr>
    <tr>
      <th>22</th>
      <td>OverTime</td>
      <td>object</td>
      <td>2</td>
      <td>[Yes, No]</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PerformanceRating</td>
      <td>int64</td>
      <td>2</td>
      <td>[3, 4]</td>
    </tr>
  </tbody>
</table>
</div>




```python
# convert all columns with dual category to integer
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
df['PerformanceRating'] = df['PerformanceRating'].apply(lambda x: 1 if x == 4 else 0)
```


```python
# add the recently converted columns to categorical group
df_cat = pd.concat([df_cat, df['Attrition'], df['Gender'], df['OverTime'], df['PerformanceRating']], axis=1)

# remove duplicated PerformanceRating column (column 5)
column_numbers = [x for x in range(df_cat.shape[1])]
column_numbers.remove(5)
df_cat = df_cat.iloc[:, column_numbers]
```

### Multivariate Analysis

#### How Attrition varies across other variables


```python
fig, ax = plt.subplots(figsize=(5,5))
sns.countplot(df['Attrition'])
plt.xticks([0,1],['No','Yes']);
```


![png](/img/posts/predicting-attrition/output_32_0.png)


The first thing that needs to be observed is that the number of employees who experience attrition is far less than those who do not experience attrition. Given its role as a target variable, it means that this dataset has *class imbalance*.

##### Department and JobRole x Attrition


```python
# create dataframe to visualize
df_dep = df.reset_index()

dep = df_dep.groupby(['Department','Attrition']).agg({'id':'sum'})
dep['total'] = dep.groupby('Department')['id'].transform('sum')
dep['Percentage'] = round(dep['id'] / dep['total'] * 100,2)
dep.reset_index(inplace=True)
dep.drop(dep[dep['Attrition'] == 'No'].index, inplace=True)

# visualize
plt.figure(figsize=(7,4))
sns.barplot('Department', 'Percentage', hue='Attrition', ci=None, data=dep)
plt.title('Persentase Attrition di Setiap Departemen');
```


![png](/img/posts/predicting-attrition/output_35_0.png)



```python
df_id = df.reset_index()

role = df_id.groupby(['Department','JobRole','Attrition']).agg({'id':'count'})
role['In-Department Percentage'] = role.groupby('Department')['id'].transform('sum')
role['In-Department Percentage'] = round(role['id'] / role['In-Department Percentage'] * 100,2)
role.reset_index(inplace=True)
role.drop(role[role['Attrition'] == 0].index, inplace=True)
role.rename({'id':'Count'}, axis=1, inplace=True)
role.sort_values('In-Department Percentage', ascending=False).reset_index(drop=True)
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
      <th>Department</th>
      <th>JobRole</th>
      <th>Attrition</th>
      <th>Count</th>
      <th>In-Department Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Human Resources</td>
      <td>Human Resources</td>
      <td>1</td>
      <td>12</td>
      <td>19.05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sales</td>
      <td>Sales Executive</td>
      <td>1</td>
      <td>57</td>
      <td>12.78</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sales</td>
      <td>Sales Representative</td>
      <td>1</td>
      <td>33</td>
      <td>7.40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Research &amp; Development</td>
      <td>Laboratory Technician</td>
      <td>1</td>
      <td>62</td>
      <td>6.45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Research &amp; Development</td>
      <td>Research Scientist</td>
      <td>1</td>
      <td>47</td>
      <td>4.89</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Research &amp; Development</td>
      <td>Manufacturing Director</td>
      <td>1</td>
      <td>10</td>
      <td>1.04</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Research &amp; Development</td>
      <td>Healthcare Representative</td>
      <td>1</td>
      <td>9</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sales</td>
      <td>Manager</td>
      <td>1</td>
      <td>2</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Research &amp; Development</td>
      <td>Manager</td>
      <td>1</td>
      <td>3</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Research &amp; Development</td>
      <td>Research Director</td>
      <td>1</td>
      <td>2</td>
      <td>0.21</td>
    </tr>
  </tbody>
</table>
</div>



There is no significant difference in the proportion of Attrition by department. However, within the scope of Job Role, the highest Attrition percentage was found in Human Resources (19.05%). However, this number seems relatively large due to the small number of team members. The Sales Department seems to have more employees than Human Resources, so the percentage in the Sales department falls and loses to the Human Resources department.

#### BusinessTravel x Attrition


```python
# create dataframe to visualize
df_bt = df.reset_index()
bt = df_bt.groupby(['BusinessTravel','Attrition']).agg({'id':'count'}).reset_index()
bt.rename({'id':'Count'}, axis=1, inplace=True)
bt['total'] = bt.groupby('BusinessTravel')['Count'].transform('sum')
bt['Percentage'] = round(bt['Count'] / bt['total']*100,2)

# visualize
sns.barplot(x='BusinessTravel', y='Percentage', hue='Attrition', data=bt);
```


![png](/img/posts/predicting-attrition/output_39_0.png)


Employees with Travel_Frequently seem to have the highest proportion of Attritions.

#### EducationField x Attrition


```python
# create dataframe to visualize
df_edu = df.reset_index()
bt = df_edu.groupby(['EducationField','Attrition']).agg({'id':'count'}).reset_index()
bt.rename({'id':'Count'}, axis=1, inplace=True)
bt['total'] = bt.groupby('EducationField')['Count'].transform('sum')
bt['Percentage'] = round(bt['Count'] / bt['total']*100,2)

# visualize
plt.figure(figsize=(10,4))
sns.barplot(x='EducationField', y='Percentage', hue='Attrition', data=bt);
```


![png](/img/posts/predicting-attrition/output_42_0.png)


There is no *insightful* Attrition pattern based on the field of education.

#### MaritalStatus x Attrition


```python
# create dataframe to visualize
df_mar = df.reset_index()
bt = df_mar.groupby(['MaritalStatus','Attrition']).agg({'id':'count'}).reset_index()
bt.rename({'id':'Count'}, axis=1, inplace=True)
bt['total'] = bt.groupby('MaritalStatus')['Count'].transform('sum')
bt['Percentage'] = round(bt['Count'] / bt['total']*100,2)

# visualize
plt.figure(figsize=(10,4))
sns.barplot(x='MaritalStatus', y='Percentage', hue='Attrition', data=bt);
```


![png](/img/posts/predicting-attrition/output_45_0.png)


*Single* employees have a higher proportion of Attrition than Divorced or Married employees.

#### Gender x Attrition


```python
# create dataframe to visualize
df_sex = df.reset_index()
bt = df_sex.groupby(['Gender','Attrition']).agg({'id':'count'}).reset_index()
bt.rename({'id':'Count'}, axis=1, inplace=True)
bt['total'] = bt.groupby('Gender')['Count'].transform('sum')
bt['Percentage'] = round(bt['Count'] / bt['total']*100,2)

# visualize
plt.figure(figsize=(10,4))
sns.barplot(x='Gender', y='Percentage', hue='Attrition', data=bt);
```


![png](/img/posts/predicting-attrition/output_48_0.png)


There is no significant relationship between Gender and Attrition.

#### Other variables x Attrition


```python
# calculate pearson correlation between continuous vars with Attrition
df_cont['Attrition'] = df['Attrition']

att_corr = df_cont.corr(method='pearson')['Attrition'].reset_index()
att_corr['abs_att'] = abs(att_corr['Attrition'])
top_corrcont_att = att_corr.sort_values('abs_att', ascending=False).iloc[1:11].reset_index(drop=True)
top_corrcont_att.columns = ['Variables','Correlation with Attrition','Absolute Correlation with Attrition']

df_cont.drop('Attrition', axis=1, inplace=True)
```


```python
# calculate spearman correlation between categorical vars with Attrition
att_corr = df_cat.corr(method='spearman')['Attrition'].reset_index()
att_corr['abs_att'] = abs(att_corr['Attrition'])
top_corrcat_att = att_corr.sort_values('abs_att', ascending=False).iloc[1:11].reset_index(drop=True)
top_corrcat_att.columns = ['Variables','Correlation with Attrition','Absolute Correlation with Attrition']
```


```python
top_corr_att = top_corrcont_att.append(top_corrcat_att, ignore_index=True) \
               .sort_values('Absolute Correlation with Attrition', ascending=False) \
               .reset_index(drop=True)

# visualize relationship between other variabels and Attrition
plt.figure(figsize=(20,5))
sns.barplot('Variables', 'Absolute Correlation with Attrition', data=top_corr_att[:11])
plt.ylim(bottom=0.1)
plt.xticks(rotation=10)
plt.title('Absolute Correlation between Attrition and Other Variables');
```


![png](/img/posts/predicting-attrition/output_53_0.png)



```python
top_corr_att.head()
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
      <th>Variables</th>
      <th>Correlation with Attrition</th>
      <th>Absolute Correlation with Attrition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>OverTime</td>
      <td>0.246118</td>
      <td>0.246118</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JobLevel</td>
      <td>-0.190370</td>
      <td>0.190370</td>
    </tr>
    <tr>
      <th>2</th>
      <td>StockOptionLevel</td>
      <td>-0.172296</td>
      <td>0.172296</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TotalWorkingYears</td>
      <td>-0.171063</td>
      <td>0.171063</td>
    </tr>
    <tr>
      <th>4</th>
      <td>YearsInCurrentRole</td>
      <td>-0.160545</td>
      <td>0.160545</td>
    </tr>
  </tbody>
</table>
</div>



The variable that has the strongest relationship with Attrition is OverTime. This relationship is positive, meaning that OverTime employees tend to experience Attrition. However other variables that have a relatively strong relationship with Attrition are actually negative.

In absolute terms, the relationships between these variables and Attrition are all relatively weak (< 0.25).

### How all variables correlate with one another


```python
# draw heatmap to view correlation among all variables
mask = np.triu(np.ones_like(df.corr()))

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot=True, fmt='.1f', mask=mask, cbar=True, cmap="RdYlGn", vmin=-1, vmax=1);
```


![png](/img/posts/predicting-attrition/output_57_0.png)


Several variables that have a strong correlation (>= 0.7) were found between the candidate variables *features*. Here are the details:
- TotalWorkingYears: Age, JobLevel, MonthlyIncome
- YearsWithCurrManager: YearsAtCompany, YearsInCurrentRole
- PerformanceRating: PercentSalaryHike
- YearsAtCompany: YearsInCurrentRole

## Feature Encoding


```python
# one-hot encode
bt_label = ['bt_' + i for i in df['BusinessTravel'].unique()]
df[bt_label] = pd.get_dummies(df['BusinessTravel'], prefix='bt')

dep_label = ['dep_' + i for i in df['Department'].unique()]
df[dep_label] = pd.get_dummies(df['Department'], prefix='dep')

edu_label = ['edu_' + i for i in df['EducationField'].unique()]
df[edu_label] = pd.get_dummies(df['EducationField'], prefix='edu')

jobrole_label = ['role_' + i for i in df['JobRole'].unique()]
df[jobrole_label] = pd.get_dummies(df['JobRole'], prefix='role')

marital_label = ['mar_' + i for i in df['MaritalStatus'].unique()]
df[marital_label] = pd.get_dummies(df['MaritalStatus'], prefix='mar')
```

## Modelling


```python
df_model = df.drop(df.select_dtypes(include=['object']).columns, axis=1)
```


```python
# scale
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_model)
df_scaled = pd.DataFrame(scaled, columns = df_model.columns)
```

### Oversample with SMOTE


```python
target = df['Attrition']

plt.figure(figsize=(4,4))
sns.countplot(target, data=df)
plt.xlabel('target')
plt.title('Comparison between both target groups before SMOTE', size=12);
```


![png](/img/posts/predicting-attrition/output_65_0.png)



```python
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

features = df_scaled.drop('Attrition', axis=1)
target = df_scaled.Attrition

X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, random_state=1)
x_over, y_over = SMOTE().fit_resample(X_train, y_train)

y_over.value_counts()
```




    0.0    997
    1.0    997
    Name: Attrition, dtype: int64




```python
type(y_over)
```




    pandas.core.series.Series



## Modelling with Gradient Boosting


```python
df_model.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1470 entries, 1 to 2068
    Data columns (total 50 columns):
     #   Column                          Non-Null Count  Dtype
    ---  ------                          --------------  -----
     0   Age                             1470 non-null   int64
     1   Attrition                       1470 non-null   int64
     2   DailyRate                       1470 non-null   int64
     3   DistanceFromHome                1470 non-null   int64
     4   Education                       1470 non-null   int64
     5   EnvironmentSatisfaction         1470 non-null   int64
     6   Gender                          1470 non-null   int64
     7   HourlyRate                      1470 non-null   int64
     8   JobInvolvement                  1470 non-null   int64
     9   JobLevel                        1470 non-null   int64
     10  JobSatisfaction                 1470 non-null   int64
     11  MonthlyIncome                   1470 non-null   int64
     12  MonthlyRate                     1470 non-null   int64
     13  NumCompaniesWorked              1470 non-null   int64
     14  OverTime                        1470 non-null   int64
     15  PercentSalaryHike               1470 non-null   int64
     16  PerformanceRating               1470 non-null   int64
     17  RelationshipSatisfaction        1470 non-null   int64
     18  StockOptionLevel                1470 non-null   int64
     19  TotalWorkingYears               1470 non-null   int64
     20  TrainingTimesLastYear           1470 non-null   int64
     21  WorkLifeBalance                 1470 non-null   int64
     22  YearsAtCompany                  1470 non-null   int64
     23  YearsInCurrentRole              1470 non-null   int64
     24  YearsSinceLastPromotion         1470 non-null   int64
     25  YearsWithCurrManager            1470 non-null   int64
     26  bt_Travel_Rarely                1470 non-null   uint8
     27  bt_Travel_Frequently            1470 non-null   uint8
     28  bt_Non-Travel                   1470 non-null   uint8
     29  dep_Sales                       1470 non-null   uint8
     30  dep_Research & Development      1470 non-null   uint8
     31  dep_Human Resources             1470 non-null   uint8
     32  edu_Life Sciences               1470 non-null   uint8
     33  edu_Other                       1470 non-null   uint8
     34  edu_Medical                     1470 non-null   uint8
     35  edu_Marketing                   1470 non-null   uint8
     36  edu_Technical Degree            1470 non-null   uint8
     37  edu_Human Resources             1470 non-null   uint8
     38  role_Sales Executive            1470 non-null   uint8
     39  role_Research Scientist         1470 non-null   uint8
     40  role_Laboratory Technician      1470 non-null   uint8
     41  role_Manufacturing Director     1470 non-null   uint8
     42  role_Healthcare Representative  1470 non-null   uint8
     43  role_Manager                    1470 non-null   uint8
     44  role_Sales Representative       1470 non-null   uint8
     45  role_Research Director          1470 non-null   uint8
     46  role_Human Resources            1470 non-null   uint8
     47  mar_Single                      1470 non-null   uint8
     48  mar_Married                     1470 non-null   uint8
     49  mar_Divorced                    1470 non-null   uint8
    dtypes: int64(26), uint8(24)
    memory usage: 376.8 KB
    


```python
X = df_model.drop(['Attrition'], axis=1)   # features
y = df_model['Attrition'].values           # target

X.shape
```




    (1470, 49)




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)
```


```python
!pip install imblearn
```


```python
from imblearn import under_sampling, over_sampling
X_under, y_under = under_sampling.RandomUnderSampler(0.5).fit_resample(X, y)
X_over, y_over = over_sampling.RandomOverSampler(0.5).fit_resample(X, y)
X_over_SMOTE, y_over_SMOTE = over_sampling.SMOTE().fit_resample(X, y)
```


```python
features = df_model.drop('Attrition', axis=1)
target = df_model.Attrition
```


```python
print(pd.Series(y).value_counts())
print(pd.Series(y_under).value_counts())
print(pd.Series(y_over).value_counts())
print(pd.Series(y_over_SMOTE).value_counts())
```

    0    1233
    1     237
    dtype: int64
    0    474
    1    237
    dtype: int64
    0    1233
    1     616
    dtype: int64
    1    1233
    0    1233
    dtype: int64
    


```python
from sklearn.ensemble import GradientBoostingClassifier
```


```python
models = {"GradientBoosting"  : GradientBoostingClassifier()}
for model_name, clf in models.items():
    clf.fit(X_over, y_over)
    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]

    print("\n")
    print("Evaluate model: {}".format(model_name))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, probs, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("AUC: "+str(round(auc*100,2))+'%')

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("accuracy: "+str(round(accuracy*100,2))+'%')
    
    precission = metrics.precision_score(y_test, y_pred)
    print("precission: "+str(round(precission*100,2))+'%')

    f1score = metrics.f1_score(y_test, y_pred)
    print("f1score: "+str(round(f1score*100,2))+'%')

    recall = metrics.recall_score(y_test, y_pred)
    print("recall: "+str(round(recall*100,2))+'%')

    print("\n")

```

    
    
    Evaluate model: GradientBoosting
    AUC: 97.72%
    accuracy: 95.24%
    precission: 87.84%
    f1score: 86.09%
    recall: 84.42%
    
    
    


```python
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

```


![png](/img/posts/predicting-attrition/output_78_0.png)



```python
model=GradientBoostingClassifier()
 
model.fit(features,df['Attrition'])
 
feature_importances=pd.DataFrame({'features':features.columns,'feature_importance':model.feature_importances_})
print(feature_importances.sort_values('feature_importance',ascending=False))
```

                              features  feature_importance
    10                   MonthlyIncome            0.123387
    13                        OverTime            0.117868
    0                              Age            0.073109
    18               TotalWorkingYears            0.052309
    17                StockOptionLevel            0.048343
    4          EnvironmentSatisfaction            0.048021
    7                   JobInvolvement            0.038296
    2                 DistanceFromHome            0.037761
    1                        DailyRate            0.035264
    8                         JobLevel            0.034536
    44          role_Research Director            0.031590
    9                  JobSatisfaction            0.031030
    24            YearsWithCurrManager            0.029772
    6                       HourlyRate            0.029322
    21                  YearsAtCompany            0.028248
    12              NumCompaniesWorked            0.027963
    11                     MonthlyRate            0.024053
    26            bt_Travel_Frequently            0.021448
    23         YearsSinceLastPromotion            0.021205
    20                 WorkLifeBalance            0.020034
    39      role_Laboratory Technician            0.017559
    16        RelationshipSatisfaction            0.013754
    19           TrainingTimesLastYear            0.013507
    48                    mar_Divorced            0.012659
    43       role_Sales Representative            0.011132
    45            role_Human Resources            0.009126
    36             edu_Human Resources            0.007445
    14               PercentSalaryHike            0.007376
    22              YearsInCurrentRole            0.005520
    33                     edu_Medical            0.005452
    29      dep_Research & Development            0.004433
    25                bt_Travel_Rarely            0.004408
    5                           Gender            0.004039
    3                        Education            0.003773
    34                   edu_Marketing            0.002548
    31               edu_Life Sciences            0.002387
    38         role_Research Scientist            0.000793
    32                       edu_Other            0.000504
    47                     mar_Married            0.000028
    35            edu_Technical Degree            0.000000
    30             dep_Human Resources            0.000000
    37            role_Sales Executive            0.000000
    15               PerformanceRating            0.000000
    40     role_Manufacturing Director            0.000000
    41  role_Healthcare Representative            0.000000
    42                    role_Manager            0.000000
    28                       dep_Sales            0.000000
    27                   bt_Non-Travel            0.000000
    46                      mar_Single            0.000000
    


```python
feature_importances.plot(kind='bar');
```


![png](/img/posts/predicting-attrition/output_80_0.png)



```python
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import itertools

```


```python
def get_confusion_matrix(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    conf = np.zeros((n_classes, n_classes))
    for actual, pred in zip(y_true, y_pred):
        conf[int(actual)][int(pred)] += 1
    return conf.astype('int')
```


```python
conf = get_confusion_matrix(y_test, y_pred)
```


```python
classes = [0, 1]
# plot confusion matrix
plt.imshow(conf, interpolation='nearest', cmap=plt.cm.Greens)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = conf.max() / 2.
for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
    plt.text(j, i, format(conf[i, j], fmt),
             horizontalalignment="center",
             color="white" if conf[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
```




    Text(0.5, 0, 'Predicted label')




![png](/img/posts/predicting-attrition/output_84_1.png)



```python
import shap
```


```python
explainer = shap.Explainer(clf)
shap_values = explainer(X)
```


```python
shap.plots.beeswarm(shap_values)
```


![png](/img/posts/predicting-attrition/output_87_0.png)



```python

```
