---
layout: post
title: "Airline Customer Segmentation"
subtitle: "Airline customer value analysis case"
background: '/img/posts/airline-customer-segmentation/airline.jpg'
---

# Airline customer value analysis case

Use airline customer data to classify customers, analyze characteristics of different customer categories, compare customer values ​​of different customer categories, provide personalized services to customer categories of different values, and formulate corresponding marketing strategy.

You can find the dataset in **[here](https://drive.google.com/drive/folders/1v7BjYPybGlhQ9oNiPwgA-1l1uh3Vi3yW)**.

This data contains the basic information, flight information and point information of 62,988 customers. It contains the membership card number, membership time, gender, age, membership card level, and the mileage in the observation window. 44 characteristic attributes such as number and flight time.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

## Load data


```python
raw = pd.read_csv('flight.csv')
```

## Data Pre-Processing


```python
raw.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 62988 entries, 0 to 62987
    Data columns (total 23 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   MEMBER_NO          62988 non-null  int64  
     1   FFP_DATE           62988 non-null  object 
     2   FIRST_FLIGHT_DATE  62988 non-null  object 
     3   GENDER             62985 non-null  object 
     4   FFP_TIER           62988 non-null  int64  
     5   WORK_CITY          60719 non-null  object 
     6   WORK_PROVINCE      59740 non-null  object 
     7   WORK_COUNTRY       62962 non-null  object 
     8   AGE                62568 non-null  float64
     9   LOAD_TIME          62988 non-null  object 
     10  FLIGHT_COUNT       62988 non-null  int64  
     11  BP_SUM             62988 non-null  int64  
     12  SUM_YR_1           62437 non-null  float64
     13  SUM_YR_2           62850 non-null  float64
     14  SEG_KM_SUM         62988 non-null  int64  
     15  LAST_FLIGHT_DATE   62988 non-null  object 
     16  LAST_TO_END        62988 non-null  int64  
     17  AVG_INTERVAL       62988 non-null  float64
     18  MAX_INTERVAL       62988 non-null  int64  
     19  EXCHANGE_COUNT     62988 non-null  int64  
     20  avg_discount       62988 non-null  float64
     21  Points_Sum         62988 non-null  int64  
     22  Point_NotFlight    62988 non-null  int64  
    dtypes: float64(5), int64(10), object(8)
    memory usage: 11.1+ MB
    


```python
# convert date column to datatime format
raw['FFP_DATE'] = pd.to_datetime(raw['FFP_DATE'])
raw['FIRST_FLIGHT_DATE'] = pd.to_datetime(raw['FIRST_FLIGHT_DATE'])
raw['LOAD_TIME'] = pd.to_datetime(raw['LOAD_TIME'])
raw['LAST_FLIGHT_DATE'] = pd.to_datetime(raw['LAST_FLIGHT_DATE'], errors='coerce')
```


```python
overview = [[column, raw[column].dtypes, raw[column].nunique(), raw[column].unique()] for column in raw.columns]
raw_overview = pd.DataFrame(overview, columns = ['cols','dtype','unique values','values'])
raw_overview
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
      <th>cols</th>
      <th>dtype</th>
      <th>unique values</th>
      <th>values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MEMBER_NO</td>
      <td>int64</td>
      <td>62988</td>
      <td>[54993, 28065, 55106, 21189, 39546, 56972, 449...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FFP_DATE</td>
      <td>datetime64[ns]</td>
      <td>3068</td>
      <td>[2006-11-02T00:00:00.000000000, 2007-02-19T00:...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FIRST_FLIGHT_DATE</td>
      <td>datetime64[ns]</td>
      <td>3406</td>
      <td>[2008-12-24T00:00:00.000000000, 2007-08-03T00:...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GENDER</td>
      <td>object</td>
      <td>2</td>
      <td>[Male, Female, nan]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FFP_TIER</td>
      <td>int64</td>
      <td>3</td>
      <td>[6, 5, 4]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>WORK_CITY</td>
      <td>object</td>
      <td>3234</td>
      <td>[., nan, Los Angeles, guiyang, guangzhou, wulu...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>WORK_PROVINCE</td>
      <td>object</td>
      <td>1165</td>
      <td>[beijing, CA, guizhou, guangdong, xinjiang, zh...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>WORK_COUNTRY</td>
      <td>object</td>
      <td>118</td>
      <td>[CN, US, FR, AN, JP, HK, MY, AU, NL, MX, CA, K...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AGE</td>
      <td>float64</td>
      <td>84</td>
      <td>[31.0, 42.0, 40.0, 64.0, 48.0, 46.0, 50.0, 43....</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LOAD_TIME</td>
      <td>datetime64[ns]</td>
      <td>1</td>
      <td>[2014-03-31T00:00:00.000000000]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FLIGHT_COUNT</td>
      <td>int64</td>
      <td>153</td>
      <td>[210, 140, 135, 23, 152, 92, 101, 73, 56, 64, ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BP_SUM</td>
      <td>int64</td>
      <td>23449</td>
      <td>[505308, 362480, 351159, 337314, 273844, 31333...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SUM_YR_1</td>
      <td>float64</td>
      <td>15828</td>
      <td>[239560.0, 171483.0, 163618.0, 116350.0, 12456...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SUM_YR_2</td>
      <td>float64</td>
      <td>16767</td>
      <td>[234188.0, 167434.0, 164982.0, 125500.0, 13070...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SEG_KM_SUM</td>
      <td>int64</td>
      <td>29081</td>
      <td>[580717, 293678, 283712, 281336, 309928, 29458...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LAST_FLIGHT_DATE</td>
      <td>datetime64[ns]</td>
      <td>730</td>
      <td>[2014-03-31T00:00:00.000000000, 2014-03-25T00:...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>LAST_TO_END</td>
      <td>int64</td>
      <td>731</td>
      <td>[1, 7, 11, 97, 5, 79, 3, 6, 15, 22, 67, 2, 65,...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>AVG_INTERVAL</td>
      <td>float64</td>
      <td>10706</td>
      <td>[3.483253589, 5.194244604, 5.298507463, 27.863...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MAX_INTERVAL</td>
      <td>int64</td>
      <td>706</td>
      <td>[18, 17, 73, 47, 52, 28, 45, 94, 95, 42, 112, ...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>EXCHANGE_COUNT</td>
      <td>int64</td>
      <td>28</td>
      <td>[34, 29, 20, 11, 27, 10, 7, 5, 13, 1, 15, 3, 1...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>avg_discount</td>
      <td>float64</td>
      <td>54179</td>
      <td>[0.961639043, 1.25231444, 1.254675516, 1.09086...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Points_Sum</td>
      <td>int64</td>
      <td>25062</td>
      <td>[619760, 415768, 406361, 372204, 338813, 34312...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Point_NotFlight</td>
      <td>int64</td>
      <td>99</td>
      <td>[50, 33, 26, 12, 39, 15, 29, 14, 7, 16, 3, 66,...</td>
    </tr>
  </tbody>
</table>
</div>



## Missing Values


```python
raw.isna().any()
```




    MEMBER_NO            False
    FFP_DATE             False
    FIRST_FLIGHT_DATE    False
    GENDER                True
    FFP_TIER             False
    WORK_CITY             True
    WORK_PROVINCE         True
    WORK_COUNTRY          True
    AGE                   True
    LOAD_TIME            False
    FLIGHT_COUNT         False
    BP_SUM               False
    SUM_YR_1              True
    SUM_YR_2              True
    SEG_KM_SUM           False
    LAST_FLIGHT_DATE      True
    LAST_TO_END          False
    AVG_INTERVAL         False
    MAX_INTERVAL         False
    EXCHANGE_COUNT       False
    avg_discount         False
    Points_Sum           False
    Point_NotFlight      False
    dtype: bool




```python
raw.isnull().sum()
```




    MEMBER_NO               0
    FFP_DATE                0
    FIRST_FLIGHT_DATE       0
    GENDER                  3
    FFP_TIER                0
    WORK_CITY            2269
    WORK_PROVINCE        3248
    WORK_COUNTRY           26
    AGE                   420
    LOAD_TIME               0
    FLIGHT_COUNT            0
    BP_SUM                  0
    SUM_YR_1              551
    SUM_YR_2              138
    SEG_KM_SUM              0
    LAST_FLIGHT_DATE      421
    LAST_TO_END             0
    AVG_INTERVAL            0
    MAX_INTERVAL            0
    EXCHANGE_COUNT          0
    avg_discount            0
    Points_Sum              0
    Point_NotFlight         0
    dtype: int64




```python
# check the proportion of missing value

print(raw[raw['WORK_CITY'].isna() == True].shape[0]/ raw.shape[0])
print(raw[raw['WORK_PROVINCE'].isna() == True].shape[0]/ raw.shape[0])
```

    0.03602273448910904
    0.05156537753222836
    

The two features that have the most missing values, if calculated the percentage of missing values ​​to the total value, are only 3.6% and 5.1%. So, data rows that have missing values ​​will be deleted because they are very few in number.


```python
# drop missing value
raw.drop(raw[raw['GENDER'].isna() == True].index, inplace=True)
raw.drop(raw[raw['WORK_CITY'].isna() == True].index, inplace=True)
raw.drop(raw[raw['WORK_PROVINCE'].isna() == True].index, inplace=True)
raw.drop(raw[raw['WORK_COUNTRY'].isna() == True].index, inplace=True)
raw.drop(raw[raw['AGE'].isna() == True].index, inplace=True)
raw.drop(raw[raw['SUM_YR_1'].isna() == True].index, inplace=True)
raw.drop(raw[raw['SUM_YR_2'].isna() == True].index, inplace=True)
raw.drop(raw[raw['LAST_FLIGHT_DATE'].isna() == True].index, inplace=True)
```


```python
raw.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 57860 entries, 0 to 62986
    Data columns (total 23 columns):
     #   Column             Non-Null Count  Dtype         
    ---  ------             --------------  -----         
     0   MEMBER_NO          57860 non-null  int64         
     1   FFP_DATE           57860 non-null  datetime64[ns]
     2   FIRST_FLIGHT_DATE  57860 non-null  datetime64[ns]
     3   GENDER             57860 non-null  object        
     4   FFP_TIER           57860 non-null  int64         
     5   WORK_CITY          57860 non-null  object        
     6   WORK_PROVINCE      57860 non-null  object        
     7   WORK_COUNTRY       57860 non-null  object        
     8   AGE                57860 non-null  float64       
     9   LOAD_TIME          57860 non-null  datetime64[ns]
     10  FLIGHT_COUNT       57860 non-null  int64         
     11  BP_SUM             57860 non-null  int64         
     12  SUM_YR_1           57860 non-null  float64       
     13  SUM_YR_2           57860 non-null  float64       
     14  SEG_KM_SUM         57860 non-null  int64         
     15  LAST_FLIGHT_DATE   57860 non-null  datetime64[ns]
     16  LAST_TO_END        57860 non-null  int64         
     17  AVG_INTERVAL       57860 non-null  float64       
     18  MAX_INTERVAL       57860 non-null  int64         
     19  EXCHANGE_COUNT     57860 non-null  int64         
     20  avg_discount       57860 non-null  float64       
     21  Points_Sum         57860 non-null  int64         
     22  Point_NotFlight    57860 non-null  int64         
    dtypes: datetime64[ns](4), float64(5), int64(10), object(4)
    memory usage: 10.6+ MB
    


```python
57860/62988
```




    0.9185876674922208



After deleting all missing values ​​in all features, a total of 91.8% of the data was cleared of missing values.

## Duplicated Values


```python
raw.duplicated().sum()
```




    0



There is no duplicated data on this dataset.

## Exploratory Data Analysis


```python
df = raw.copy()
```


```python
df = df.drop('MEMBER_NO', axis=1)
```

### Statistic Description


```python
df.describe()
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
      <th>FFP_TIER</th>
      <th>AGE</th>
      <th>FLIGHT_COUNT</th>
      <th>BP_SUM</th>
      <th>SUM_YR_1</th>
      <th>SUM_YR_2</th>
      <th>SEG_KM_SUM</th>
      <th>LAST_TO_END</th>
      <th>AVG_INTERVAL</th>
      <th>MAX_INTERVAL</th>
      <th>EXCHANGE_COUNT</th>
      <th>avg_discount</th>
      <th>Points_Sum</th>
      <th>Point_NotFlight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>57860.000000</td>
      <td>57860.000000</td>
      <td>57860.000000</td>
      <td>57860.000000</td>
      <td>57860.000000</td>
      <td>57860.000000</td>
      <td>57860.000000</td>
      <td>57860.00000</td>
      <td>57860.000000</td>
      <td>57860.000000</td>
      <td>57860.000000</td>
      <td>57860.000000</td>
      <td>57860.000000</td>
      <td>57860.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.104666</td>
      <td>42.233253</td>
      <td>12.043000</td>
      <td>11047.843726</td>
      <td>5363.816955</td>
      <td>5679.279658</td>
      <td>17324.371863</td>
      <td>172.20598</td>
      <td>67.963638</td>
      <td>167.221673</td>
      <td>0.327981</td>
      <td>0.720626</td>
      <td>12721.368960</td>
      <td>2.819703</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.378206</td>
      <td>9.763364</td>
      <td>14.239523</td>
      <td>16294.179086</td>
      <td>8110.434363</td>
      <td>8714.783954</td>
      <td>20982.734648</td>
      <td>180.80718</td>
      <td>77.533059</td>
      <td>122.901236</td>
      <td>1.149762</td>
      <td>0.183942</td>
      <td>20621.601695</td>
      <td>7.497873</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>368.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>35.000000</td>
      <td>3.000000</td>
      <td>2599.750000</td>
      <td>1020.000000</td>
      <td>833.000000</td>
      <td>4882.000000</td>
      <td>28.00000</td>
      <td>23.666667</td>
      <td>81.000000</td>
      <td>0.000000</td>
      <td>0.612019</td>
      <td>2863.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>41.000000</td>
      <td>7.000000</td>
      <td>5814.000000</td>
      <td>2804.000000</td>
      <td>2830.000000</td>
      <td>10208.000000</td>
      <td>105.00000</td>
      <td>44.812500</td>
      <td>144.000000</td>
      <td>0.000000</td>
      <td>0.711429</td>
      <td>6468.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>48.000000</td>
      <td>15.000000</td>
      <td>12976.250000</td>
      <td>6584.000000</td>
      <td>6931.000000</td>
      <td>21519.000000</td>
      <td>259.25000</td>
      <td>82.000000</td>
      <td>228.000000</td>
      <td>0.000000</td>
      <td>0.808333</td>
      <td>14491.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.000000</td>
      <td>110.000000</td>
      <td>213.000000</td>
      <td>505308.000000</td>
      <td>239560.000000</td>
      <td>234188.000000</td>
      <td>580717.000000</td>
      <td>731.00000</td>
      <td>728.000000</td>
      <td>728.000000</td>
      <td>46.000000</td>
      <td>1.500000</td>
      <td>985572.000000</td>
      <td>140.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Heat map to check feature correlation


```python
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap='Blues', annot=True, fmt='.2f');
```


![png](/img/posts/airline-customer-segmentation/output_27_0.png)


### Numeric Distribution and Outlier


```python
h = df.hist(bins=25,figsize=(25,25),xlabelsize='10',ylabelsize='10',xrot=-15)
sns.despine(left=True, bottom=True)
[x.title.set_size(12) for x in h.ravel()];
[x.yaxis.tick_left() for x in h.ravel()];
```


![png](/img/posts/airline-customer-segmentation/output_29_0.png)


## Feature Selection

Based on the LRFMC model, customer grouping is carried out based on the passenger value LRFMC model, and the characteristics of each customer group are analyzed to identify valuable customers.

There are too many attributes in the original data. According to the LRFMC model of airline customer value, <br/>six attributes related to LRFMC index are selected: **FFP_DATE, LOAD_TIME, FLIGHT_COUNT, AVG_DISCOUNT, SEG_KM_SUM, LAST_TO_END**. 

The data transformations carried out are attributing construction and data standardization. Because the LRFMC feature is not directly contained in the dataset, it needs to be extracted from the original dataset, with the following conditions:
1. L (customer relationship length) = LOAD_TIME-FFP_DATE
2. R (consumption time interval) =LAST_TO_END
3. F (consumption frequency) = FLIGHT_COUNT
4. M (flight mileage) =SEG_KM_SUM
5. C (average value of discount) =AVG_DISCOUNT


```python
df_final = df[['LOAD_TIME', 'FFP_DATE', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount']]
df_final
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
      <th>LOAD_TIME</th>
      <th>FFP_DATE</th>
      <th>LAST_TO_END</th>
      <th>FLIGHT_COUNT</th>
      <th>SEG_KM_SUM</th>
      <th>avg_discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-03-31</td>
      <td>2006-11-02</td>
      <td>1</td>
      <td>210</td>
      <td>580717</td>
      <td>0.961639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-03-31</td>
      <td>2007-02-01</td>
      <td>11</td>
      <td>135</td>
      <td>283712</td>
      <td>1.254676</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-03-31</td>
      <td>2008-08-22</td>
      <td>97</td>
      <td>23</td>
      <td>281336</td>
      <td>1.090870</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-03-31</td>
      <td>2009-04-10</td>
      <td>5</td>
      <td>152</td>
      <td>309928</td>
      <td>0.970658</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2014-03-31</td>
      <td>2008-02-10</td>
      <td>79</td>
      <td>92</td>
      <td>294585</td>
      <td>0.967692</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>62982</th>
      <td>2014-03-31</td>
      <td>2013-01-20</td>
      <td>437</td>
      <td>2</td>
      <td>3848</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>62983</th>
      <td>2014-03-31</td>
      <td>2011-05-20</td>
      <td>297</td>
      <td>2</td>
      <td>1134</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>62984</th>
      <td>2014-03-31</td>
      <td>2010-03-08</td>
      <td>89</td>
      <td>4</td>
      <td>8016</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>62985</th>
      <td>2014-03-31</td>
      <td>2006-03-30</td>
      <td>29</td>
      <td>2</td>
      <td>2594</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>62986</th>
      <td>2014-03-31</td>
      <td>2013-02-06</td>
      <td>400</td>
      <td>2</td>
      <td>3934</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>57860 rows × 6 columns</p>
</div>




```python
L = df_final['LOAD_TIME'] - df_final['FFP_DATE']
lrfmc = pd.DataFrame((L / np.timedelta64(1, 'D')) / 30, columns= ['L'])

lrfmc['R'] = df_final['LAST_TO_END'] / 30

lrfmc['F'] = df_final['FLIGHT_COUNT']

lrfmc['M'] = df_final['SEG_KM_SUM']

lrfmc['C'] = df_final['avg_discount']
lrfmc
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
      <th>L</th>
      <th>R</th>
      <th>F</th>
      <th>M</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90.200000</td>
      <td>0.033333</td>
      <td>210</td>
      <td>580717</td>
      <td>0.961639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>87.166667</td>
      <td>0.366667</td>
      <td>135</td>
      <td>283712</td>
      <td>1.254676</td>
    </tr>
    <tr>
      <th>3</th>
      <td>68.233333</td>
      <td>3.233333</td>
      <td>23</td>
      <td>281336</td>
      <td>1.090870</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60.533333</td>
      <td>0.166667</td>
      <td>152</td>
      <td>309928</td>
      <td>0.970658</td>
    </tr>
    <tr>
      <th>5</th>
      <td>74.700000</td>
      <td>2.633333</td>
      <td>92</td>
      <td>294585</td>
      <td>0.967692</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>62982</th>
      <td>14.500000</td>
      <td>14.566667</td>
      <td>2</td>
      <td>3848</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>62983</th>
      <td>34.866667</td>
      <td>9.900000</td>
      <td>2</td>
      <td>1134</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>62984</th>
      <td>49.466667</td>
      <td>2.966667</td>
      <td>4</td>
      <td>8016</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>62985</th>
      <td>97.433333</td>
      <td>0.966667</td>
      <td>2</td>
      <td>2594</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>62986</th>
      <td>13.933333</td>
      <td>13.333333</td>
      <td>2</td>
      <td>3934</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>57860 rows × 5 columns</p>
</div>



After the data from the five indicators have been extracted and the distribution of the data analyzed on the EDA, these data have a fairly wide range so it is necessary to *standardize* the data.


```python
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(lrfmc)
std_df = pd.DataFrame(data = X_std, columns=[i for i in lrfmc.columns])

std_df
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
      <th>L</th>
      <th>R</th>
      <th>F</th>
      <th>M</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.482461</td>
      <td>-0.946906</td>
      <td>13.902061</td>
      <td>26.850528</td>
      <td>1.310280</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.374135</td>
      <td>-0.891598</td>
      <td>8.634985</td>
      <td>12.695673</td>
      <td>2.903386</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.697994</td>
      <td>-0.415949</td>
      <td>0.769485</td>
      <td>12.582436</td>
      <td>2.012847</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.423014</td>
      <td>-0.924783</td>
      <td>9.828855</td>
      <td>13.945092</td>
      <td>1.359311</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.928929</td>
      <td>-0.515504</td>
      <td>5.615194</td>
      <td>13.213865</td>
      <td>1.343190</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>57855</th>
      <td>-1.220915</td>
      <td>1.464523</td>
      <td>-0.705297</td>
      <td>-0.642266</td>
      <td>-3.917713</td>
    </tr>
    <tr>
      <th>57856</th>
      <td>-0.493586</td>
      <td>0.690211</td>
      <td>-0.705297</td>
      <td>-0.771611</td>
      <td>-3.917713</td>
    </tr>
    <tr>
      <th>57857</th>
      <td>0.027804</td>
      <td>-0.460196</td>
      <td>-0.564841</td>
      <td>-0.443624</td>
      <td>-3.917713</td>
    </tr>
    <tr>
      <th>57858</th>
      <td>1.740775</td>
      <td>-0.792044</td>
      <td>-0.705297</td>
      <td>-0.702030</td>
      <td>-3.917713</td>
    </tr>
    <tr>
      <th>57859</th>
      <td>-1.241151</td>
      <td>1.259884</td>
      <td>-0.705297</td>
      <td>-0.638167</td>
      <td>-3.917713</td>
    </tr>
  </tbody>
</table>
<p>57860 rows × 5 columns</p>
</div>



## Model Construction

### K-Number Tuning Using Elbow Method

The modeling is done using K Means because it considers:
1. Groups are not explicitly labeled in the dataset
2. The number of datasets tends to be large
3. Ease and speed of implementation

In order to determine the optimal K (the number of cluster) for the K-Means Model later, visualization of the elbow method is carried out.


```python
from sklearn.cluster import KMeans
inertia = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(std_df.values)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(20, 10))

sns.lineplot(x=range(1, 10), y=inertia, color='#000087', linewidth = 2);
sns.scatterplot(x=range(1, 10), y=inertia, s=300, color='#800000',  linestyle='--');
```


![png](/img/posts/airline-customer-segmentation/output_39_0.png)


The fault point on the *elbow* graph is at X = 4 so the K value chosen for *clustering* is *4*.

### Perform PCA


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(std_df)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
principalDf
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
      <th>pc1</th>
      <th>pc2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25.772196</td>
      <td>-2.384333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.153543</td>
      <td>0.832360</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.741976</td>
      <td>0.540395</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15.187636</td>
      <td>-1.101030</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.104293</td>
      <td>-0.412134</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>57855</th>
      <td>-2.326677</td>
      <td>-3.331867</td>
    </tr>
    <tr>
      <th>57856</th>
      <td>-1.905904</td>
      <td>-3.263502</td>
    </tr>
    <tr>
      <th>57857</th>
      <td>-1.017829</td>
      <td>-3.439390</td>
    </tr>
    <tr>
      <th>57858</th>
      <td>-0.699953</td>
      <td>-2.844311</td>
    </tr>
    <tr>
      <th>57859</th>
      <td>-2.244897</td>
      <td>-3.398755</td>
    </tr>
  </tbody>
</table>
<p>57860 rows × 2 columns</p>
</div>




```python
# scree plot
plt.figure(figsize=(10,10))
var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
lbls = [str(x) for x in range(1,len(var)+1)]
plt.bar(x=range(1,len(var)+1), height = var, tick_label = lbls)
plt.show()
```


![png](/img/posts/airline-customer-segmentation/output_43_0.png)


PCA with two axes can explain about 70% of the true data variance.

### Clustering by 4


```python
model = KMeans(n_clusters = 4)
label = model.fit_predict(principalComponents)

plt.figure(figsize=(10,10))
uniq = np.unique(label)
for i in uniq:
   plt.scatter(principalComponents[label == i , 0] , principalComponents[label == i , 1] , label = i)

plt.legend()
plt.show()
```


![png](/img/posts/airline-customer-segmentation/output_46_0.png)



```python
result = pd.DataFrame({"Cluster category": ['Customer base 1', 'Customer base 2', 'Customer base 3', 'Customer base 4']})

n = pd.Series(model.labels_).value_counts()
c = pd.DataFrame(model.cluster_centers_)
r = pd.concat([n, c], axis=1)

result = pd.concat([result, r], axis =1)
result.columns = ['Cluster category', 'Data points', 'pc1', 'pc2']
result
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
      <th>Cluster category</th>
      <th>Data points</th>
      <th>pc1</th>
      <th>pc2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Customer base 1</td>
      <td>24038</td>
      <td>-0.763611</td>
      <td>-0.647971</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Customer base 2</td>
      <td>15271</td>
      <td>-0.707220</td>
      <td>1.118463</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Customer base 3</td>
      <td>15390</td>
      <td>1.020182</td>
      <td>-0.094622</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Customer base 4</td>
      <td>3161</td>
      <td>4.293313</td>
      <td>-0.023380</td>
    </tr>
  </tbody>
</table>
</div>



## Conclusions

1. Based on the results of clustering, this flight's customer can be divided into 4 groups
2. Four customer categories based on variations in a number of features, namely:
    - The duration of the customer registered as a member (customer length relationship)
    - Distance from the last flight time of the most recent flight order (consumption time interval)
    - Number of flights followed (consumption frequency)
    - Distance traveled during flight (flight mileage)
    - Average discount obtained (average value of discount)

## Recommendations

1. Passengers with high flight count and long mileage are given a free upgrade class promo after the flight count and mileage is above quantile 3
2. Unspent points can be exchanged for souvenirs or items sold on board (perfume, neck pillows, mugs, pins, or key chains)
