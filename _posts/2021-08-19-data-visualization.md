---
layout: post
title: "Data Visualization using Python"
subtitle: "Simple Data Visualization using Python for beginner"
date: 2021-08-17 23:45:13 -0400
background: '/img/posts/data-visualization/bg-1.jpg'
---

# Homework Data Visualization


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# please import as much as you need
```


```python
# read your data

df = pd.read_csv('telco_customer.csv')
df.head()
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
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



# Normal

## Dalam rata-rata, payment method mana yang memiliki monthly charge terbesar per customernya?

```python
# filter data
df_2 = df.groupby('PaymentMethod')['MonthlyCharges'].mean().reset_index()
#df_2.head()

# create a plot
plt.figure(figsize=(10,8))
sns.barplot(x = 'PaymentMethod', y= 'MonthlyCharges', data=df_2);
```


![png](/img/posts/data-visualization/out_1.png)


Jadi, dari plot di atas, dapat disimpulkan bahwa payment method Electronic check memiliki rata-rata monthly charges terbesar daripada payment method lainnya yaitu sebesar 76.25.

## Bagaimana jumlah customer dilihat dari tenure group?

Tenure group:
- low_tenure: User dengan tenure < 21 hari
- medium_tenure: User dengan tenure 21 - 40 hari
- high_tenure: User dengan tenure > 40 hari

```python
# import numpy package
import numpy as np

# mengkategorikan data berdasarkan tenure
df_2 = df.copy()
df_2['tenure_group'] = np.where(df_2['tenure'] > 40, 'High', 
                                             np.where(df_2['tenure'] >= 21, 'Mid', 'Low'))

# menghitung jumlah customer per kategori tenure
df_3 = df_2.groupby('tenure_group')['customerID'].count().reset_index()

#membuat visualisasi
plt.figure(figsize=(10,8))
sns.barplot(x = 'tenure_group', y= 'customerID', data=df_3);
```


![png](/img/posts/data-visualization/out_2.png)


Jadi, dari plot di atas, dapat disimpulkan bahwa jumlah customer tertinggi pada tenure group Low. Sedangkan jumlah customer pada tenure group High tidak berbeda jauh dengan Low, tetapi jumlah customer terendah ada pada tenure grup Mid.

## Apakah kebanyakan dari Senior Citizen berlangganan PhoneService?

```python
# filter data
df.groupby(['SeniorCitizen','PhoneService'])['customerID'].count()
```




    SeniorCitizen  PhoneService
    0              No               578
                   Yes             5323
    1              No               104
                   Yes             1038
    Name: customerID, dtype: int64




```python
# create a plot
sns.countplot(x='SeniorCitizen', data=df, hue='PhoneService');
```


![png](/img/posts/data-visualization/out_3a.png)


Jadi, dari plot di atas, dapat disimpulkan bahwa customer yang bukan senior citizen kebanyakan berlangganan PhoneService, tetapi jika dilihat dari rasionya, customer yang senior citizen maupun bukan memiliki rasio yang hampir sama dalam berlangganan phone service.

## Bagaimana Distribusi dari TotalCharge?

```python
#mengubah tipe data TotalCharge menjadi integer
df["TotalCharges"] = pd.to_numeric( df["TotalCharges"], errors="coerce")
#untuk mengatasi NA
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# mengubah type data object menjadi int dan menyimpannya ke variabel total_charges
total_charges = df["TotalCharges"].astype(int)

#membuat distribution plot
sns.distplot(total_charges, color='red', bins=100);
```


![png](/img/posts/data-visualization/out_3b.png)


Jadi, dari plot di atas, dapat disimpulkan bahwa distribusi data pada kolom TotalCharges memiliki distribusi positively skew. Nilai rata-ratanya lebih besar daripada median, dikarenakan terdapatnya data-data TotalCharges yang besar/outlier (letaknya disebelah kanan dari modus dan median). 
Selain itu, dari grafik diatas dapat diketahui datanya lumayan banyak memiliki nilai 0, dimungkinkan karena mengisi data NaN dengan 0 pada kode diatas. Data TotalCharges semua bernilai positif dengan banyak sebaran data pada rentang 0-2000, tetapi ada yang memiliki nilai yang sangat besar hingga 10000.

# Intermediate

## Apakah customer yang memiliki monthly charges yang tinggi cenderung churn?

```python
# filter data berdasarkan Churn
churn_yes = df[df['Churn'] == 'Yes']
churn_no = df[df['Churn'] == 'No']

#membuat grafik
plt.figure(figsize = (10,7))
sns.distplot(churn_yes['MonthlyCharges'],label='yes', color='#582630')
sns.distplot(churn_no['MonthlyCharges'],label='no', color='#96E29B')
plt.title('Monthly Charges Based On Churn', fontsize = 16)
plt.legend()
plt.show()
```


![png](/img/posts/data-visualization/out_4.png)


Jadi, dari plot di atas, dapat disimpulkan bahwa customer yang memiliki MonthlyCharges tinggi, lebih banyak yang churn daripada yang tidak churn. Selain itu, pada customer yang MonthlyCharges nya lebih rendah, cenderung lebih banyak yang tidak churn daripada yang churn.

## Bagaimana pengaruh memiliki partner & dependents terhadap tingkat churn customer?

```python
# concat partner dan dependants kolom pada dataframe
df['partner_dependents'] = df['Partner'] + df['Dependents']
df['partner_dependents']

# membuat grafik
sns.countplot(x='partner_dependents', data=df, hue='Churn');
```


![png](/img/posts/data-visualization/out_5.png)


Jadi, dari plot di atas, dapat disimpulkan bahwa saat customer memiliki partner & dependents (YesYes), akan memiliki tingkat churn paling rendah (berdasarkan rasio churn yes dan no) dibandingkan dengan kombinasi pemilikan partner & dependents yang lain.

# Soal Hard

## Buatlah satu insight dari data telco customer, dan sertakan storyline pada visualisasi tersebut! dengan hanya memanfaatkan atribut PaymentMethod, CustomerID, dan Churn!

#### Customer Churn Rate based on Payment Method


```python
# filter data
data_payment = df.groupby(['PaymentMethod','Churn']).agg({'customerID' : ['count']}).reset_index()
data_payment.columns = ['PaymentMethod','Churn','Customers']
data_payment
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
      <th>PaymentMethod</th>
      <th>Churn</th>
      <th>Customers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bank transfer (automatic)</td>
      <td>No</td>
      <td>1286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bank transfer (automatic)</td>
      <td>Yes</td>
      <td>258</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Credit card (automatic)</td>
      <td>No</td>
      <td>1290</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Credit card (automatic)</td>
      <td>Yes</td>
      <td>232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Electronic check</td>
      <td>No</td>
      <td>1294</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Electronic check</td>
      <td>Yes</td>
      <td>1071</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Mailed check</td>
      <td>No</td>
      <td>1304</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Mailed check</td>
      <td>Yes</td>
      <td>308</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_payment['TotalCustomers'] = data_payment.groupby(['PaymentMethod'])['Customers'].transform('sum')
data_payment
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
      <th>PaymentMethod</th>
      <th>Churn</th>
      <th>Customers</th>
      <th>TotalCustomers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bank transfer (automatic)</td>
      <td>No</td>
      <td>1286</td>
      <td>1544</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bank transfer (automatic)</td>
      <td>Yes</td>
      <td>258</td>
      <td>1544</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Credit card (automatic)</td>
      <td>No</td>
      <td>1290</td>
      <td>1522</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Credit card (automatic)</td>
      <td>Yes</td>
      <td>232</td>
      <td>1522</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Electronic check</td>
      <td>No</td>
      <td>1294</td>
      <td>2365</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Electronic check</td>
      <td>Yes</td>
      <td>1071</td>
      <td>2365</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Mailed check</td>
      <td>No</td>
      <td>1304</td>
      <td>1612</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Mailed check</td>
      <td>Yes</td>
      <td>308</td>
      <td>1612</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_payment['Percentage'] = (data_payment['Customers']/data_payment['TotalCustomers'])*100
data_payment
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
      <th>PaymentMethod</th>
      <th>Churn</th>
      <th>Customers</th>
      <th>TotalCustomers</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bank transfer (automatic)</td>
      <td>No</td>
      <td>1286</td>
      <td>1544</td>
      <td>83.290155</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bank transfer (automatic)</td>
      <td>Yes</td>
      <td>258</td>
      <td>1544</td>
      <td>16.709845</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Credit card (automatic)</td>
      <td>No</td>
      <td>1290</td>
      <td>1522</td>
      <td>84.756899</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Credit card (automatic)</td>
      <td>Yes</td>
      <td>232</td>
      <td>1522</td>
      <td>15.243101</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Electronic check</td>
      <td>No</td>
      <td>1294</td>
      <td>2365</td>
      <td>54.714588</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Electronic check</td>
      <td>Yes</td>
      <td>1071</td>
      <td>2365</td>
      <td>45.285412</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Mailed check</td>
      <td>No</td>
      <td>1304</td>
      <td>1612</td>
      <td>80.893300</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Mailed check</td>
      <td>Yes</td>
      <td>308</td>
      <td>1612</td>
      <td>19.106700</td>
    </tr>
  </tbody>
</table>
</div>



#### Create Plot


```python
import matplotlib.patches as mpatches

# set the figure size
plt.figure(figsize=(12, 8))

# from raw value to percentage
churn = data_payment[data_payment['Churn'] == 'Yes']

# bar chart 1 -> top bars (group of total customer)
bar1 = sns.barplot(x='PaymentMethod', y='TotalCustomers', data=data_payment, color='#eeddd3')

# bar chart 2 -> bottom bars (group of 'Churn=Yes')
bar2 = sns.barplot(x='PaymentMethod', y='Customers', data=churn, color='#9e2a2b')

# add legend
top_bar = mpatches.Patch(color='#eeddd3', label='Total Customers')
bottom_bar = mpatches.Patch(color='#9e2a2b', label='Churn')
plt.legend(handles=[top_bar, bottom_bar])

# The signature bar
bar1.text(x=-1, y = -270,
    s = ' ©HOMEWORK DATA VISUALIZATION: Firdha Maulidya S',fontsize = 14, 
               color = '#f0f0f0', backgroundcolor = 'grey')

# Generate a bolded horizontal line at y = 0
bar1.axhline(y = 10, color = 'black', linewidth = 1.5, alpha = .7)

# Edit xlabel
bar1.set_xticklabels(labels = ['Bank Transfer', 'Credit Card', 'Electronic Card', 'Mailed Check'])

# Remove the label of the x-axis and y-axis
bar1.xaxis.label.set_visible(False)
bar1.yaxis.label.set_visible(False)

# Adding a title and a subtitle
bar1.text(x = -1, y = 2800, s = "Telco Customer Churn Rate Based On Payment Method",
               fontsize = 26, weight = 'bold', alpha = .75)
bar1.text(x = -1, y = 2650,
               s = 'Electronic card is the most widely used payment method by customers.',
              fontsize = 19, alpha = .85)
bar1.text(x = -1, y = 2520,
               s = 'However, the electronic card payment method has the highest churn rate of 45.3%.',
              fontsize = 19, alpha = .85)

# Add data point
bar1.text(x = 1, y = 260, s = "15.2%", ha='center')
bar1.text(x = 0, y = 275, s = "16.7%", ha='center')
bar1.text(x = 2, y = 1100, s = "45.3%", ha='center')
bar1.text(x = 3, y = 345, s = "19.1%", ha='center')
```




    Text(3, 345, '19.1%')




![png](/img/posts/data-visualization/out_6.png)

