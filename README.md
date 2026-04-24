# Platform Shipment Performance

### Introduction

Online retail data set is a data set taken from Machine learning Repository website (http://archive.ics.uci.edu/ml/datasets/online+retail).The context of the data set is to predict customer segmentation. With the help of this data we will learn about some of the basic marketing analytical skills. We will create our own RFM model (Recency, frequency, monetary value), perform K- Mean clustering and make prediction about customer loyalty. 	

### Details about Data Set:

The data source comprises of data set the details of which are as follows;

InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product
Description: Product (item) name. Nominal.
Quantity: The quantities of each product (item) per transaction. Numeric.
InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
UnitPrice: Unit price. Numeric, Product price per unit in sterling.
CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
Country: Country name. Nominal, the name of the country where each customer resides.

#### Project Description

In this project we will be performing tasks that are normally performed in while conducting marketing analytical research. Marketing concepts such as customer segmentation, calculating the Recency, frequency, monetary value in order to identify customer loyalty. We will use RFM model in order to calculate the recency, frequency, monetary value and Customer Segmentation.
```python
#Import all the required libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set()
```

# Data Exploration


```python
# Read 'Data.csv' into a DataFrame named data
data = pd.read_csv('C:/Users/sabih/OneDrive/Desktop/OnlineRetail.csv', encoding = 'unicode_escape' )

#(A parameter unicode_escape has been used to avoid any error that might have occured due to certaininvalid character)


# Examine the head of the DataFrame
data.head()

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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>12/1/2010 8:26</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>12/1/2010 8:26</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>12/1/2010 8:26</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536365</td>
      <td>84029G</td>
      <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
      <td>6</td>
      <td>12/1/2010 8:26</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536365</td>
      <td>84029E</td>
      <td>RED WOOLLY HOTTIE WHITE HEART.</td>
      <td>6</td>
      <td>12/1/2010 8:26</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check the top ten countries in the dataset with highest transactions
data.Country.value_counts(normalize=True).head(10).mul(100).round(1).astype(str) + '%'
```




    United Kingdom    91.4%
    Germany            1.8%
    France             1.6%
    EIRE               1.5%
    Spain              0.5%
    Netherlands        0.4%
    Belgium            0.4%
    Switzerland        0.4%
    Portugal           0.3%
    Australia          0.2%
    Name: Country, dtype: object



#### 90% of records belong to the sales are from United Kingdom 


```python
# Examine the shape of the DataFrame
print(data.shape)
```

    (541909, 8)
    


```python
# Print the data types of dataset
print(data.dtypes)
```

    InvoiceNo       object
    StockCode       object
    Description     object
    Quantity         int64
    InvoiceDate     object
    UnitPrice      float64
    CustomerID     float64
    Country         object
    dtype: object
    

##### Since we see repitation in the country column as each customer ID and description has been linked to their respective country hence we will call out customer distribution in terms of country 


```python
# Dropping the duplicate values
country_data=data[['Country','CustomerID']].drop_duplicates()

#Customer count in respect to their specific country
country_data.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)
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
      <th>Country</th>
      <th>CustomerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>United Kingdom</td>
      <td>3950</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Germany</td>
      <td>95</td>
    </tr>
    <tr>
      <th>13</th>
      <td>France</td>
      <td>87</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Spain</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Belgium</td>
      <td>25</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Switzerland</td>
      <td>21</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Portugal</td>
      <td>19</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Italy</td>
      <td>15</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Finland</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Austria</td>
      <td>11</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Norway</td>
      <td>10</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Netherlands</td>
      <td>9</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Channel Islands</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Denmark</td>
      <td>9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cyprus</td>
      <td>8</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Sweden</td>
      <td>8</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Japan</td>
      <td>8</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Poland</td>
      <td>6</td>
    </tr>
    <tr>
      <th>34</th>
      <td>USA</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Canada</td>
      <td>4</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Unspecified</td>
      <td>4</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Israel</td>
      <td>4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Greece</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>EIRE</td>
      <td>3</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Malta</td>
      <td>2</td>
    </tr>
    <tr>
      <th>35</th>
      <td>United Arab Emirates</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bahrain</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lithuania</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Czech Republic</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Lebanon</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>RSA</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Saudi Arabia</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Singapore</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Iceland</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>European Community</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Hong Kong</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Since we have a large majority of data from UK hence we will focus on the UK data and drop the remaining countries data


```python
#Customers only from United Kingdom
data = data.query("Country=='United Kingdom'").reset_index(drop=True)

#Check for missing values in the dataset
data.isnull().sum(axis=0)

print(data)
```

           InvoiceNo StockCode                          Description  Quantity  \
    0         536365    85123A   WHITE HANGING HEART T-LIGHT HOLDER         6   
    1         536365     71053                  WHITE METAL LANTERN         6   
    2         536365    84406B       CREAM CUPID HEARTS COAT HANGER         8   
    3         536365    84029G  KNITTED UNION FLAG HOT WATER BOTTLE         6   
    4         536365    84029E       RED WOOLLY HOTTIE WHITE HEART.         6   
    ...          ...       ...                                  ...       ...   
    495473    581585     22466       FAIRY TALE COTTAGE NIGHT LIGHT        12   
    495474    581586     22061  LARGE CAKE STAND  HANGING STRAWBERY         8   
    495475    581586     23275     SET OF 3 HANGING OWLS OLLIE BEAK        24   
    495476    581586     21217        RED RETROSPOT ROUND CAKE TINS        24   
    495477    581586     20685                DOORMAT RED RETROSPOT        10   
    
                InvoiceDate  UnitPrice  CustomerID         Country  
    0        12/1/2010 8:26       2.55     17850.0  United Kingdom  
    1        12/1/2010 8:26       3.39     17850.0  United Kingdom  
    2        12/1/2010 8:26       2.75     17850.0  United Kingdom  
    3        12/1/2010 8:26       3.39     17850.0  United Kingdom  
    4        12/1/2010 8:26       3.39     17850.0  United Kingdom  
    ...                 ...        ...         ...             ...  
    495473  12/9/2011 12:31       1.95     15804.0  United Kingdom  
    495474  12/9/2011 12:49       2.95     13113.0  United Kingdom  
    495475  12/9/2011 12:49       1.25     13113.0  United Kingdom  
    495476  12/9/2011 12:49       8.95     13113.0  United Kingdom  
    495477  12/9/2011 12:49       7.08     13113.0  United Kingdom  
    
    [495478 rows x 8 columns]
    


```python
#Removing the missing values from CustomerID column
data = data[pd.notnull(data['CustomerID'])]


#Checking whether the Quantity column has a negative value in it
data.Quantity.min()
```




    -80995




```python
data.UnitPrice.min()
```




    0.0




```python
#Checking whether the Quantity column has a negative value in it
data=data[(data['Quantity']>0) & (data['UnitPrice']>0)]
data.describe() 
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
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>354321.000000</td>
      <td>354321.000000</td>
      <td>354321.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.013795</td>
      <td>2.963994</td>
      <td>15552.486392</td>
    </tr>
    <tr>
      <th>std</th>
      <td>189.267956</td>
      <td>17.862655</td>
      <td>1594.527150</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.001000</td>
      <td>12346.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>1.250000</td>
      <td>14194.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>1.950000</td>
      <td>15522.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>12.000000</td>
      <td>3.750000</td>
      <td>16931.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80995.000000</td>
      <td>8142.750000</td>
      <td>18287.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the data types of dataset
print(data.dtypes)
```

    InvoiceNo       object
    StockCode       object
    Description     object
    Quantity         int64
    InvoiceDate     object
    UnitPrice      float64
    CustomerID     float64
    Country         object
    dtype: object
    


```python
#Convert the InvoiceDate to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
```


```python
#Adding a total Sales column in the data set 
data['Total_Sales'] = data['Quantity'] * data['UnitPrice']
data.shape
```




    (354321, 9)




```python
# Now the overall data set looks somethng like this
data.head()
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>Total_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>15.30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>20.34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>22.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536365</td>
      <td>84029G</td>
      <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>20.34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536365</td>
      <td>84029E</td>
      <td>RED WOOLLY HOTTIE WHITE HEART.</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>20.34</td>
    </tr>
  </tbody>
</table>
</div>




```python
import datetime as dt

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['Month'] = data['InvoiceDate'].dt.month

```


```python
# Express Monthly Sales
monthly_sales = data[['Month', 'Quantity']].groupby('Month').sum()
monthly_sales
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
      <th>Quantity</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>278251</td>
    </tr>
    <tr>
      <th>2</th>
      <td>213375</td>
    </tr>
    <tr>
      <th>3</th>
      <td>276304</td>
    </tr>
    <tr>
      <th>4</th>
      <td>260448</td>
    </tr>
    <tr>
      <th>5</th>
      <td>301824</td>
    </tr>
    <tr>
      <th>6</th>
      <td>280974</td>
    </tr>
    <tr>
      <th>7</th>
      <td>303601</td>
    </tr>
    <tr>
      <th>8</th>
      <td>310831</td>
    </tr>
    <tr>
      <th>9</th>
      <td>454559</td>
    </tr>
    <tr>
      <th>10</th>
      <td>476984</td>
    </tr>
    <tr>
      <th>11</th>
      <td>571215</td>
    </tr>
    <tr>
      <th>12</th>
      <td>528374</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Bar plot
plt.figure(figsize=(8,8))
plt.title("Monthly Sales", fontsize=20)
sns.barplot(monthly_sales.index, monthly_sales['Quantity'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x135e4804400>




![png](output_20_1.png)


#### Majority of Sales occur in betwene the month of September and December Saint Nicholas Day (Christian) ,Fiesta of Our Lady of Guadalupe (Mexican), St. Lucia Day (Swedish), Hanukkah (Jewish), Christmas Day (Christian), Three Kings Day/Epiphany (Christian)




```python
data['Hour'] = data['InvoiceDate'].dt.hour
hourly_sales = data[['Hour', 'Quantity']].groupby('Hour').sum()

# Bar plot
plt.figure(figsize=(8,8))
plt.title("Hourly Sales", fontsize=15)
sns.barplot(hourly_sales.index, hourly_sales['Quantity'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x135e32f94e0>




![png](output_22_1.png)


#### A large amonut of the sales occurs in between 10am to 3pm


```python
# Visualizing the top selling products

top_products = data['Description'].value_counts()[:20]
plt.figure(figsize=(10,6))
sns.set_context("paper", font_scale=1.5)
sns.barplot(y = top_products.index,
            x = top_products.values)
plt.title("Top selling products")
plt.show();
```


![png](output_24_0.png)


# RFM Modeling

#### Behavioral customer segmnetation based on three metrics RECENCY, FREQUENCY & MONETARY VALUE


```python
# For recency will check what was the last date of transaction
#First will convert the InvoiceDate as date variable
data['InvoiceDate'].max()
```




    Timestamp('2011-12-09 12:49:00')




```python
#RFM factors calculation:
Latest_date = dt.datetime(2011,12,10)
RFM_data = data.groupby('CustomerID').agg({'InvoiceDate' : lambda x :(Latest_date - x.max()).days, 'InvoiceNo' : 'count','Total_Sales' : 'sum'}).reset_index()

#converting the names of the columns
RFM_data.rename(columns = {'InvoiceDate' : 'Recency',
                          'InvoiceNo' : "Frequency",
                          'Total_Sales' : "Monetary"},inplace = True)
RFM_data.head()
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
      <th>CustomerID</th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12346.0</td>
      <td>325</td>
      <td>1</td>
      <td>77183.60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12747.0</td>
      <td>2</td>
      <td>103</td>
      <td>4196.01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12748.0</td>
      <td>0</td>
      <td>4595</td>
      <td>33719.73</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12749.0</td>
      <td>3</td>
      <td>199</td>
      <td>4090.88</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12820.0</td>
      <td>3</td>
      <td>59</td>
      <td>942.34</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Summarized RFM table
RFM_data.iloc[:,1:4].describe()
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
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3920.000000</td>
      <td>3920.000000</td>
      <td>3920.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>91.742092</td>
      <td>90.388010</td>
      <td>1864.385601</td>
    </tr>
    <tr>
      <th>std</th>
      <td>99.533485</td>
      <td>217.808385</td>
      <td>7482.817477</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.750000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.000000</td>
      <td>17.000000</td>
      <td>300.280000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>41.000000</td>
      <td>652.280000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>142.000000</td>
      <td>99.250000</td>
      <td>1576.585000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>373.000000</td>
      <td>7847.000000</td>
      <td>259657.300000</td>
    </tr>
  </tbody>
</table>
</div>



#### We observe from the above results that  average recency of the customers is almost 92 days, an average customer purchases a product 90 times and spending an average of 1863.91 unitprice.


```python
#Recency distribution plot
x = RFM_data['Recency']
ax = sns.distplot(x)
```


![png](output_31_0.png)



```python
#Frequency distribution plot, taking observations which have frequency less than 1000
x = RFM_data.query('Frequency < 1000')['Frequency']
ax = sns.distplot(x)
```


![png](output_32_0.png)



```python
#Monateray distribution plot, taking observations which have monetary value less than 10000
x = RFM_data.query('Monetary < 10000')['Monetary']
ax = sns.distplot(x)
```


![png](output_33_0.png)


# Customer Segmentation

#### Market segmentation is the activity of dividing a broad consumer or business market, normally consisting of existing and potential customers, into sub-groups of consumers based on some type of shared characteristics


```python
quantiles = RFM_data.drop('CustomerID',axis = 1).quantile(q = [0.25,0.5,0.75])
quantiles.to_dict()
```




    {'Recency': {0.25: 17.0, 0.5: 50.0, 0.75: 142.0},
     'Frequency': {0.25: 17.0, 0.5: 41.0, 0.75: 99.25},
     'Monetary': {0.25: 300.28000000000003,
      0.5: 652.2800000000002,
      0.75: 1576.5850000000005}}




```python
#Creating table with R,F and M scoring
#Recency scoring 
def R_score(var,p,d):
    if var <= d[p][0.25]:
        return 1
    elif var <= d[p][0.50]:
        return 2
    elif var <= d[p][0.75]:
        return 3
    else:
        return 4
#Frequency and Monetary 
def FM_score(var,p,d):
    if var <= d[p][0.25]:
        return 4
    elif var <= d[p][0.50]:
        return 3
    elif var <= d[p][0.75]:
        return 2
    else:
        return 1

#Scoring:
RFM_data['R_score'] = RFM_data['Recency'].apply(R_score,args = ('Recency',quantiles,))
RFM_data['F_score'] = RFM_data['Frequency'].apply(FM_score,args = ('Frequency',quantiles,))
RFM_data['M_score'] = RFM_data['Monetary'].apply(FM_score,args = ('Monetary',quantiles,))
RFM_data.head()
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
      <th>CustomerID</th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
      <th>R_score</th>
      <th>F_score</th>
      <th>M_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12346.0</td>
      <td>325</td>
      <td>1</td>
      <td>77183.60</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12747.0</td>
      <td>2</td>
      <td>103</td>
      <td>4196.01</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12748.0</td>
      <td>0</td>
      <td>4595</td>
      <td>33719.73</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12749.0</td>
      <td>3</td>
      <td>199</td>
      <td>4090.88</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12820.0</td>
      <td>3</td>
      <td>59</td>
      <td>942.34</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Now we will create : RFMGroup and RFMScore
RFM_data['RFM_Group'] = RFM_data['R_score'].astype(str) + RFM_data['F_score'].astype(str) + RFM_data['M_score'].astype(str)

#Score
RFM_data['RFM_Score'] = RFM_data[['R_score','F_score','M_score']].sum(axis = 1)
RFM_data.head()
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
      <th>CustomerID</th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
      <th>R_score</th>
      <th>F_score</th>
      <th>M_score</th>
      <th>RFM_Group</th>
      <th>RFM_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12346.0</td>
      <td>325</td>
      <td>1</td>
      <td>77183.60</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>441</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12747.0</td>
      <td>2</td>
      <td>103</td>
      <td>4196.01</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12748.0</td>
      <td>0</td>
      <td>4595</td>
      <td>33719.73</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12749.0</td>
      <td>3</td>
      <td>199</td>
      <td>4090.88</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12820.0</td>
      <td>3</td>
      <td>59</td>
      <td>942.34</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>122</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Customer segmnetation
Customer_loyality = ['Tier_1','Tier_2','Tier_3','Tier_4']
C = pd.qcut(RFM_data['RFM_Score'],q = 4,labels=Customer_loyality)
RFM_data['RFM_Customer_loyality'] = C.values
RFM_data.tail(15)
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
      <th>CustomerID</th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
      <th>R_score</th>
      <th>F_score</th>
      <th>M_score</th>
      <th>RFM_Group</th>
      <th>RFM_Score</th>
      <th>RFM_Customer_loyality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3905</th>
      <td>18265.0</td>
      <td>72</td>
      <td>46</td>
      <td>801.51</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>322</td>
      <td>7</td>
      <td>Tier_2</td>
    </tr>
    <tr>
      <th>3906</th>
      <td>18268.0</td>
      <td>134</td>
      <td>1</td>
      <td>25.50</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>344</td>
      <td>11</td>
      <td>Tier_4</td>
    </tr>
    <tr>
      <th>3907</th>
      <td>18269.0</td>
      <td>366</td>
      <td>7</td>
      <td>168.60</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>444</td>
      <td>12</td>
      <td>Tier_4</td>
    </tr>
    <tr>
      <th>3908</th>
      <td>18270.0</td>
      <td>38</td>
      <td>11</td>
      <td>283.15</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>244</td>
      <td>10</td>
      <td>Tier_3</td>
    </tr>
    <tr>
      <th>3909</th>
      <td>18272.0</td>
      <td>2</td>
      <td>166</td>
      <td>3078.58</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
      <td>3</td>
      <td>Tier_1</td>
    </tr>
    <tr>
      <th>3910</th>
      <td>18273.0</td>
      <td>2</td>
      <td>3</td>
      <td>204.00</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>144</td>
      <td>9</td>
      <td>Tier_3</td>
    </tr>
    <tr>
      <th>3911</th>
      <td>18274.0</td>
      <td>30</td>
      <td>11</td>
      <td>175.92</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>244</td>
      <td>10</td>
      <td>Tier_3</td>
    </tr>
    <tr>
      <th>3912</th>
      <td>18276.0</td>
      <td>43</td>
      <td>14</td>
      <td>335.86</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>243</td>
      <td>9</td>
      <td>Tier_3</td>
    </tr>
    <tr>
      <th>3913</th>
      <td>18277.0</td>
      <td>58</td>
      <td>8</td>
      <td>110.38</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>344</td>
      <td>11</td>
      <td>Tier_4</td>
    </tr>
    <tr>
      <th>3914</th>
      <td>18278.0</td>
      <td>73</td>
      <td>9</td>
      <td>173.90</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>344</td>
      <td>11</td>
      <td>Tier_4</td>
    </tr>
    <tr>
      <th>3915</th>
      <td>18280.0</td>
      <td>277</td>
      <td>10</td>
      <td>180.60</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>444</td>
      <td>12</td>
      <td>Tier_4</td>
    </tr>
    <tr>
      <th>3916</th>
      <td>18281.0</td>
      <td>180</td>
      <td>7</td>
      <td>80.82</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>444</td>
      <td>12</td>
      <td>Tier_4</td>
    </tr>
    <tr>
      <th>3917</th>
      <td>18282.0</td>
      <td>7</td>
      <td>12</td>
      <td>178.05</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>144</td>
      <td>9</td>
      <td>Tier_3</td>
    </tr>
    <tr>
      <th>3918</th>
      <td>18283.0</td>
      <td>3</td>
      <td>756</td>
      <td>2094.88</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
      <td>3</td>
      <td>Tier_1</td>
    </tr>
    <tr>
      <th>3919</th>
      <td>18287.0</td>
      <td>42</td>
      <td>70</td>
      <td>1837.28</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>221</td>
      <td>5</td>
      <td>Tier_1</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```
