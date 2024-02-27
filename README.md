<H3>ENTER YOUR NAME: Dharini PV</H3>
<H3>ENTER YOUR REGISTER NO.: 212222240024</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
#Importing libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv('/content/Churn_Modelling.csv')
print(df)
df.head()

#Checking for null values
print(df.isnull().sum())

#Checking for duplicates
df.duplicated()

#Describing the dataset
df.describe()
#Description of CustomerId
print(df['CustomerId'].describe())

#Dropping string value data
df1 = df.drop(['Surname', 'Geography','Gender'], axis=1)
df1.head()

#Scaling
scaler=MinMaxScaler()
df2=pd.DataFrame(scaler.fit_transform(df1))
print(df2)

#Assigning x and y
x=df.iloc[:, :-1].values
print(x)
y=df.iloc[:, -1].values
print(y)

#Splitting into traing and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)print("Length of x_train:",len(x_train))
print(x_test)
print("Length of x_test:",len(x_test))

```
## OUTPUT:
### Initial DataFrame
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/03bb2d0b-75f7-4466-8b12-21761996a8d6)
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/6295a5d2-ecba-47fe-8531-1dcaa5c84c0e)
![Screenshot 2024-02-27 164930](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/d0ce736d-d1b6-4744-94c5-4fb2d466f428)

### Null count
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/86075a81-8bc1-4f5e-a52d-0e9fe65c4ef6)

### Duplicates
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/689faedf-24db-46c4-bf92-fd25b39bf984)

### Describing the dataset
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/df12cd3f-b922-46e0-904f-ede089fa2dfe)

### Description of CustomerId
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/e3134c7e-0cf4-4073-9aa9-416acfa73987)

### Dropping string value data
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/118f2d5a-7c0c-4437-9ff0-78331e0a6cbf)

### Scaling the dataset
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/697da918-78a1-4bb0-ac8b-1099892336d2)

### x values
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/f1748fbe-7419-4358-93da-5ec8c5205a34)

### y values
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/297531d6-e8a9-4fe5-a182-5aff931f766a)

### Splitting into training and testing data
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/fb57a179-d337-41ba-a36e-a22b2989d6fe)
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/df2d6677-5a4e-4ce6-acb8-5987011fd886)
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/54b4a609-21f9-431c-a3f8-b9a914637284)
![image](https://github.com/DHARINIPV/Ex-1-NN/assets/119400845/30027206-0881-443e-9e04-33c827eab748)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


