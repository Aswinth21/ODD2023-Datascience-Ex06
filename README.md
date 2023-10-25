### Datascience-Ex06 FEATURE TRANSFORMATION
## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.

## Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## Algorithm:
* Step1: Read the given Data.
* Step2: Clean the Data Set using Data Cleaning Process.
* Step3: Apply Feature Transformation techniques to all the features of the data set.
* Step4: Print the transformed features.

## Program:

```
Developed By: ASWINTH T
Register number: 212222230015
```

## Importing libraries and reading csv file:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
df=pd.read_csv("Data_to_Transform.csv")
```

### Basic Information:
```
df.head()
```
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/892bf390-9c21-470d-be34-99a8995f6534)
```
df.info()
df.describe()
```
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/4163412f-c649-4d22-9399-58b3993f739e)

### Before Transformation:
```
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.title("Highly Negative Skew")
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/57c88c2a-e369-4582-937b-f463f069eaa3)
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/a0584aab-07be-49e7-aa19-9069a0e1b2a4)
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/36007502-e942-45ac-b0a4-6fbc2105b482)
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/28135fee-5170-49d4-bebd-e0f8d15b2f28)



### Log Transformation:
```
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/e7576d57-3471-4273-86b2-57c272de5b50)
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/727438e2-b812-491a-97ad-5b200ed952ea)



### Reciprocal Transformation:
```
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/f79305c0-2c41-48fd-842c-5d15c572dc65)

### SquareRoot Transformation:
```
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/edc6de10-88e0-4997-a8fa-ef9dd403c6ee)

### Power Transformation:
```
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/5ecedf87-1a54-422c-a33c-4c3a4627d4f0)
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/684c3da7-1761-4744-86fa-bb2075834bb7)


### Quantile Transormation:
```
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate  Negative Skew")
plt.show()
```
![image](https://github.com/Aswinth21/ODD2023-Datascience-Ex06/assets/120236638/2beb4850-180f-4c66-8b4b-e1836296c8a9)



## Result:
Thus feature transformation is done for the given dataset.
