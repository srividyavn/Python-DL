# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#%matplotlib inline
# Importing the dataset
advertise = pd.read_csv('Advertising.csv')

advertise.describe()

#Next, we'll check for skewness
print ("Skew is:", advertise.sales.skew())
plt.hist(advertise.sales, color='blue')
plt.xlabel('sales', size = 10)
plt.ylabel('freq', size = 10)
plt.title('histogram distribution', size = 6)
plt.show()

target = np.log(advertise.sales)
print ("Skew is:", target.skew())
plt.hist(target, color='green')
plt.xlabel('sales', size = 10)
plt.ylabel('freq', size = 10)
plt.title('logarithmic histogram distribution', size = 6)
plt.show()


#Working with Numeric Features
numeric_features = advertise.select_dtypes(include=[np.number])

##Null values
nulls = pd.DataFrame(advertise.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

##handling missing value
A = advertise.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(A.isnull().sum() != 0))

corr = advertise.corr()

print (corr['sales'].sort_values(ascending=False)[:5], '\n')
print (corr['sales'].sort_values(ascending=False)[-5:])

sns.boxplot(advertise["sales"],orient= "v")
plt.title("sales Outlier detection", size=15)
plt.xlabel("sales", size=15)
plt.ylabel("freq")
plt.show()

advertise.drop(advertise[advertise["sales"] < 25].index)

X = advertise.iloc[:, :-2].values
y = advertise.iloc[:, 4].values

X.shape

X=X[:,1:]


X_train, X_test, Y_train, Y_test= train_test_split(X,y, test_size=0.3, random_state=5)

regressor=LinearRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)


rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
r2=r2_score(Y_test,y_pred)
print("R2 score: {}".format(r2))

plt.scatter(y_pred,Y_test,alpha=0.9,color='r')
plt.show()
