import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
advertising=pd.read_csv('Advertising.csv')
train,test=train_test_split(advertising,test_size=0.2)#splitting data into training and test data
train_label=train['sales']
test_label=test['sales']
#train_eda=copy.deepcopy(train)
train=train.drop(columns=['sales'])
test=test.drop(columns=['sales'])
clf1=LinearRegression()
clf1.fit(train,train_label) #fitting Linear regression without EDA
#train_eda=train.dropna(how='any',axis=0) #removing null data
answer=clf1.predict(test)
mean_squared_error = mean_squared_error(test_label, answer)
r2_score = r2_score(test_label,answer)
print("mean squared error before applying EDA is :",mean_squared_error)
print("R2 score before applying EDA is :",r2_score)