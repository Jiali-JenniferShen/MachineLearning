import quandl as Quandl
import math, datetime
from sklearn import cross_validation
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVC

df = Quandl.get('WIKI/GOOGL')
# print(df.head())
# print('******************')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] *100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] *100.0
df = df[['Adj. Close','HL_PCT', 'PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.01*len(df)))
# print(forecast_out)

df['lable'] = df[forecast_col].shift(-forecast_out)
# print(df.head())
x = np.array(df.drop(['lable'], 1))
# print("drop:\r\n")
# print(x)
x = preprocessing.scale(x)
# print("processing:\r\n")
# print(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]
# print("lately\r\n")
print(x)

df.dropna(inplace = True)
y = np.array(df['lable'])
# print("np.array:\r\n")
print(y)


x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size = 0.5)

a = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
b = np.array([1, 1, 2, 2])

clf = SVC()
clf.fit(a, b)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
max_iter=-1, probability=False, random_state=None, shrinking=True,
tol=0.001, verbose=False)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.plot(a)
plt.plot(b)
plt.legend(loc=4)

# ax.set_xticklabels(names)
plt.show()
print(clf.predict(x_test))