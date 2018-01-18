import quandl as Quandl
import math, datetime
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
# print(x)

df.dropna(inplace = True)
y = np.array(df['lable'])
# print("np.array:\r\n")
# print(y)


x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size = 0.5)


# svr_lin = SVC(kernel='linear', C=1e3)
# svr_poly = SVC(kernel='poly', C=1e3, degree= 2)
svr_rbf = SVC(kernel='rbf',C=1e3, gamma=0.1)
# svr_lin.fit(x_train, y_train)
# svr_poly.fit(x_train, y_train)
svr_rbf.fit(x_train, y_train)

# predictions_lin = svr_lin.predict(x_test)
# predictions_poly = svr_poly.predict(x_test)
predictions_rbf = svr_rbf.predict(x_test)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.plot(predictions_rbf)
plt.plot(y_test)
plt.legend()

# ax.set_xticklabels(names)
plt.show()
