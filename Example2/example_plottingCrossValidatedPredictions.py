from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

lr = linear_model.LinearRegression()
boston = datasets.load_boston()

y = boston.target
# print(y.shape)

error = np.zeros(y.size)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, boston.data, y, cv=10)
# print(dir(predicted))
for i in range(predicted.size):
    error[i]=y[i]-predicted[i]
    # print("y=%5.3f, p=%5.3f, e=%7.3f"%(y[i], predicted[i],error))

index = np.arange(predicted.size)
# print(index)

## figure 1
# fig1 = plt.figure(1)
plt.subplot(3,1,1)
plt.scatter(y,predicted)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'g--', lw=4)
plt.title('target and predicted scatter graph')

## figure 2
# fig2 = plt.figure(2)
plt.subplot(3,1,2)
plt.plot(y, 'g--')
plt.plot(predicted, 'r-.')
plt.title('predicted values and errors')

plt.subplot(3,1,3)
plt.plot(error, 'b-.')
plt.title('error')

plt.show()