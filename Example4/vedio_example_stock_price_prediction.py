import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import model_selection


dates = []
prices =[]

def get_data(filename):
    with open(filename,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append((row[0]))
            prices.append(float(row[1]))
    return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree= 2)
    svr_rbf = SVR(kernel='rbf',C=1e3, gamma=0.2)

    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    predict_svr_lin = svr_lin.predict(dates)
    predict_svr_poly = svr_poly.predict(dates)
    predict_svr_rbf = svr_rbf.predict(dates)

    error_lin = predict_svr_lin - prices
    error_poly = predict_svr_poly - prices
    error_rbf = predict_svr_rbf - prices

    plt.figure(1)
    plt.scatter(dates,error_lin)

    plt.figure(2)
    plt.scatter(dates, error_poly)

    plt.figure(3)
    plt.scatter(dates, error_rbf)

    plt.figure(4)
    plt.plot(dates, prices, 'ro-.' ,label='actual price')
    plt.plot(dates, predict_svr_rbf, 'b--',label = 'RBF model')
    ## the following prediction failed
    # plt.plot(dates, predict_svr_poly, color='green', label='polynomial model')
    # plt.plot(dates, predict_svr_lin, color='blue', label='linear model')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('support vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_poly.predict(x)[0], svr_lin.predict(x)[0]

get_data('Apple_last_3months_stock_price.csv')

predict_prices(dates, prices, 29)
