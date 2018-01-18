from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.image as matimg
from sklearn import svm
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle
from sklearn.externals import joblib
from sklearn import random_projection


##1. Machine learning: the problem setting
##2. Loading an example dataset
##3. Learning and predicting
##4. Model persistence
##5. Conventions

iris = datasets.load_iris()
digits = datasets.load_digits()

## save datasets image at local place
# for i in range(np.size(digits.target)):
#     name = './savedImages/digits'+ str(i) + '.jpg'
#     matimg.imsave(name, digits.images[i])

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

## test hardness
seed = 7
scoring = 'accuracy'
cv_value = 10
## result values
results = []
names = []
i = 1
plt.figure(1)

for name, model in models:
    predicted = cross_val_predict(model, digits.data, digits.target, cv=10)
    cv_results = model_selection.cross_val_score(model, digits.data, digits.target, cv=10, scoring='accuracy')
    plt.plot(cv_results, label = name)
    print(cv_results)

plt.legend()
plt.show()


