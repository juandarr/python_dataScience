from sklearn import tree,svm    
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
#We are gonna use a decision tree here

#[height, weight, shoe size]

X = np.array([[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,55,37],[171,75,42],[181,85,43]])

Y = ['male','female','female','female','male','male','male','female','male','female','male']

#clf = tree.DecisionTreeClassifier()

#clf = svm.SVC()
#clf = SGDClassifier(loss='hinge', penalty='l2')
clf = GaussianNB()

output = clf.fit(X,Y)


prediction = output.predict([[190,70,43]])

print(prediction)

#Challenge 
#1. Use any 3 SciKit-Learn Models on this dataset

#2. Compare results

#3. Print the best one

