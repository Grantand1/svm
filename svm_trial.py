#Andrew  Grant
#Big Data Analytics project
#Movie Success Predictor using SVM

import pandas as pd

#Data loading and pre-processing
data = pd.read_csv("tmdb_5000_movies1.csv")
#print(data.head(2))
data = data.drop(['homepage', 'keywords','original_language','original_title','overview','production_companies',
           'production_countries','release_date', 'spoken_languages','tagline','status','genres','id','title'],
                 axis= 1)
#create a classifier column by subtracting budget from the revenue
data['success'] = data.revenue - data.budget
data = data[data.success != 0]
bins = [-100000000000000000000000000,0,1000000000000000000000000000000000000000000]
group_names = [0, 1]
data['categories'] = pd.cut(data['success'], bins, labels=group_names)
data = data.dropna()

X = data.drop('categories', axis = 1)
y = data.categories

# print(pd.value_counts(y))

#split data into test and train in-order to train the model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = .5)

#Apply the svm machine learning with the rbf kernel
from sklearn.svm import SVC
sv_classifier = SVC(kernel= 'rbf', random_state= 0,gamma = .001)
sv_classifier.fit(X_train,y_train)

y_pred = sv_classifier.predict(X_test)

#calculate the accaracy of the vector machine
from sklearn.metrics import accuracy_score
print('Accuracy:')
print(accuracy_score(y_test, y_pred))
# print('Accuracy Normalized:')
# print(accuracy_score(y_test, y_pred, normalize= False ))

from pandas import DataFrame
from sklearn.cross_validation import cross_val_score
cv_score = cross_val_score(estimator= sv_classifier, X =X, y = y ,cv= 10)
print((DataFrame(cv_score, columns= ['CV']).describe()).T)



