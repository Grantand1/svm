#Andrew  Grant
#Big Data Analytics project
#Movie Success Predictor using SVM

import pandas as pd
import numpy as np



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
#print(data.shape)
X = data.drop('categories', axis = 1)
y = data.categories
#calculate mean vectors
label = [0,1]
mean_vecs = []
for label in range(0, 2):
    mean_vecs.append(np.mean(X[y==label], axis=0))
    #print('MV %s: %s\n' %(label, mean_vecs[label-1])

d = X.shape[1]
S_W = np.zeros((d, d))
for label, mv in zip(range(0, 2), mean_vecs):
    class_scatter = np.cov(X[y==label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
print(S_W)

mean_overall = np.mean(X, axis=0)
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B +=  (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))

# calculating eigen values and corresponding eigen vectors
eigen_values, eigen_vectors = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
# projecting the data set in new space
#sorted in descending order of eigen values and corresponding eigen vector considering only the real parts
sorted_ix= eigen_values.argsort()[::-1]
EigenValues = np.real(eigen_values[sorted_ix])
EigenVectors = np.real(eigen_vectors[:,sorted_ix])
Y = X.dot( EigenVectors)
Y.head()
import matplotlib.pyplot as plt
from ggplot import *
import warnings
warnings.filterwarnings('ignore')
# projection onto new space
LD_component_1 = Y[0]
LD_component_2 = Y[1]
plt.scatter(Y[y==0][0],Y[y==0][1],label='Class 0', c='red' ,s = 50, alpha = .5)
plt.scatter(Y[y==1][0],Y[y==1][1],label='Class 1', c='blue',s = 50, alpha = .5)
plt.title('Visualization data using LDA', fontsize  =15)
plt.legend(loc=0)
fig = plt.gcf()
fig.set_size_inches(9, 7)
plt.show()
