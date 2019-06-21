import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import  accuracy_score

## load data
df = pd.read_csv('train.csv', parse_dates = True)
target, train = df['label'], df.ix[:, 'pixel0':'pixel783']
X, y = train.copy(), target.copy()
df_test = pd.read_csv('test.csv', parse_dates = True)
X_test = df_test
'''
## plot data in histogram
plt.figure(figsize = (5, 5))
sns.distplot(target)

## plot data into digits
nr, nc = 4, 4
num = nr*nc
fig, axs = plt.subplots(nr, nc)
images = []
for i in range(nr):
    for j in range(nc):
        num = num - 1
        pic = df.iloc[num, 1:785].values
        pic = np.reshape(pic, (28, 28))
        images.append(axs[i, j].imshow(pic))        
plt.show()
'''
## normalize data
X = X/255
X_test = X_test/255
## reduce dimensions to 200 by PCA
pca = decomposition.PCA(n_components=200)
pca.fit(X)
#plt.plot(pca.explained_variance_ratio_)
#plt.ylabel('% of variance explained')
pca = decomposition.PCA(n_components=50)
pca.fit(X)
PCtrain = pd.DataFrame(pca.transform(X))
PCtest = pd.DataFrame(pca.transform(X_test))
## split train and test data
X_train, X_valid, y_train, y_valid = train_test_split(PCtrain, y, 
                                                      train_size = 0.8, 
                                                      random_state = 1)
#print (X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
## using NN Classifier 

clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5,
                    hidden_layer_sizes = (3500,), random_state = 1)
clf.fit(X_train, y_train)
#predicted = clf.predict(X_valid)
#print (accuracy_score(y_valid, predicted))
#print("Classification report for classifier %s:\n%s\n" 
#      % (clf, metrics.classification_report(y_valid, predicted)))
output = pd.DataFrame(clf.predict(PCtest), columns =['Label'])
output.reset_index(inplace=True)
output.rename(columns={'index': 'ImageId'}, inplace=True)
output['ImageId']=output['ImageId']+1
output.to_csv('output.csv', index=False)
