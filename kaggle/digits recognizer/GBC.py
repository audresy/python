import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.cross_validation import KFold

## load data
df = pd.read_csv('train.csv', parse_dates = True)
target, train = df['label'], df.ix[:, 'pixel0':'pixel783']
X, y = train.copy(), target.copy()
## split train and test 
X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                      train_size=0.8, 
                                                      test_size=0.2,
                                                      random_state=0)
## select one training model
my_model = OneVsRestClassifier(GradientBoostingClassifier(n_estimators = 100,
                                                          learning_rate = 0.1,
                                                          max_depth = 1,
                                                          random_state=0))

my_model = my_model.fit(X_train, y_train)
y_predict = my_model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_predict)
print (accuracy)