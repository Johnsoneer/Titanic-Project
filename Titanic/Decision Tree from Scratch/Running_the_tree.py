import pandas as pd
from Decision_Tree import Tree
from sklearn.cross_validation import train_test_split
import numpy as np

df = pd.read_csv('train.csv')
df = df[np.isfinite(df['Age'])] #cleaning up the Nan values here
y = df.pop('Survived')
X = df[['Age','Fare']]

#splitting up the data for some cross-validation using train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y)


X_train = np.array(X_train[:200])
y_train = np.array(y_train[:200])
X_test = np.array(X_test[:100])
y_test = np.array(y_test[:100])
tree = Tree()
tree.fit(X_train,y_train,df.columns)

predicted_y = tree.predict(X_test)
ans = np.logical_and(
    np.logical_and(predicted_y != 0, y_test != 0),
    predicted_y == y_test )
score = np.sum(ans)/float(len(ans))

print predicted_y
print y_test
print 'the accuracy of my predictions is: ', score
