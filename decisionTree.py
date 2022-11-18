import numpy as np
import pandas as pd
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv('/Users/lixinyu/GitRepository/StatisticalMethod/diabetes.csv', index_col=0)
df.head()

X = df.iloc[:, 1:]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

def cv_score(d):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return (tr_score, cv_score)

depths = range(2, 15)
scores = [cv_score(d) for d in depths]
tr_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = depths[best_score_index]
print('best param: {0}; best score: {1}'.format(best_param, best_score))

clf = DecisionTreeClassifier(max_depth=best_param)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
result = cross_val_score(clf, X_train, y_train, cv=10)

feature_name = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_name, class_names=['0', '1'], filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render('DecisionTree')
