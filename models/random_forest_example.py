import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tree import RandomForest


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    return accuracy


data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# using own decision tree
self_clf = RandomForest.RandomForestClassifier(n_estimators=30)
self_clf.fit(X_train, y_train)
# using sklearn decision tree
sk_clf = RandomForestClassifier(
    n_estimators=30,
    max_depth=100,
)
sk_clf.fit(X_train, y_train)

self_y_pred = self_clf.predict(X_test)
sk_y_pred = sk_clf.predict(X_test)

self_acc = accuracy(y_test, self_y_pred)
sk_acc = accuracy(y_test, sk_y_pred)

print("Accuracy Self-implemented Random Forest: ", self_acc)
print("Accuracy Scikit-learn Random Forest: ", sk_acc)
