import click

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from probabilistic_models import naive_bayes


@click.command()
def main():
    """Example of using Naive Bayes.
    """
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    model = naive_bayes.NaiveBayes()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"Naive Bayes accuracy: {accuracy(y_test, preds)}")


if __name__ == "__main__":
    main()
