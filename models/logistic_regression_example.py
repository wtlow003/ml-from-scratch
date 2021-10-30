import click
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from linear_models import logistic_regression


@click.command()
@click.option("--lr", type=float, default=0.0001, help="Learning rate.")
@click.option("--n_iters", type=int, default=500, help="Number of training iterations.")
def main(lr: float, n_iters: int):
    """Example of using Logistic Regression."""
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    model = logistic_regression.LogisticRegression(lr=lr, n_iters=n_iters)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"Logistic Regression accuracy: {accuracy(y_test, preds)}")


if __name__ == "__main__":
    main()
