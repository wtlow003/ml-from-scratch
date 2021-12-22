import click
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from neighbour import knn


@click.command()
@click.option("--k", type=int, default=5, help="K-number of nearest samples.")
def main(k: int = 5):
    """Example of using K-Nearest Neighbor."""
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    model = knn.KNearestNeighbors(k=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"K-Nearest Neighbors accuracy: {accuracy(y_test, preds)}")


if __name__ == "__main__":
    main()
