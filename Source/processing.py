from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return X, y


def split_data(X, y, test_size=0.2, seed=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )


def preprocess(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # return scaler as third element
    return X_train_scaled, X_test_scaled, scaler
