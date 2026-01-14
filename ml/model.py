import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=0)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions."""
    return model.predict(X)


def save_model(model, path):
    """Serializes model (or encoder/lb) to a file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """Loads pickle file from `path` and returns it."""
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Computes model metrics on a slice of the data specified by a column name and value.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing features and label.
    column_name : str
        Column used to slice the data.
    slice_value : str
        Value in column_name to filter on.
    categorical_features : list[str]
        List of categorical feature names.
    label : str
        Name of label column.
    encoder : OneHotEncoder
        Fitted encoder.
    lb : LabelBinarizer
        Fitted label binarizer.
    model
        Trained model.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    slice_df = data[data[column_name] == slice_value]

    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
