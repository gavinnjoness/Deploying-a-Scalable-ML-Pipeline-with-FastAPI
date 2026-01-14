import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


def main():
    # Use the project root (folder containing this train_model.py)
    project_path = os.path.dirname(os.path.abspath(__file__))

    # Load the census.csv data
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)

    # Split into train/test
    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["salary"],
    )

    # DO NOT MODIFY (per starter instructions)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process training data
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    # Process test data (reuse encoder/lb)
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Train model
    model = train_model(X_train, y_train)

    # Ensure model directory exists
    model_dir = os.path.join(project_path, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Save model + encoder (+ lb for later API/inference use)
    model_path = os.path.join(model_dir, "model.pkl")
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    lb_path = os.path.join(model_dir, "lb.pkl")

    save_model(model, model_path)
    save_model(encoder, encoder_path)
    save_model(lb, lb_path)

    # Load model back (sanity check)
    model = load_model(model_path)

    # Inference on test set
    preds = inference(model, X_test)

    # Print overall metrics
    p, r, fb = compute_model_metrics(y_test, preds)
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

    # Write slice output (clear file first)
    slice_output_path = os.path.join(project_path, "slice_output.txt")
    with open(slice_output_path, "w") as f:
        f.write("")  # clear file

    # Compute performance on slices and write to slice_output.txt
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]

            p_s, r_s, fb_s = performance_on_categorical_slice(
                test,
                col,
                slicevalue,
                cat_features,
                "salary",
                encoder,
                lb,
                model,
            )

            with open(slice_output_path, "a") as f:
                print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
                print(
                    f"Precision: {p_s:.4f} | Recall: {r_s:.4f} | F1: {fb_s:.4f}",
                    file=f,
                )


if __name__ == "__main__":
    main()
