# Model Card

## Model Details
This model is a binary classification model that predicts whether an individual’s income is `>50K` or `<=50K` based on U.S. Census-style demographic and employment-related features. The model is a scikit-learn Logistic Regression classifier trained on tabular data. Categorical features are one-hot encoded using a fitted `OneHotEncoder`, and the target label (`salary`) is binarized using a `LabelBinarizer`. The trained model and preprocessing artifacts are saved as `model/model.pkl`, `model/encoder.pkl`, and `model/lb.pkl`.

## Intended Use
This model is intended for educational purposes as part of an ML deployment pipeline project. It demonstrates a full workflow including preprocessing, training, evaluation, slice-based performance reporting, and deployment behind a REST API. The model should not be used to make real-world decisions about individuals (e.g., hiring, compensation, lending, or eligibility decisions).

## Training Data
The training data comes from `data/census.csv` and contains 32,561 records with 15 columns. The target label is `salary`, and the feature set includes both numerical and categorical attributes such as age, workclass, education, occupation, relationship, race, sex, hours-per-week, and native-country. Categorical features are one-hot encoded during preprocessing.

## Evaluation Data
The evaluation data is a held-out test split created from the same `census.csv` dataset. The dataset is split into training and test sets using an 80/20 split with stratification on the `salary` label to preserve class balance across splits.

## Metrics
The following metrics are used to evaluate model performance on the test set:
- Precision
- Recall
- F1 score (FBeta with beta=1)

Model performance on the held-out test set:
- Precision: 0.7419
- Recall: 0.5721
- F1: 0.6460

In addition to overall metrics, model performance is computed on slices of the data across categorical feature values. Slice results are written to `slice_output.txt`, with metrics reported for each unique value in each categorical feature.

## Ethical Considerations
This dataset contains sensitive demographic attributes (e.g., race and sex). Models trained on these features can learn and reproduce historical biases present in the data. Performance may vary across subgroups, and differences in error rates can lead to unfair outcomes if used in real decision-making. Even if the model performs well overall, subgroup performance should be reviewed and monitored using the slice metrics and additional fairness evaluation where appropriate.

## Caveats and Recommendations
This model is trained on a fixed historical dataset and may not generalize to other populations, time periods, or data collection processes. The `salary` label is a proxy target and does not capture the full context of income or employment outcomes. The model’s performance varies across slices of categorical features; users should consult `slice_output.txt` for subgroup metrics and investigate any low-performing slices. If this model were used beyond a learning context, recommendations would include more robust data validation, bias/fairness testing, monitoring for drift, and careful consideration of whether sensitive features should be included.
