from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

BASE = Path(__file__).resolve().parent
CSV_FILE = BASE / "titanic.csv"
PLOTS = BASE / "plots"
MODELS = BASE / "models"
PLOTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# The modeling stage reads the committed CSV, NOT sns.load_dataset again.
df = pd.read_csv(CSV_FILE)

print("Loaded:", CSV_FILE)
print("Shape:", df.shape)

# Drop columns that are redundant/unsuitable for the required modeling setup.
# adult_male and alone are excluded because they are derived flags.
drop_cols = [
    "survived", "alive", "class", "who", "adult_male",
    "alone", "embark_town"
]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y = df["survived"]

# Required stratified split BEFORE preprocessing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nClass balance:")
print(y.value_counts(normalize=True).sort_index())

numeric_features = [
    c for c in ["age", "sibsp", "parch", "fare"]
    if c in X.columns
]
categorical_features = [
    c for c in ["sex", "embarked"]
    if c in X.columns
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, oob_score=True
    ),
}

results = []
roc_data = {}

def evaluate_model(name, pipeline):
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": roc_auc_score(y_test, prob),
    }
    results.append(metrics)
    roc_data[name] = (y_test, prob)

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOTS / f"cm_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()

    return pipeline

fitted = {}
for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    fitted[name] = evaluate_model(name, pipe)

# ROC curve for all three.
plt.figure(figsize=(8, 6))
for name, (actual, prob) in roc_data.items():
    fpr, tpr, _ = roc_curve(actual, prob)
    auc = roc_auc_score(actual, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "roc_curves.png", dpi=150)
plt.close()

classification_table = pd.DataFrame(results)

# Decision tree visualization with transformed feature names.
tree_pipe = fitted["Decision Tree"]
feature_names = tree_pipe.named_steps["preprocessor"].get_feature_names_out()
plt.figure(figsize=(24, 12))
plot_tree(
    tree_pipe.named_steps["model"],
    feature_names=feature_names,
    class_names=["Not survived", "Survived"],
    filled=False,
    max_depth=4,
    fontsize=7,
)
plt.title("Decision Tree")
plt.tight_layout()
plt.savefig(PLOTS / "decision_tree.png", dpi=150)
plt.close()

# ---------- Imbalance comparison ----------
imbalance_rows = []

def evaluate_variant(name, pipe):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return {
        "strategy": name,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }

baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
balanced = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )),
])
smote_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

imbalance_rows.extend([
    evaluate_variant("Baseline", baseline),
    evaluate_variant("class_weight='balanced'", balanced),
    evaluate_variant("SMOTE on training fold only", smote_pipe),
])

imbalance_table = pd.DataFrame(imbalance_rows)

# ---------- Random Forest GridSearchCV ----------
rf_grid_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        oob_score=True
    )),
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"],
}

grid = GridSearchCV(
    rf_grid_pipe,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
grid.fit(X_train, y_train)

best_rf_pipeline = grid.best_estimator_
best_rf = best_rf_pipeline.named_steps["model"]
best_params = grid.best_params_
oob_score = best_rf.oob_score_

print("\nBest RF parameters:", best_params)
print("RF OOB score:", oob_score)

# ---------- Regression: predict fare ----------
reg_target = "fare"
reg_drop = [
    "fare", "survived", "alive", "class", "who",
    "adult_male", "alone", "embark_town"
]
reg_drop = [c for c in reg_drop if c in df.columns]

X_reg = df.drop(columns=reg_drop)
y_reg = df[reg_target]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

reg_num = [c for c in X_reg.columns if pd.api.types.is_numeric_dtype(X_reg[c])]
reg_cat = [c for c in X_reg.columns if c not in reg_num]

reg_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), reg_num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), reg_cat),
])

reg_pipeline = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("model", LinearRegression()),
])

reg_pipeline.fit(Xr_train, yr_train)
reg_pred = reg_pipeline.predict(Xr_test)

mae = mean_absolute_error(yr_test, reg_pred)
rmse = np.sqrt(mean_squared_error(yr_test, reg_pred))
r2 = r2_score(yr_test, reg_pred)

n = len(yr_test)
p = reg_pipeline.named_steps["preprocessor"].transform(Xr_test).shape[1]
adjusted_r2 = (
    1 - (1 - r2) * (n - 1) / (n - p - 1)
    if n - p - 1 > 0 else np.nan
)

residuals = yr_test - reg_pred
plt.figure(figsize=(8, 5))
plt.scatter(reg_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Fare Regression Residual Plot")
plt.tight_layout()
plt.savefig(PLOTS / "fare_residual_plot.png", dpi=150)
plt.close()

# Simple explicit diagnostic based on residual spread across prediction bins.
residual_check = pd.DataFrame({
    "predicted": reg_pred,
    "absolute_residual": np.abs(residuals)
}).sort_values("predicted")
residual_check["bin"] = pd.qcut(
    residual_check["predicted"],
    q=4,
    duplicates="drop"
)
spread_by_bin = residual_check.groupby("bin", observed=True)["absolute_residual"].mean()
spread_ratio = spread_by_bin.max() / max(spread_by_bin.min(), 1e-9)
hetero_statement = (
    "The residual spread suggests possible heteroscedasticity because "
    f"the largest-to-smallest quartile mean absolute residual ratio is {spread_ratio:.2f}."
    if spread_ratio >= 2
    else
    "The residual spread does not show strong evidence of heteroscedasticity by this quartile-spread check."
)

# ---------- Final complete classification pipeline ----------
# Use the best F1 classifier from the three required classifiers.
best_row = classification_table.sort_values("f1", ascending=False).iloc[0]
best_name = best_row["model"]
full_pipeline = fitted[best_name]
joblib_path = MODELS / "full_pipeline.joblib"
joblib.dump(full_pipeline, joblib_path)

# Reload and prove prediction works on raw test input.
reloaded_pipeline = joblib.load(joblib_path)
reload_predictions = reloaded_pipeline.predict(X_test.head(5))

# ---------- Save all reports ----------
classification_table.to_csv(BASE / "classification_metrics.csv", index=False)
imbalance_table.to_csv(BASE / "imbalance_comparison.csv", index=False)

regression_metrics = pd.DataFrame([{
    "model": "Multivariate Linear Regression",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted_R2": adjusted_r2,
}])
regression_metrics.to_csv(BASE / "regression_metrics.csv", index=False)

final_table = classification_table.copy()
for col in ["MAE", "RMSE", "R2", "Adjusted_R2"]:
    final_table[col] = np.nan
final_table.to_csv(BASE / "final_model_comparison.csv", index=False)

summary = f"""
# Module 2 Modeling Results

## Stratified split
The data was split into training and test sets before preprocessing using `stratify=y`.
Stratification preserves approximately the same survived/not-survived class proportions in both sets,
which makes evaluation more representative when the target classes are not perfectly balanced.

## Classification comparison

{classification_table.to_string(index=False)}

## Imbalance comparison

{imbalance_table.to_string(index=False)}

SMOTE was applied only inside the training pipeline after the train/test split, so test observations were
not used to create synthetic training examples.

## Random Forest tuning

Best parameters:
`{best_params}`

OOB score:
`{oob_score:.4f}`

## Regression

MAE = {mae:.4f}
RMSE = {rmse:.4f}
R² = {r2:.4f}
Adjusted R² = {adjusted_r2:.4f}

{hetero_statement}

## Saved pipeline

The complete fitted preprocessing + estimator pipeline was saved to:
`models/full_pipeline.joblib`

It was reloaded with `joblib.load` and successfully produced predictions on raw test rows:
`{reload_predictions.tolist()}`

## Final classifier recommendation

Based on the required classification metrics, the highest-F1 classifier in this run was
**{best_name}** with F1 = {best_row["f1"]:.4f}, precision = {best_row["precision"]:.4f},
recall = {best_row["recall"]:.4f}, accuracy = {best_row["accuracy"]:.4f}, and
AUC = {best_row["auc"]:.4f}. These metrics should be considered together rather than relying on
accuracy alone. The saved deployable artifact is the complete fitted pipeline for this selected classifier.
"""
(BASE / "modeling_report.md").write_text(textwrap.dedent(summary).lstrip(), encoding="utf-8")

print("\n=== CLASSIFICATION COMPARISON ===")
print(classification_table.to_string(index=False))
print("\n=== IMBALANCE COMPARISON ===")
print(imbalance_table.to_string(index=False))
print("\n=== REGRESSION ===")
print(regression_metrics.to_string(index=False))
print("\nBest RF parameters:", best_params)
print("OOB score:", oob_score)
print("Saved:", joblib_path)
