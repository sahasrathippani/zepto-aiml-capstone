from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
CSV_FILE = BASE / "titanic.csv"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

# The raw Titanic dataset is loaded from Seaborn exactly once in this module.
df = sns.load_dataset("titanic")

# Immediate offline fallback required by the assignment.
df.to_csv(CSV_FILE, index=False)

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== INFO ===")
df.info()

print("\n=== DESCRIBE ===")
print(df.describe(include="all").transpose())

print("\n=== MISSING VALUES ===")
missing = df.isna().sum()
missing_pct = (df.isna().mean() * 100).round(2)
missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_pct
})
missing_report = missing_report[missing_report["missing_count"] > 0]
print(missing_report)

# ---------- Missing-value handling ----------
clean = df.copy()

# High-missingness 'deck' is dropped because 77%+ missing values make
# reliable imputation difficult. The exact percentage is reported above.
if "deck" in clean.columns:
    clean = clean.drop(columns=["deck"])

# Under 5%: drop affected rows.
for col in ["embarked", "embark_town"]:
    if col in clean.columns and clean[col].isna().any():
        rate = clean[col].isna().mean() * 100
        if rate < 5:
            clean = clean.dropna(subset=[col])

# 5%-30%: median-impute age.
if "age" in clean.columns and clean["age"].isna().any():
    age_rate = clean["age"].isna().mean() * 100
    if 5 <= age_rate <= 30:
        clean["age"] = clean["age"].fillna(clean["age"].median())

clean.to_csv(CSV_FILE, index=False)

print("\n=== CLEANED DATA ===")
print("Shape after cleaning:", clean.shape)
print("\nRemaining missing values:")
print(clean.isna().sum()[clean.isna().sum() > 0])

# ---------- Helper functions ----------
def iqr_outlier_count(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return int(((series < low) | (series > high)).sum())

def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=150, bbox_inches="tight")
    plt.close()

# ---------- Univariate analysis ----------
for col in ["age", "fare"]:
    plt.figure(figsize=(7, 4))
    sns.histplot(clean[col], kde=True)
    plt.title(f"Histogram of {col}")
    savefig(f"{col}_histogram.png")

    plt.figure(figsize=(7, 4))
    sns.boxplot(x=clean[col])
    plt.title(f"Boxplot of {col}")
    savefig(f"{col}_boxplot.png")

age_outliers = iqr_outlier_count(clean["age"])
fare_outliers = iqr_outlier_count(clean["fare"])

fare_mean = clean["fare"].mean()
fare_median = clean["fare"].median()
fare_mode = clean["fare"].mode().iloc[0]

print("\n=== IQR OUTLIERS ===")
print("Age outliers:", age_outliers)
print("Fare outliers:", fare_outliers)

print("\n=== FARE SUMMARY ===")
print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)

if fare_mean > fare_median > fare_mode:
    skew_statement = "Fare is right-skewed because mean > median > mode."
elif fare_mean < fare_median < fare_mode:
    skew_statement = "Fare is left-skewed because mean < median < mode."
else:
    skew_statement = "Fare does not follow a strict mean/median/mode ordering; the distribution is not classified as clearly right- or left-skewed from this ordering alone."
print(skew_statement)

# ---------- Bivariate survival rates ----------
survival_sex = clean.groupby("sex")["survived"].mean().mul(100).round(2)
survival_pclass = clean.groupby("pclass")["survived"].mean().mul(100).round(2)
survival_sex_pclass = clean.groupby(["sex", "pclass"])["survived"].mean().mul(100).round(2)

# Boolean masking examples required by the assignment.
female_rate = clean.loc[clean["sex"] == "female", "survived"].mean() * 100
male_rate = clean.loc[clean["sex"] == "male", "survived"].mean() * 100
first_class_rate = clean.loc[clean["pclass"] == 1, "survived"].mean() * 100
second_class_rate = clean.loc[clean["pclass"] == 2, "survived"].mean() * 100
third_class_rate = clean.loc[clean["pclass"] == 3, "survived"].mean() * 100

print("\n=== SURVIVAL BY SEX ===")
print(survival_sex)
print("\n=== SURVIVAL BY PCLASS ===")
print(survival_pclass)
print("\n=== SURVIVAL BY SEX + PCLASS ===")
print(survival_sex_pclass)
print("\nBoolean masking checks:")
print("Female:", round(female_rate, 2), "%")
print("Male:", round(male_rate, 2), "%")
print("1st class:", round(first_class_rate, 2), "%")
print("2nd class:", round(second_class_rate, 2), "%")
print("3rd class:", round(third_class_rate, 2), "%")

# Chart 1: survival by sex
plt.figure(figsize=(7, 4))
sns.barplot(data=clean, x="sex", y="survived")
plt.title("Survival Rate by Sex")
plt.ylabel("Survival rate")
savefig("01_survival_by_sex.png")

# Chart 2: survival by class
plt.figure(figsize=(7, 4))
sns.barplot(data=clean, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival rate")
savefig("02_survival_by_class.png")

# Chart 3: sex + class
plt.figure(figsize=(8, 5))
sns.barplot(data=clean, x="pclass", y="survived", hue="sex")
plt.title("Survival Rate by Sex and Passenger Class")
plt.ylabel("Survival rate")
savefig("03_survival_by_sex_and_class.png")

# Chart 4: age distribution by survival
plt.figure(figsize=(8, 5))
sns.boxplot(data=clean, x="survived", y="age")
plt.title("Age Distribution by Survival")
savefig("04_age_by_survival.png")

# Chart 5: fare by class and survival
plt.figure(figsize=(8, 5))
sns.boxplot(data=clean, x="pclass", y="fare", hue="survived")
plt.title("Fare by Class and Survival")
savefig("05_fare_by_class_and_survival.png")

# ---------- Exactly six columns for correlation ----------
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = clean[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Titanic Correlation Heatmap — Required Six Columns")
savefig("06_correlation_heatmap.png")

pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        pairs.append(
            (corr_cols[i], corr_cols[j], corr.iloc[i, j], abs(corr.iloc[i, j]))
        )
pairs = sorted(pairs, key=lambda x: x[3], reverse=True)
top_two = pairs[:2]

print("\n=== TOP TWO ABSOLUTE OFF-DIAGONAL CORRELATIONS ===")
for a, b, value, absolute in top_two:
    print(f"{a} vs {b}: correlation={value:.4f}, absolute={absolute:.4f}")

# ---------- Exploratory standardization ----------
scaler = StandardScaler()
standardized = clean[["age", "fare"]].copy()
standardized[["age_z", "fare_z"]] = scaler.fit_transform(
    clean[["age", "fare"]]
)

standardization_summary = pd.DataFrame({
    "original_mean": clean[["age", "fare"]].mean(),
    "original_std": clean[["age", "fare"]].std(ddof=0),
    "standardized_mean": standardized[["age_z", "fare_z"]].mean().values,
    "standardized_std": standardized[["age_z", "fare_z"]].std(ddof=0).values,
}, index=["age", "fare"])

print("\n=== STANDARDIZATION CHECK ===")
print(standardization_summary)

# ---------- Written interpretations ----------
interpretations = f"""
# Module 2 EDA Results and Interpretations

## Missing-value strategy
The raw dataset was loaded once with `sns.load_dataset("titanic")` and immediately saved as `titanic.csv`.
Missing percentages were measured before cleaning. Columns below 5% missingness had affected rows dropped,
columns from 5% to 30% were imputed, and the high-missingness `deck` column was dropped because its missingness
is too high for reliable imputation. See the printed missing-value table in the execution output.

## Age and fare
Age has {age_outliers} IQR outliers and fare has {fare_outliers} IQR outliers.
Fare mean = {fare_mean:.2f}, median = {fare_median:.2f}, mode = {fare_mode:.2f}.
{skew_statement}

## Survival by sex
The survival-by-sex chart shows a clear difference in survival rates between female and male passengers.
This provides a useful first indication that sex is associated with survival in this dataset.

## Survival by passenger class
The survival-by-class chart compares the three passenger classes directly.
The rates differ across classes, showing that passenger class is also associated with survival.

## Sex and passenger class together
The combined chart shows how sex and class interact rather than treating either variable in isolation.
This helps reveal subgroup differences that can be hidden when only one grouping variable is examined.

## Age and survival
The age-by-survival boxplot compares the age distributions of survivors and non-survivors.
It provides a visual check of whether age distributions differ between the two outcome groups.

## Fare, class and survival
The fare/class chart connects ticket price, passenger class and survival.
Because fare is related to class, the plot provides additional context for interpreting socioeconomic differences in the data.

## Strongest correlations
The two strongest absolute off-diagonal correlations are:
1. {top_two[0][0]} vs {top_two[0][1]}: {top_two[0][2]:.4f}
2. {top_two[1][0]} vs {top_two[1][1]}: {top_two[1][2]:.4f}

These coefficients describe linear association, not causation. The signs indicate the direction of the association,
while the absolute values determine the strength ranking requested by the assignment.

## Exploratory standardization
The z-score transformation makes age and fare have approximately mean 0 and standard deviation 1.
This is an exploratory EDA check only; it is not used as the modeling pipeline's preprocessing.
The modeling pipeline performs its own train-only preprocessing to avoid test-set leakage.
"""
(BASE / "eda_report.md").write_text(textwrap.dedent(interpretations).lstrip(), encoding="utf-8")

print("EDA script completed.")
