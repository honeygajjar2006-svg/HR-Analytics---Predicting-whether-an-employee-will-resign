import pandas as pd
import numpy as np

# load the dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("First five records:")
print(df.head())

print("\nDataset shape:", df.shape)

# check the dataset
print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\nAttrition count:")
print(df["Attrition"].value_counts())

# remove unnecessary columns
df = df[[col for col in df.columns if col not in [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]]]

print("\nShape after removing unnecessary columns:", df.shape)

# convert attrition into numbers
df["Attrition"] = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})

# separate input and output
y = df["Attrition"]
X = df.drop("Attrition", axis=1)

# convert categorical columns into numbers
X = pd.get_dummies(X, dtype=int)

print("Total features:", X.shape[1])

# split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

# apply min-max normalization
minimum = X_train.min()
maximum = X_train.max()

X_train = (X_train - minimum) / (maximum - minimum)
X_test = (X_test - minimum) / (maximum - minimum)

# replace missing values in constant columns
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# save minimum and maximum values for prediction
minimum.to_pickle("minimum.pkl")
maximum.to_pickle("maximum.pkl")

# final check
print("\nFinal training shape:", X_train.shape)
print("Final testing shape:", X_test.shape)

print("\nPreprocessing completed successfully!")

# save processed data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Preprocessed data saved successfully!")