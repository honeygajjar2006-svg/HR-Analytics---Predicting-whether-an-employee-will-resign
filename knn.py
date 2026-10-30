import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Preprocessing
df = df.drop(columns=["EmployeeCount", "EmployeeNumber",
                      "Over18", "StandardHours"], errors="ignore")
df = df.dropna(subset=["Attrition"])

# Convert categorical columns into numbers
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Attrition_Yes", axis=1)
y = df["Attrition_Yes"]

# Split dataset (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Evaluation
actual = y_test.values

correct = sum(actual == predictions)
accuracy = correct / len(actual)

tp = sum((actual == 1) & (predictions == 1))
tn = sum((actual == 0) & (predictions == 0))
fp = sum((actual == 0) & (predictions == 1))
fn = sum((actual == 1) & (predictions == 0))

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = (2 * precision * recall / (precision + recall)
      if precision + recall else 0)

print("KNN Results (k=5, 80%-20% Split)")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision (Yes):", round(precision * 100, 2), "%")
print("Recall (Yes):", round(recall * 100, 2), "%")
print("F1-Score (Yes):", round(f1 * 100, 2), "%")
# Print Confusion Matrix
print("\nConfusion Matrix:")
print("                 Predicted")
print("                 No     Yes")
print("Actual No       ", tn, "   ", fp)
print("Actual Yes      ", fn, "    ", tp)

# Predicted table
table = pd.DataFrame({
    "Actual Attrition": actual,
    "Predicted Attrition": predictions
})

table["Actual Attrition"] = table["Actual Attrition"].map(
    {0: "No", 1: "Yes"}
)
table["Predicted Attrition"] = table["Predicted Attrition"].map(
    {0: "No", 1: "Yes"}
)

table["Result"] = [
    "Correct" if a == p else "Incorrect"
    for a, p in zip(table["Actual Attrition"],
                    table["Predicted Attrition"])
]


# Evaluation chart
plt.figure(figsize=(8, 5))
plt.bar(
    ["Accuracy", "Precision", "Recall", "F1-Score"],
    [accuracy * 100, precision * 100, recall * 100, f1 * 100]
)
plt.title("KNN Evaluation (80%-20% Split)")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("knn_evaluation_chart.png")
plt.show()