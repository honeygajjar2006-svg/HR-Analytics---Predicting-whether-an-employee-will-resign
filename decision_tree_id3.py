import pandas as pd
import matplotlib.pyplot as plt
import random
from math import log2
from collections import Counter

# Load dataset
data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Preprocessing
data = data.drop(columns=["EmployeeCount", "EmployeeNumber",
                          "Over18", "StandardHours"], errors="ignore")
data = data.dropna(subset=["Attrition"])

# Convert numerical columns into categories
for col in data.select_dtypes(include="number").columns:
    data[col] = pd.cut(data[col], bins=4, labels=False).astype(str)

data = data.fillna("Unknown")

# Separate input and output
X = data.drop("Attrition", axis=1)
y = data["Attrition"]


# Split dataset
def split_data(X, y, test_size):
    random.seed(42)
    train = []
    test = []

    for label in y.unique():
        rows = data[data["Attrition"] == label].to_dict("records")
        random.shuffle(rows)

        count = int(len(rows) * test_size)
        test += rows[:count]
        train += rows[count:]

    return pd.DataFrame(train), pd.DataFrame(test)


# Calculate Entropy
def entropy(labels):
    total = len(labels)

    if total == 0:
        return 0

    counts = Counter(labels)

    return -sum((n / total) * log2(n / total)
                for n in counts.values())


# Find best feature using Information Gain
def best_feature(data, features):
    best_gain = -1
    best = None

    for feature in features:
        gain = entropy(data["Attrition"])

        for value in data[feature].unique():
            group = data[data[feature] == value]
            gain -= len(group) / len(data) * entropy(group["Attrition"])

        if gain > best_gain:
            best_gain = gain
            best = feature

    return best


# Build ID3 Decision Tree
def build_tree(data, features):
    labels = data["Attrition"]

    if len(labels.unique()) == 1 or len(features) == 0:
        return labels.mode()[0]

    feature = best_feature(data, features)

    if feature is None:
        return labels.mode()[0]

    tree = {feature: {}}

    for value in data[feature].unique():
        group = data[data[feature] == value]
        remaining = [f for f in features if f != feature]
        tree[feature][value] = build_tree(group, remaining)

    return tree


# Predict using the tree
def predict(tree, row, default):
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    value = row[feature]

    if value not in tree[feature]:
        return default

    return predict(tree[feature][value], row, default)


# Evaluation
def evaluate(actual, predicted):
    correct = sum(a == p for a, p in zip(actual, predicted))
    accuracy = correct / len(actual)

    tp = sum(a == "Yes" and p == "Yes"
             for a, p in zip(actual, predicted))
    tn = sum(a == "No" and p == "No"
             for a, p in zip(actual, predicted))
    fp = sum(a == "No" and p == "Yes"
             for a, p in zip(actual, predicted))
    fn = sum(a == "Yes" and p == "No"
             for a, p in zip(actual, predicted))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0)

    return accuracy, precision, recall, f1, [[tn, fp], [fn, tp]]


# Scenario 1: 80%-20%
train1, test1 = split_data(X, y, 0.2)
features = list(X.columns)

tree1 = build_tree(train1, features)
default1 = train1["Attrition"].mode()[0]

actual1 = test1["Attrition"].tolist()
predicted1 = [predict(tree1, row, default1)
              for _, row in test1.iterrows()]

result1 = evaluate(actual1, predicted1)

print("Scenario 1: 80%-20%")
print("Accuracy:", round(result1[0] * 100, 2), "%")
print("Precision:", round(result1[1] * 100, 2), "%")
print("Recall:", round(result1[2] * 100, 2), "%")
print("F1-Score:", round(result1[3] * 100, 2), "%")
# Print Confusion Matrix
tn, fp = result1[4][0]
fn, tp = result1[4][1]

print("\nConfusion Matrix:")
print("                 Predicted")
print("                 No     Yes")
print("Actual No       ", tn, "   ", fp)
print("Actual Yes      ", fn, "    ", tp)

# Predicted table
predicted_table = pd.DataFrame({
    "Actual Attrition": actual1,
    "Predicted Attrition": predicted1
})

predicted_table["Result"] = predicted_table.apply(
    lambda row: "Correct" if row["Actual Attrition"] ==
    row["Predicted Attrition"] else "Incorrect", axis=1
)

# Evaluation chart
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values = [result1[0] * 100, result1[1] * 100,
          result1[2] * 100, result1[3] * 100]

plt.figure(figsize=(9, 5))
plt.bar(metrics, values)
plt.title("ID3 Decision Tree Evaluation (80%-20% Split)")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("evaluation_chart.png")
plt.show()


# Plot the decision tree
def plot_node(tree, x, y, width):
    if not isinstance(tree, dict):
        plt.text(x, y, str(tree), ha="center",
                 bbox=dict(facecolor="lightblue"))
        return

    feature = next(iter(tree))
    plt.text(x, y, feature, ha="center",
             bbox=dict(facecolor="lightgreen"))

    left_x = x - width / 2
    right_x = x + width / 2
    next_y = y - 1

    values = list(tree[feature].values())

    plt.plot([x, left_x], [y, next_y], "k-")
    plot_node(values[0], left_x, next_y, width / 2)

    if len(values) > 1:
        plt.plot([x, right_x], [y, next_y], "k-")
        plot_node(values[1], right_x, next_y, width / 2)


plt.figure(figsize=(16, 9))
plot_node(tree1, 0, 5, 8)
plt.title("ID3 Decision Tree - 80%-20% Split")
plt.axis("off")
plt.tight_layout()
plt.show()
