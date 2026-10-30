
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# Load preprocessed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").iloc[:, 0]
y_test = pd.read_csv("y_test.csv").iloc[:, 0]

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])
print("Total features:", X_train.shape[1])


#KNN MODEL

print("\nKNN MODEL")

k = 5

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred, zero_division=0)
knn_recall = recall_score(y_test, knn_pred, zero_division=0)
knn_f1 = f1_score(y_test, knn_pred, zero_division=0)

print("K value:", k)
print("Accuracy:", round(knn_accuracy * 100, 2), "%")
print("Precision:", round(knn_precision * 100, 2), "%")
print("Recall:", round(knn_recall * 100, 2), "%")
print("F1-score:", round(knn_f1 * 100, 2), "%")

print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test, knn_pred,
    display_labels=["No Attrition", "Attrition"],
    cmap="Blues"
)
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.savefig("knn_confusion_matrix.png")
plt.show()


# Save KNN training data to reconstruct the model later
X_train.to_csv("knn_training_features.csv", index=False)
y_train.to_csv("knn_training_labels.csv", index=False)

with open("knn_settings.txt", "w") as file:
    file.write(str(k))

print("KNN training data saved successfully!")

#ACTUAL VS KNN PREDICTED GRAPH

actual_counts = pd.Series(y_test).value_counts().reindex([0, 1], fill_value=0)
knn_counts = pd.Series(knn_pred).value_counts().reindex([0, 1], fill_value=0)

labels = ["No Attrition", "Attrition"]
x = range(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    [i - width / 2 for i in x],
    actual_counts.values,
    width,
    label="Actual"
)

plt.bar(
    [i + width / 2 for i in x],
    knn_counts.values,
    width,
    label="KNN Predicted"
)

plt.xlabel("Attrition Class")
plt.ylabel("Number of Employees")
plt.title("KNN: Actual vs Predicted Attrition")
plt.xticks(list(x), labels)
plt.legend()
plt.tight_layout()
plt.savefig("knn_actual_vs_predicted.png")
plt.show()

# KNN PREDICTED TABLE - FIRST 10 RECORDS

knn_table = pd.DataFrame({
    "Actual Attrition": ["Yes" if value == 1 else "No"
                         for value in y_test.iloc[:10]],
    "Predicted Attrition": ["Yes" if value == 1 else "No"
                            for value in knn_pred[:10]]
})

knn_table["Result"] = [
    "Correct" if actual == predicted else "Incorrect"
    for actual, predicted in zip(
        knn_table["Actual Attrition"],
        knn_table["Predicted Attrition"]
    )
]

print("\nKNN PREDICTION TABLE (First 10 Records)")
print(knn_table.to_string(index=False))


#ID3 DECISION TREE 

print("\nID3 DECISION TREE")


# Calculate entropy
def entropy(labels):
    total = len(labels)

    if total == 0:
        return 0

    counts = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    result = 0

    for count in counts.values():
        probability = count / total
        result = result - probability * math.log2(probability)

    return result


# Calculate information gain
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    weighted_entropy = 0
    total = len(data)

    for value in data[feature].unique():
        subset = data[data[feature] == value]
        weighted_entropy += (len(subset) / total) * entropy(subset[target])

    return total_entropy - weighted_entropy


# Build ID3 tree
def build_tree(data, features, target, default=0):
    if len(data) == 0:
        return default

    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    if len(features) == 0:
        return data[target].mode().iloc[0]

    best_feature = features[0]
    best_gain = information_gain(data, best_feature, target)

    for feature in features[1:]:
        gain = information_gain(data, feature, target)

        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    tree = {"feature": best_feature, "branches": {}}
    remaining_features = [
        feature for feature in features if feature != best_feature
    ]

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]

        tree["branches"][str(value)] = build_tree(
            subset,
            remaining_features,
            target,
            data[target].mode().iloc[0]
        )

    return tree


# Predict using ID3 tree
def predict_tree(tree, row, default=0):
    if not isinstance(tree, dict):
        return tree

    feature = tree["feature"]
    value = str(row[feature])

    if value in tree["branches"]:
        return predict_tree(tree["branches"][value], row, default)

    return default


# Load original dataset for ID3
original = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

original = original.drop(
    columns=[
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ],
    errors="ignore"
)

# Keep the same train-test split as the preprocessing code
from sklearn.model_selection import train_test_split

train_indices, test_indices = train_test_split(
    original.index,
    test_size=0.2,
    random_state=42,
    stratify=original["Attrition"]
)

tree_data = original.copy()

# Convert numeric features into categories for ID3
for column in tree_data.columns:
    if column != "Attrition" and pd.api.types.is_numeric_dtype(
        tree_data[column]
    ):
        tree_data[column] = pd.cut(
            tree_data[column],
            bins=4,
            labels=False,
            duplicates="drop"
        )

tree_data = tree_data.fillna("Unknown").astype(str)

target = "Attrition"
features = [column for column in tree_data.columns if column != target]

tree_train = tree_data.loc[train_indices].copy()
tree_test = tree_data.loc[test_indices].copy()

# Use the majority class as the default prediction
default_class = tree_train[target].mode().iloc[0]

# Build tree using training data only
id3_tree = build_tree(tree_train, features, target, default_class)

# Predict test data
id3_pred = []

for index, row in tree_test.iterrows():
    prediction = predict_tree(id3_tree, row, default_class)
    id3_pred.append(prediction)

y_test_tree = tree_test[target].tolist()

id3_accuracy = accuracy_score(y_test_tree, id3_pred)
id3_precision = precision_score(
    y_test_tree, id3_pred, pos_label="Yes", zero_division=0
)
id3_recall = recall_score(
    y_test_tree, id3_pred, pos_label="Yes", zero_division=0
)
id3_f1 = f1_score(
    y_test_tree, id3_pred, pos_label="Yes", zero_division=0
)

print("Accuracy:", round(id3_accuracy * 100, 2), "%")
print("Precision:", round(id3_precision * 100, 2), "%")
print("Recall:", round(id3_recall * 100, 2), "%")
print("F1-score:", round(id3_f1 * 100, 2), "%")
print("\nID3 Decision Tree Confusion Matrix:")

cm = confusion_matrix(
    y_test_tree,
    id3_pred,
    labels=["No", "Yes"]
)

print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Attrition", "Attrition"]
).plot(cmap="Greens")

plt.title("ID3 Decision Tree Confusion Matrix")
plt.tight_layout()
plt.savefig("id3_confusion_matrix.png")
plt.show()

# Save ID3 tree as readable text
tree_model = {
    "tree": id3_tree,
    "default": default_class,
    "features": features
}

with open("id3_model.txt", "w") as file:
    file.write(repr(tree_model))

print("ID3 tree saved successfully!")

# ID3 PREDICTED TABLE - FIRST 10 RECORDS

id3_table = pd.DataFrame({
    "Actual Attrition": y_test_tree[:10],
    "Predicted Attrition": id3_pred[:10]
})

id3_table["Result"] = [
    "Correct" if actual == predicted else "Incorrect"
    for actual, predicted in zip(
        id3_table["Actual Attrition"],
        id3_table["Predicted Attrition"]
    )
]

print("\nID3 DECISION TREE PREDICTION TABLE (First 10 Records)")
print(id3_table.to_string(index=False))


#ACTUAL VS ID3 PREDICTED GRAPH

id3_actual_counts = pd.Series(
    [1 if value == "Yes" else 0 for value in y_test_tree]
).value_counts().reindex([0, 1], fill_value=0)

id3_predicted_counts = pd.Series(
    [1 if value == "Yes" else 0 for value in id3_pred]
).value_counts().reindex([0, 1], fill_value=0)

plt.figure(figsize=(8, 5))

plt.bar(
    [i - width / 2 for i in x],
    id3_actual_counts.values,
    width,
    label="Actual"
)

plt.bar(
    [i + width / 2 for i in x],
    id3_predicted_counts.values,
    width,
    label="ID3 Predicted"
)

plt.xlabel("Attrition Class")
plt.ylabel("Number of Employees")
plt.title("ID3 Decision Tree: Actual vs Predicted Attrition")
plt.xticks(list(x), labels)
plt.legend()
plt.tight_layout()
plt.savefig("id3_actual_vs_predicted.png")
plt.show()


#MODEL COMPARISON

print("\nMODEL COMPARISON")

results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "KNN": [
        knn_accuracy * 100,
        knn_precision * 100,
        knn_recall * 100,
        knn_f1 * 100
    ],
    "ID3 Decision Tree": [
        id3_accuracy * 100,
        id3_precision * 100,
        id3_recall * 100,
        id3_f1 * 100
    ]
})

print(results.round(2).to_string(index=False))

results.to_csv("model_comparison.csv", index=False)

# Comparison bar graph
results.set_index("Metric").plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("KNN vs ID3 Decision Tree")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

print("\nAll models trained and evaluated successfully!")

