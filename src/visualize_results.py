import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Load the saved test data
with open("../docs/testing_results.json", "r") as f:
    test_data = json.load(f)

predicted = np.array(test_data["predicted_action"])
ground_truth = np.array(test_data["ground_truth"])

# Compute confusion matrix
cm = confusion_matrix(ground_truth, predicted)

# Define class labels (assuming 3 actions: 0, 1, 2)
class_labels = ["Pick", "Insert", "Lock"]

# Plot confusion matrix
plt.figure(figsize=(20, 15))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, annot_kws={"size": 20})
# Adjust font sizes
plt.xlabel("Predicted Label", fontsize=30)
plt.ylabel("True Label", fontsize=30)
plt.title("Confusion Matrix : 3-class", fontsize=35)
plt.xticks(fontsize=25)  # Adjust x-axis tick labels
plt.yticks(fontsize=25)  # Adjust y-axis tick labels

plt.savefig("../docs/confusion_test_3_class.png")
plt.show()

# Compute precision, recall, and F1-score
report = classification_report(ground_truth, predicted, target_names=class_labels, output_dict=True)

# Convert to DataFrame
df = pd.DataFrame(report).transpose()

# Save to CSV
df.to_csv("../docs/classification_report_3_class.csv", index=True)