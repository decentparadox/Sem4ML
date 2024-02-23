import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./simplified_coffee.csv')  # Make sure to update the path to your dataset file

# Categorize the rating column into "High" and "Low"
df['Label'] = ['High' if x >= 93 else 'Low' for x in df['rating']]

# Define functions for calculations

# Function to calculate class centroids for '100g_USD'
def calculate_class_centroids(grouped_data):
    class_centroids = {}
    for class_label, group_data in grouped_data:
        class_mean = group_data['100g_USD'].mean()
        class_centroids[class_label] = class_mean
    return class_centroids

# Function to calculate standard deviations for each class for '100g_USD'
def calculate_class_standard_deviations(grouped_data):
    class_standard_deviations = {}
    for class_label, group_data in grouped_data:
        class_std = group_data['100g_USD'].std()
        class_standard_deviations[class_label] = class_std
    return class_standard_deviations

# Group the data by the 'Label' column
grouped_data = df.groupby('Label')

# Calculate class centroids and standard deviations
class_centroids = calculate_class_centroids(grouped_data)
class_standard_deviations = calculate_class_standard_deviations(grouped_data)

# Plot histogram for '100g_USD'
plt.hist(df['100g_USD'], bins=5, edgecolor='black', alpha=0.7)
plt.xlabel('Price per 100g USD')
plt.ylabel('Frequency')
plt.title('Histogram of Price per 100g USD')
plt.grid(True)
plt.show()

# Calculate mean and variance for '100g_USD'
mean_100g_USD = np.mean(df['100g_USD'])
variance_100g_USD = np.var(df['100g_USD'], ddof=1)

print(f"Mean of '100g_USD': {mean_100g_USD}")
print(f"Variance of '100g_USD': {variance_100g_USD}")


import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./simplified_coffee.csv')  # Update the path to your dataset

# Categorize the rating column into two classes
df['Label'] = ['High' if x >= 93 else 'Low' for x in df['rating']]

# Select two arbitrary values from '100g_USD' as feature vectors for Minkowski distance calculation
vector1, vector2 = df['100g_USD'].iloc[0], df['100g_USD'].iloc[1]

# Calculate Minkowski distance for r from 1 to 10 and plot
rs = range(1, 11)
distances = [minkowski([vector1], [vector2], r) for r in rs]
plt.figure(figsize=(10, 5))
plt.plot(rs, distances, marker='o', linestyle='-', color='blue')
plt.xlabel('Value of r (Minkowski parameter)')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance between Two Feature Vectors for r=1 to 10')
plt.grid(True)
plt.show()

# Prepare data for model training and evaluation
X = df[['100g_USD']]  # Feature vector
y = df['Label']  # Target variable

# Divide dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a k-NN classifier (k=3), test its accuracy, and evaluate performance metrics
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
knn_accuracy = knn_classifier.score(X_test, y_test)
y_pred = knn_classifier.predict(X_test)
confusion_mat = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'k-NN Classifier (k=3) Accuracy: {knn_accuracy}')
print('Confusion Matrix:', confusion_mat)
print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1}')

# Compare NN classifier with k-NN for varying values of k
ks = range(1, 12)
accuracies = []
for k in ks:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    accuracies.append(classifier.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(ks, accuracies, marker='o', linestyle='-', color='red')
plt.xlabel('Number of Neighbors k')
plt.ylabel('Accuracy')
plt.title('Accuracy Trend for k-NN Classifier with varying k')
plt.grid(True)
plt.show()

