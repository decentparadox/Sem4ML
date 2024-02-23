import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_excel("embeddingsdataoutNER.xlsx")

# Function to calculate intraclass variance
def calculate_intra_class_variance(class_data):
    return np.var(class_data[['embed_1', 'embed_2']], ddof=1)

# Function to calculate interclass distance
def calculate_inter_class_distance(class_a_mean, class_b_mean):
    return np.linalg.norm(class_a_mean - class_b_mean)

# Function to calculate class centroids
def calculate_class_centroids(grouped_data):
    class_centroids = {}
    for class_label, group_data in grouped_data:
        class_mean = group_data[['embed_1', 'embed_2']].mean(axis=0)
        class_centroids[class_label] = class_mean
    return class_centroids

# Function to calculate standard deviations for each class
def calculate_class_standard_deviations(grouped_data):
    class_standard_deviations = {}
    for class_label, group_data in grouped_data:
        class_std = group_data[['embed_1', 'embed_2']].std(axis=0)
        class_standard_deviations[class_label] = class_std
    return class_standard_deviations

# Function to calculate distances between mean vectors of different classes
def calculate_class_distances(class_centroids):
    class_labels = list(class_centroids.keys())
    num_classes = len(class_labels)
    class_distances = {}

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            class_label1 = class_labels[i]
            class_label2 = class_labels[j]
            distance = np.linalg.norm(class_centroids[class_label1] - class_centroids[class_label2])
            class_distances[(class_label1, class_label2)] = distance
    return class_distances

# Function to plot histogram
def plot_histogram(feature_data):
    plt.hist(feature_data, bins=5, edgecolor='black', alpha=0.7)
    plt.xlabel('Feature')
    plt.ylabel('Frequency')
    plt.title('Histogram of Feature')
    plt.grid(True)
    plt.show()

# Function to train and evaluate k-NN classifier
def train_evaluate_knn(X_train, X_test, y_train, y_test):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    accuracy = neigh.score(X_test, y_test)
    return neigh, accuracy

# Function to predict classes for test vectors
def predict_classes(classifier, test_vectors):
    return classifier.predict(test_vectors)

# Function to calculate performance metrics
def calculate_performance_metrics(y_true, y_pred):
    confusion_mat = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return confusion_mat, precision, recall, f1

# Split the dataset into features (X) and target (y)
X = df[['embed_1', 'embed_2']]
y = df['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate k-NN classifier
classifier, accuracy = train_evaluate_knn(X_train, X_test, y_train, y_test)
print("Accuracy:", accuracy)

# Example of predicting classes for specific test vectors
test_vect = [[0.009625, 0.003646]]
predicted_classes = predict_classes(classifier, test_vect)
print("Predicted Classes:", predicted_classes)

# Calculate performance metrics
y_test_pred = classifier.predict(X_test)
confusion_mat, precision, recall, f1 = calculate_performance_metrics(y_test, y_test_pred)
print("Confusion Matrix:")
print(confusion_mat)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
