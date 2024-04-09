import numpy as np

class TreeNode:
    def _init_(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold          # Threshold value for binary split
        self.value = value                  # Class label for leaf node
        self.left = left                    # Left child node
        self.right = right                  # Right child node

class CustomDecisionTreeClassifier:
    def _init_(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def calculate_entropy(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def calculate_information_gain(self, features, labels, feature_idx, threshold):
        total_entropy = self.calculate_entropy(labels)
        left_indices = np.where(features[:, feature_idx] <= threshold)[0]
        right_indices = np.where(features[:, feature_idx] > threshold)[0]
        left_entropy = self.calculate_entropy(labels[left_indices])
        right_entropy = self.calculate_entropy(labels[right_indices])
        weighted_entropy = (len(left_indices) / len(features)) * left_entropy + (len(right_indices) / len(features)) * right_entropy
        information_gain = total_entropy - weighted_entropy
        return information_gain

    def find_best_split(self, features, labels):
        num_features = features.shape[1]
        best_information_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        for feature_idx in range(num_features):
            thresholds = np.unique(features[:, feature_idx])
            for threshold in thresholds:
                current_information_gain = self.calculate_information_gain(features, labels, feature_idx, threshold)
                if current_information_gain > best_information_gain:
                    best_information_gain = current_information_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        return best_feature_idx, best_threshold

    def build_tree(self, features, labels, depth=0):
        if depth == self.max_depth or len(np.unique(labels)) == 1:
            leaf_value = np.argmax(np.bincount(labels))
            return TreeNode(value=leaf_value)
        
        feature_idx, threshold = self.find_best_split(features, labels)
        if feature_idx is None:
            leaf_value = np.argmax(np.bincount(labels))
            return TreeNode(value=leaf_value)
        
        left_indices = np.where(features[:, feature_idx] <= threshold)[0]
        right_indices = np.where(features[:, feature_idx] > threshold)[0]

        left_subtree = self.build_tree(features[left_indices], labels[left_indices], depth+1)
        right_subtree = self.build_tree(features[right_indices], labels[right_indices], depth+1)

        return TreeNode(feature_index=feature_idx, threshold=threshold, left=left_subtree, right=right_subtree)

    def fit(self, features, labels):
        self.tree = self.build_tree(features, labels)

    def predict_sample(self, sample, node):
        if node.value is not None:
            return node.value
        if sample[node.feature_index] <= node.threshold:
            return self.predict_sample(sample, node.left)
        else:
            return self.predict_sample(sample, node.right)

    def predict(self, samples):
        if self.tree is None:
            raise ValueError("Tree not fitted.")
        predictions = []
        for sample in samples:
            prediction = self.predict_sample(sample, self.tree)
            predictions.append(prediction)
        return np.array(predictions)

if __name__ == "_main_":
    features = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
    labels = np.array([1, 1, 0, 0])

    custom_tree = CustomDecisionTreeClassifier(max_depth=1)
    custom_tree.fit(features, labels)

    print("Tree structure:")
    print("Root feature index:", custom_tree.tree.feature_index)
    print("Root threshold:", custom_tree.tree.threshold)

    test_samples = np.array([[1, 0], [0, 1]])
    predictions = custom_tree.predict(test_samples)
    print("Predictions for test samples:", predictions)