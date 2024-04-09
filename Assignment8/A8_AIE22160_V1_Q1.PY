import numpy as np

class CustomDecisionTreeRootFinder:
    def _init_(self):
        pass

    def calculate_entropy(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def calculate_information_gain(self, features, labels, feature_index):
        total_entropy = self.calculate_entropy(labels)
        unique_values, value_counts = np.unique(features[:, feature_index], return_counts=True)
        weighted_entropy = 0
        for value, count in zip(unique_values, value_counts):
            subset_indices = np.where(features[:, feature_index] == value)[0]
            subset_entropy = self.calculate_entropy(labels[subset_indices])
            weighted_entropy += (count / len(features)) * subset_entropy
        information_gain = total_entropy - weighted_entropy
        return information_gain

    def find_optimal_root_feature(self, features, labels):
        num_features = features.shape[1]
        best_feature_index = None
        best_information_gain = -np.inf
        for feature_index in range(num_features):
            current_information_gain = self.calculate_information_gain(features, labels, feature_index)
            if current_information_gain > best_information_gain:
                best_information_gain = current_information_gain
                best_feature_index = feature_index
        return best_feature_index

# Example usage
if __name__ == "_main_":
    features = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
    labels = np.array([1,1,1,1])
    custom_tree_root_finder = CustomDecisionTreeRootFinder()
    optimal_root_feature_index = custom_tree_root_finder.find_optimal_root_feature(features, labels)
    print("Optimal root feature index:", optimal_root_feature_index)



