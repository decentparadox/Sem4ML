import numpy as np

class CustomDecisionTreeRootFinder:
    def _init_(self):
        pass

    def calculate_entropy(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def calculate_information_gain(self, features, labels, feature_idx):
        total_entropy = self.calculate_entropy(labels)
        unique_values, value_counts = np.unique(features[:, feature_idx], return_counts=True)
        weighted_entropy = 0
        for value, count in zip(unique_values, value_counts):
            subset_indices = np.where(features[:, feature_idx] == value)[0]
            subset_entropy = self.calculate_entropy(labels[subset_indices])
            weighted_entropy += (count / len(features)) * subset_entropy
        information_gain = total_entropy - weighted_entropy
        return information_gain

    def perform_equal_width_binning(self, feature, num_bins):
        min_val = np.min(feature)
        max_val = np.max(feature)
        bin_width = (max_val - min_val) / num_bins
        bins = [min_val + i * bin_width for i in range(num_bins)]
        binned_feature = np.digitize(feature, bins)
        return binned_feature

    def perform_frequency_binning(self, feature, num_bins):
        sorted_indices = np.argsort(feature)
        bin_size = len(feature) // num_bins
        binned_feature = np.zeros_like(feature)
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < num_bins - 1 else len(feature)
            binned_feature[sorted_indices[start_idx:end_idx]] = i
        return binned_feature

    def find_optimal_root_feature(self, features, labels, binning_method='equal_width', num_bins=None):
        if binning_method == 'equal_width':
            binning_function = self.perform_equal_width_binning
        elif binning_method == 'frequency':
            binning_function = self.perform_frequency_binning
        else:
            raise ValueError("Invalid binning method. Choose 'equal_width' or 'frequency'.")

        if num_bins is None:
            num_bins = int(np.sqrt(len(features)))  # Default number of bins

        num_features = features.shape[1]
        best_feature_index = None
        best_information_gain = -np.inf
        for feature_index in range(num_features):
            if len(np.unique(features[:, feature_index])) > 1:  # Skip features with only one unique value
                binned_feature = binning_function(features[:, feature_index], num_bins)
                current_information_gain = self.calculate_information_gain(binned_feature.reshape(-1, 1), labels, 0)
                if current_information_gain > best_information_gain:
                    best_information_gain = current_information_gain
                    best_feature_index = feature_index
        return best_feature_index

# Example usage
if __name__ == "_main_":
    features = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
    labels = np.array([1, 0, 1, 0])

    custom_tree_root_finder = CustomDecisionTreeRootFinder()
    optimal_root_feature_index = custom_tree_root_finder.find_optimal_root_feature(features, labels, binning_method='equal_width', num_bins=2)
    print("Optimal root feature index:", optimal_root_feature_index)