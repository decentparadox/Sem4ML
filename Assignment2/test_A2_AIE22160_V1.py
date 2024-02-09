from A2_AIE22160_V1 import calculate_euclidean_distance, calculate_manhattan_distance, euclidean_distance_2d,k_nearest_neighbors,encode_labels,one_hot_encode,read_arff_file

import pytest

def test_calculate_euclidean_distance():
    assert calculate_euclidean_distance([0, 0], [3, 4]) == 5
    assert calculate_euclidean_distance([1, 2, 3], [4, 5, 6]) == pytest.approx(5.196152)

def test_calculate_manhattan_distance():
    assert calculate_manhattan_distance([0, 0], [3, 4]) == 7
    assert calculate_manhattan_distance([1, 2, 3], [4, 5, 6]) == 9

def test_euclidean_distance_2d():
    assert euclidean_distance_2d((0, 0), (3, 4)) == 5
    assert euclidean_distance_2d((1, 2), (4, 5)) == pytest.approx(4.242640)

def test_k_nearest_neighbors():
        file_path = 'chess.arff'
        data, _, _ = read_arff_file(file_path)
        coordinates = [(tuple(map(int, row[:-1])), int(row[-1])) for row in data]

    # Test with k = 1
        assert k_nearest_neighbors(1, coordinates) == "Belongs to the second class"

def test_encode_labels():
    labels = ['a', 'b', 'a', 'c', 'b']
    encoded_label, label_to_code = encode_labels(labels)


def test_one_hot_encode():
    labels = ['a', 'b', 'a', 'c', 'b']
    encoded_labels, one_hot_encoding = one_hot_encode(labels)
    assert encoded_labels == []
    assert one_hot_encoding == {'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1]}

def test_read_arff_file():
    data, attributes, labels = read_arff_file('chess.arff')


