def calculate_euclidean_distance(x, y):
    euclid_dist = 0
    for i in range(len(x)):
        euclid_dist += (x[i] - y[i])**2
    return euclid_dist**0.5

def calculate_manhattan_distance(x, y):
    manhattan_dist = 0
    for i in range(len(x)):
        manhattan_dist += abs(x[i] - y[i])
    
    return manhattan_dist

def euclidean_distance_2d(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def k_nearest_neighbors(k, coordinates):
    distances = []

    for i in range(1, len(coordinates)):
        calculated_distance = calculate_euclidean_distance(coordinates[0], coordinates[i])
        distances.append((calculated_distance, coordinates[i][2]))
    
    for j in range(len(distances)):
        for l in range(0, len(distances) - j - 1):
            if distances[l][0] > distances[l+1][0]:
                distances[l], distances[l+1] = distances[l+1], distances[l]
        
    k_nearest = distances[:k]

    frequency1 = 0
    for distance, label in k_nearest: 
        if label == 1:
            frequency1 += 1
    
    frequency2 = k - frequency1

    if frequency1 > frequency2: 
        return "Belongs to the first class"
    else: 
        return "Belongs to the second class"
    

def encode_labels(labels):
    unique_label_set = set(labels)
    label_to_code = {}
    code = 0

    for label in unique_label_set:
        label_to_code[label] = code
        code += 1
    
    for label in labels:
        encoded_label = label_to_code[label]
    
    return encoded_label, label_to_code

def one_hot_encode(labels):
    unique_labels = sorted(set(labels)) 
    one_hot_encoding = {}
    for label in unique_labels:
        one_hot_encoding[label] = [0] * len(unique_labels)
   
    for i, label in enumerate(unique_labels):
        one_hot_encoding[label][i] = 1
    encoded_labels = []
    for label in labels:
        one_hot_encoding[label]
    
    return encoded_labels, one_hot_encoding

def read_arff_file(file_path):
    with open(file_path, 'r') as f:
        text = f.readline()
    
    attributes = []
    data = []
    labels = []
    read_data = False

    for line in text:
        line = text.strip()

        if not line or text.startswith('%'):
            continue
        
        if line.lower().startswith('@attribute'):
            attribute_name = line.split()[1]
            attributes.append(attribute_name)
        
        elif line.lower().startswith('@data'):
            read_data = True
        
        elif read_data:
            data.append(line.split(','))
        
        elif line.lower().startswith('@relation'):
            relation_name = line.split()[1]
        
        elif line.lower().startswith('{'):
            labels.extend(line.strip('{}').split(','))
    
    return data, attributes, labels



file_path = 'chess.arff'
data, attributes, labels = read_arff_file(file_path)

coordinates = []
for row in data: 
    for value in row[:-1]:
        featured_value = float(value)
        tuple(featured_value)
        label = int(row[-1])

        coordinates.append((featured_value, label))


k = int(input("Enter a certain value of k = "))
read_arff_file("chess.arff")
result = k_nearest_neighbors(k, coordinates)
print("Classification result:", result)

