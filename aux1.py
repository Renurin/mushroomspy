import numpy as np

# Step 1: Read the values K, Ntrain, Ntest
k_neighbors = int(input())
Ntrain, Ntest = map(int, input().split())

# Step 2: Read Xtrain
Xtrain = [['-' if item == '?' else item for item in input().split()] for _ in range(Ntrain)]

# Step 3 & 4: Convert characters to numbers and normalize
class CustomLabelEncoder:
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}

    def fit(self, labels):
        unique_labels = sorted(set(labels))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        return self

    def transform(self, labels):
        return [self.label_to_index[label] for label in labels]

# Fit the encoder on all unique values in Xtrain
encoder = CustomLabelEncoder()
all_values = [item for sublist in Xtrain for item in sublist]
encoder.fit(all_values)

# Transform Xtrain
encoded_Xtrain = [encoder.transform(row) for row in Xtrain]

# Calculate mean and standard deviation without NumPy
def calculate_mean(data):
    return sum(data) / len(data)

def calculate_std(data, mean):
    return (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5

means = [calculate_mean(column) for column in zip(*encoded_Xtrain)]
std_devs = [calculate_std(column, mean) for column, mean in zip(zip(*encoded_Xtrain), means)]

# Normalize Xtrain
normalized_Xtrain = []
for row in encoded_Xtrain:
    normalized_row = []
    for i, value in enumerate(row):
        if std_devs[i] != 0:
            normalized_value = (value - means[i]) / std_devs[i]
        else:
            normalized_value = 0  # or you could use (value - means[i])
        normalized_row.append(normalized_value)
    normalized_Xtrain.append(normalized_row)

# Step 6: Read Ytrain
Ytrain = [input().strip() for _ in range(Ntrain)]

# Step 7: Read Xtest
Xtest = [['-' if item == '?' else item for item in input().split()] for _ in range(Ntest)]

# Step 8 & 9: Encode and normalize Xtest
encoded_Xtest = [encoder.transform(row) for row in Xtest]
normalized_Xtest = []
for row in encoded_Xtest:
    normalized_row = []
    for i, value in enumerate(row):
        if std_devs[i] != 0:
            normalized_value = (value - means[i]) / std_devs[i]
        else:
            normalized_value = 0  # or you could use (value - means[i])
        normalized_row.append(normalized_value)
    normalized_Xtest.append(normalized_row)

# Step 10: Define Euclidean distance
def euclidean_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

# Step 11 & 12: Predict for each test sample and output the majority label
for test_sample in normalized_Xtest:
    distances = [euclidean_distance(test_sample, train_sample) for train_sample in normalized_Xtrain]
    k_nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:k_neighbors]
    k_nearest_labels = [Ytrain[i] for i in k_nearest_indices]
    
    prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
    print(prediction)