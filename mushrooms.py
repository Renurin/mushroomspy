import numpy as np

# Step 1: Read the values K, Ntrain, Ntest
knneighbors = int(input())
Ntrain, Ntest = map(int, input().split())

# Step 2: Read Xtrain
Xtrain = [list(input().split()) for _ in range(Ntrain)]

# Step 3: Convert the characters in numbers & Step 4: Normalize values (vector μ)(vector σ)
class CustomLabelEncoder:
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}
        self.is_fitted = False

    def fit(self, labels):
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        self.is_fitted = True
        return self

    def transform(self, labels):
        if not self.is_fitted:
            raise ValueError("LabelEncoder not fitted yet.")
        return np.array([self.label_to_index[label] for label in labels])

encoder = CustomLabelEncoder()
all_labels = [item for sublist in Xtrain for item in sublist]
encoder.fit(all_labels)

encoded_Xtrain = np.array([encoder.transform(row) for row in Xtrain],dtype=float)

# Calculate standard deviations & means
means = np.array([sum(data)/len(data) for data in zip(*encoded_Xtrain)])

def calculate_deviation(data,mean):
    return np.sqrt((sum((x - mean)**2 for x in data)/len(data)))
std_devs=np.array([calculate_deviation(column, mean) for column, mean in zip(zip(*encoded_Xtrain), means)])

# Step 4 & 5: Normalize Xtrain
normalized_Xtrain = []
for row in encoded_Xtrain:
    normalized_row= []
    for i, data in enumerate (row):
        if std_devs[i] == 0:
            normalized_value = 0
        else:
            normalized_value = (data - means[i]) / std_devs[i]
        normalized_row.append(normalized_value)
    normalized_Xtrain.append(normalized_row)

# Step 6: Read Ytrain
Ytrain = [input().strip() for _ in range(Ntrain)]

# Step 7: Read Xtest
Xtest = [list(input().split()) for _ in range(Ntest)]

# Step 8: Encode Xtest
encoded_Xtest = np.array([[encoder.transform([val])[0] for val in row] for row in Xtest])

# Step 9: Normalize Xtest
normalized_Xtest= []
for row in encoded_Xtest:
    normalized_row= []
    for i, data in enumerate (row):
        if std_devs[i] == 0:
            normalized_value = 0
        else:
            normalized_value = (data - means[i]) / std_devs[i]
        normalized_row.append(normalized_value)
    normalized_Xtest.append(normalized_row)
    

# Step 10: Define Euclidean distance
def euclidean_distance(test, train):
    return sum((array1 - array2) ** 2 for array1,array2 in zip(test,train)) ** 0.5

# Step 11: Predict for each test sample
for test_sample in normalized_Xtest:

    distances = [euclidean_distance(test_sample, train_sample) for train_sample in normalized_Xtrain]
    k_indices = np.argsort(distances)[:knneighbors]  # Get indices of k smallest distances
    k_labels = [Ytrain[aux] for aux in k_indices] # Get the labels in Ytrain

    prediction_label = max(set(k_labels), key=k_labels.count)
    # Step 12: Print labels
    print(prediction_label)