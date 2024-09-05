import math
import numpy as np

# Step 1: Read the values K, Ntrain, Ntest
knneighbors= int(input())
Ntrain, Ntest = map(int,input().split())

# Step 2: Read Xtrain, a matrix of Ntrain rows(mushrooms) columns(attributes)

Xtrain = []
for i in range(Ntrain):
    row = list(map(str, input().split())) # Apply the str type to each input
    Xtrain.append(row)

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
        labels = np.array(labels)
        return np.array([self.label_to_index[label] for label in labels])

enconder = CustomLabelEncoder()
enconder.fit(Xtrain)
enconded_Xtrain= np.array(Xtrain)

for column_index in range(22):
    enconded_Xtrain[:,column_index] = enconder.transform(enconded_Xtrain[:,column_index])

enconded_Xtrain = enconded_Xtrain.astype(float) # Typecast cause DUMB numpy =)
means = np.mean(enconded_Xtrain, axis=0) # Array of all means of the attributes


# Step 4: Normalize values (vector μ)(vector σ) USE THE FORMULA IN ELD PRINT!!!

# Standard deviation for EACH mushroom (loop)
standard_deviation= np.zeros(22) # Inicialize empty array

# 22 standard deviation, 1 for each attribute
def calc_sum(attributes, mean):
    return np.sum((np.array(attributes)-mean) ** 2) # Check this too

for j in range(22):
    total_sum= sum(calc_sum(enconded_Xtrain[row_index], means[j]) for row_index in range(Ntrain))
    standard_deviation[j] = math.sqrt((1/Ntrain)*(total_sum))

# standard_deviation= math.sqrt((1/Ntrain)*(sum(1,Ntrain)*(attribute-mean)^2))  Bruv this formula is dogshit LOL
                              
# Step 5: For each attribute, substract from the mean (all attributes, 1 row) and divide by the standard deviantion, if a value does not vary in the array (standard deviation = 0),
# Set its value to 0. If some error occurs later, check this thing out =)

for i in range(Ntrain):
    for j in range(22):
        if standard_deviation[j] == 0:
            enconded_Xtrain[i,j] = 0
        else:
            enconded_Xtrain[i,j]= (enconded_Xtrain[i,j] - means[j]) / standard_deviation[j]
        
# Step 6: Label Ytrain, an array of Ntrain elements, row(p or e) and D columns
Ytrain = []
for i in range(Ntrain):
    row = map(str,input())
    Ytrain.append(row)
labels = np.array(Ytrain)

# Step 7: Read Xtest, a matrix of Ntest rows(mushrooms) columns(attributes)
Xtest = []
for i in range(Ntest):
    row = list(map(str, input().split())) # Apply the str type to each input
    Xtest.append(row)

# Step 8: Convert the characters in numbers (Ytrain)
enconder.fit(Xtest)
enconded_Xtest= np.array(Xtest)

for column_index in range(22):
    enconded_Xtest[:,column_index] = enconder.transform(enconded_Xtest[:,column_index])
enconded_Xtest = enconded_Xtest.astype(float) # Typecast cause DUMB numpy =)

# Step 9: Utilize the same vectors from Xtrain (vector μ)(vector σ) and normalize it
for i in range(Ntest):
    for j in range(22):
        if standard_deviation[j] == 0:
            enconded_Xtest[i,j] = 0
        else:
            enconded_Xtest[i,j]= (enconded_Xtest[i,j] - means[j]) / standard_deviation[j]

# Step 10: For each Xtesti: calculate the Euclidean Distance between xtesti and Xtrain's vectors. USE THE FORMULA IN ELD PRINT!!!

def euclidian_distance(array1, array2):
    return np.sqrt(np.sum((np.array(array1) - np.array(array2)) ** 2)) # Will use later

# Step 11: Verify between K neirest-neighbors next to xtesti, if the majority of them is p or e
# Okay this will be insane

poison_count = 0
editable_count = 0 # Inicializing variables

for testing in enconded_Xtest:
    distances = []
    for training in enconded_Xtrain:
        distance = euclidian_distance(testing, training)
        distances.append(distance) # Setting array with the distances

    k_indices= np.argsort(distances)[:knneighbors] # Sorting array =)
    k_labels = labels[k_indices] # Stupid error only int scalar bruh

    for i in range(knneighbors):
        if k_labels[i] == 'p':
            poison_count += 1
        else: editable_count +=1
     # Test if the majority are p or e

    # Step 12: Print the label of the majority obtained on the last step ('p' or 'e')
    if poison_count > editable_count:
        print('p')
    else: print('e')

# I mean, it IS running, just dunno if its right
# Okay I pull up !!!