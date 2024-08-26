import math
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Step 1: Read the values K, Ntrain, Ntest
knneighbors= int(input("K: "))
Ntrain, Ntest = map(int,input("Ntrain and Ntest: \n").split())

# Step 2: Read Xtrain, a matrix of Ntrain rows(mushrooms) columns(attributes)

Xtrain = []
print("Enter Xtrain rows one by one:")
for i in range(Ntrain):
    row = list(map(str, input().split())) # Apply the str type to each input
    Xtrain.append(row)

# Step 3: Convert the characters in numbers

flat_Xtrain= np.array(Xtrain).flatten() # Label Encoder expects a fuckin 1D array???
encoder = LabelEncoder() 
encoded_flat_Xtrain= encoder.fit_transform(flat_Xtrain)
enconded_Xtrain= encoded_flat_Xtrain.reshape(len(Xtrain), -1)

print("Enconded matrix:")
print(enconded_Xtrain)

"""
print("\nMapping of labels to encoded values:")
for original, encoded in zip(encoder.classes_, range(len(encoder.classes_))):
    print(f"{original}: {encoded}")
"""

    # Not sure if this will work =) 

# Step 4: Normalize values (vector μ)(vector σ) USE THE FORMULA IN ELD PRINT!!!

# Step 5: For each attribute, substract from the mean (all attributes, 1 row) and divide by the standard deviantion, if a value does not vary in the array (standard deviation = 0),
# Set its value to 0. 

# Step 6: Label Ytrain, an array of Ntrain elements, row(p or e) and D columns

# Step 7: Read Ytrain, a matrix of Ntrain rows(mushrooms) columns(attributes)

# Step 8: Convert the characters in numbers (Ytrain)

# Step 9: Utilize the same vectors from Xtrain (vector μ)(vector σ)

# Step 10: For each xtesti: calculate the Euclidean Distance between xtesti and Xtrain's vectors. USE THE FORMULA IN ELD PRINT!!!

# Step 11: Verify between K neirest-neighbors next to xtesti, if the majority of them is p or e

# Step 12: Print the label of the majority obtained on the last step ('p' or 'e')
