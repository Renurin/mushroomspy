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

# Step 3: Convert the characters in numbers & Step 4: Normalize values (vector μ)(vector σ)

enconder = LabelEncoder()
enconded_Xtrain = np.array(Xtrain)

for column_index in range(22):
    enconded_Xtrain[:,column_index] = enconder.fit_transform(enconded_Xtrain[:,column_index])

enconded_Xtrain = enconded_Xtrain.astype(float) # Typecast cause DUMB numpy =)
means = np.mean(enconded_Xtrain, axis=0) # Array of all means of the attributes

print(enconded_Xtrain) # I think this actually worked =)
print(means)

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
 
# Step 7: Read Xtest, a matrix of Ntest rows(mushrooms) columns(attributes)
Xtest = []
print("Enter Xtest rows one by one:")
for i in range(Ntest):
    row = list(map(str, input().split())) # Apply the str type to each input
    Xtest.append(row)

# Step 8: Convert the characters in numbers (Ytrain)
enconded_Xtest= np.array(Ntest)
for column_index in range(22):
    enconded_Xtest[:,column_index] = enconder.fit_transform(enconded_Xtest[:,column_index])

enconded_Xtest = enconded_Xtest.astype(float) # Typecast cause DUMB numpy =)

# Step 9: Utilize the same vectors from Xtrain (vector μ)(vector σ) and normalize it
for i in range(Ntest):
    for j in range(22):
        if standard_deviation[j] == 0:
            enconded_Xtest[i,j] = 0
        else:
            enconded_Xtest[i,j]= (enconded_Xtest[i,j] - means[j]) / standard_deviation[j]

# Step 10: For each Xtesti: calculate the Euclidean Distance between xtesti and Xtrain's vectors. USE THE FORMULA IN ELD PRINT!!!

# Step 11: Verify between K neirest-neighbors next to xtesti, if the majority of them is p or e

# Step 12: Print the label of the majority obtained on the last step ('p' or 'e')
