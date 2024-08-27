# mushroomspy

## Developing a machine learning project with python for University =D

### Steps:

- Step 1: Read the values K, Ntrain, Ntest
- Step 2: Read Xtrain, a matrix of Ntrain rows(mushrooms) columns(attributes)
- Step 3: Convert the characters in numbers
- Step 4: Normalize values (vector μ)(vector σ)
- Step 5: For each attribute, substract from the mean (all attributes, 1 row) and divide by the standard deviantion
- Step 6: Label Ytrain, an array of Ntrain elements, row(p or e) and D columns
- Step 7: Read Ytrain, a matrix of Ntrain rows(mushrooms) columns(attributes)
- Step 8: Convert the characters in numbers (Ytrain)
- Step 9: Utilize the same vectors from Xtrain (vector μ)(vector σ)
- Step 10: For each xtesti: calculate the Euclidean Distance between xtesti and Xtrain's vectors.
- Step 11: Verify between K neirest-neighbors next to xtesti, if the majority of them is p or e
- Step 12: Print the label of the majority obtained on the last step ('p' or 'e')

Mushrooms.
