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

## Devolopment Changes and Progression

- Aux1.py, and auxiliar file, was created to test new implementations without impacting the main file directly
- During the development some imprecisions were identified and fixed. To improve testing, a new testing.py was created to autotest the main program(mushrooms.py)
- An accuracy file was created to calculate the accuracy of the algorithm

## Final Thoughts

The algorithm is functional and running. Though it does not deal with user errors, it has an accuracy of approximately 83%.
