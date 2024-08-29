import pyautogui
import numpy as np
# Make a code to input in the other code =)
# =)

"""
Input:
1
33 11
f s g t f f c b w t b f s w w p w o p h v g
x f g t n f c b p t b s s w w p w o p k y d
f y n t n f c b u t b s s g w p w o p k y d
x s y t l f c b k e c s s w w p w o p k s g
b f w f n f w b g e ? s s w w p w t p w s g
s f n f n f c n g e e s s w w p w o p n v u
x f g f n f w b g e ? k s w w p w t p w s g
x s w t p f c n p e e s s w w p w o p n v g
f f e t n f c b p t b s s g g p w o p n v d
f s b t n f c b e e ? s s w e p w t e w c w
f y n t l f c b n e r s y w w p w o p k y p
x f g f c f c n n e b s s w w p w o p k v d
x y g f f f c b h e b k k p b p w o l h y g
f s b t f f c b p t b f f w w p w o p h s g
k y n f y f c n b t ? k s w w p w o e w v p
f s b t f f c b p t b f f w w p w o p h v g
x s w t p f c n n e e s s w w p w o p n v u
f y y f f f c b g e b k k n b p w o l h y p
s f n f n f c n g e e s s w w p w o p n y u
x y e t n f c b u t b s s p w p w o p n y d
f y p t n f c b g e b s s w w p w t p r v g
f s e f f f c n b t ? s s w w p w o e w v d
x s b t n f c b e e ? s s e e p w t e w c w
x y g t n f c b u t b s s g w p w o p k v d
x y y t a f c b n e r s y w w p w o p n s g
f y y f f f c b g e b k k p n p w o l h y g
f y n f y f c n b t ? s s w p p w o e w v p
x s b t f f c b p t b f s w w p w o p h v u
f y n f n f c b w e b y y n n p w t p w y p
x y n t a f c b n e r s y w w p w o p n s g
f s g f n f w b n t e f f w w p w o e k s g
f f g t n f c b w t b s s g w p w o p k y d
x y e f y f c n b t ? s s w w p w o e w v l
p
e
e
e
e
e
e
p
e
e
e
p
p
p
p
p
p
p
e
e
p
p
e
e
e
p
p
p
e
e
e
e
p
f f g f f f c b p e b k k n b p w o l h v d
b s y t l f c b w e c s s w w p w o p k n g
f f n t n f c b n t b s s w w p w o p n y d
x s w t a f w n p t b s s w w p w o p u v d
k y n f f f c n b t ? k k p p p w o e w v l
f y e f s f c n b t ? s s w w p w o e w v p
x y e t n f c b u t b s s p g p w o p k v d
f f n t n f c b p t b s s w w p w o p k y d
x s n t p f c n k e e s s w w p w o p n s u
f y w t p f c n w e e s s w w p w o p k s u
x f n t n f c b u t b s s w g p w o p k v d
"""

"""
Expected output:
p
e
e
e
p
p
e
e
p
p
e
"""
# Sample training and test data
training_data = np.array([
    [1, 2],  # Example label: 2 (even)
    [2, 3],  # Example label: 3 (odd)
    [3, 4],  # Example label: 4 (even)
    [5, 5]   # Example label: 5 (odd)
])

# Assume the labels are in the last column of the training_data
labels = training_data[:, -1]

test_data = np.array([
    [1, 1],
    [2, 2]
])

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Function to find K nearest neighbors and check majority even or odd
def find_k_nearest_neighbors_and_check(training_data, test_data, k):
    nearest_neighbors = []
    even_odd_majority = []

    for test_point in test_data:
        distances = []

        # Compute distances from test_point to all training points
        for train_point in training_data:
            distance = euclidean_distance(test_point, train_point)
            distances.append(distance)

        # Get the indices of the K smallest distances
        k_indices = np.argsort(distances)[:k]
        nearest_neighbors.append(k_indices)

        # Extract the labels of the K nearest neighbors
        k_labels = labels[k_indices]

        # Determine if the majority of the labels are even or odd
        even_count = np.sum(k_labels % 2 == 0)
        odd_count = k - even_count

        if even_count > odd_count:
            even_odd_majority.append("even")
        else:
            even_odd_majority.append("odd")

    return nearest_neighbors, even_odd_majority

# Set the value of K
k = 2

# Find K nearest neighbors and check for majority even or odd
k_nearest_neighbors, even_odd_majority = find_k_nearest_neighbors_and_check(training_data, test_data, k)

print("K nearest neighbors indices:", k_nearest_neighbors)
print("Majority of K nearest neighbors:", even_odd_majority)