import numpy as np

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
        return [self.label_to_index.get(label, -1) for label in labels]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_classifier(Xtrain, Ytrain, Xtest, k):
    # Encode Xtrain
    encoder = CustomLabelEncoder()
    all_values = [item for sublist in Xtrain for item in sublist if item != '?']
    encoder.fit(all_values)
    
    encoded_Xtrain = [encoder.transform(row) for row in Xtrain]
    encoded_Xtrain = np.array(encoded_Xtrain, dtype=float)
    
    # Handle missing values in Xtrain
    for i in range(encoded_Xtrain.shape[1]):
        col = encoded_Xtrain[:, i]
        col[col == -1] = np.mean(col[col != -1])
    
    # Normalize Xtrain
    means = np.mean(encoded_Xtrain, axis=0)
    std_devs = np.std(encoded_Xtrain, axis=0)
    normalized_Xtrain = (encoded_Xtrain - means) / np.where(std_devs != 0, std_devs, 1)

    # Encode and normalize Xtest
    encoded_Xtest = [encoder.transform(row) for row in Xtest]
    encoded_Xtest = np.array(encoded_Xtest, dtype=float)
    
    # Handle missing values in Xtest
    for i in range(encoded_Xtest.shape[1]):
        col = encoded_Xtest[:, i]
        col[col == -1] = means[i]
    
    normalized_Xtest = (encoded_Xtest - means) / np.where(std_devs != 0, std_devs, 1)

    # Predict for each test sample
    predictions = []
    for test_sample in normalized_Xtest:
        distances = [euclidean_distance(test_sample, train_sample) for train_sample in normalized_Xtrain]
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = [Ytrain[i] for i in k_nearest_indices]
        
        # Use a weighted voting system
        weights = 1 / (np.array(distances)[k_nearest_indices] + 1e-5)
        label_weights = {}
        for label, weight in zip(k_nearest_labels, weights):
            label_weights[label] = label_weights.get(label, 0) + weight
        
        prediction = max(label_weights, key=label_weights.get)
        predictions.append(prediction)
    
    return predictions
def test_knn_classifier():
    # Input data
    k_neighbors = 15
    Ntrain, Ntest = 36, 12
    
    Xtrain = [
        "s f g f n f c n g e e s s w w p w o p k v u".split(),
        "f y w t p f c n w e e s s w w p w o p k v g".split(),
        "x f n t n f c b n t b s s g g p w o p k v d".split(),
        "f y g t n f c b u t b s s w g p w o p k y d".split(),
        "f y e t n f c b w t b s s g p p w o p n v d".split(),
        "x s p f c f w n u e b s s w w p w o p k s d".split(),
        "f s n t p f c n n e e s s w w p w o p k s g".split(),
        "f f g t n f c b w t b s s p g p w o p n y d".split(),
        "k y n f n f c n w e ? k y w n p w o e w v d".split(),
        "k s n f n a c b o e ? s s o o p n o p n c l".split(),
        "f y e t n f c b u t b s s w g p w o p n v d".split(),
        "x s e f s f c n b t ? k s w p p w o e w v l".split(),
        "x f g f n f w b h t e f s w w p w o e k a g".split(),
        "f s g t f f c b h t b s s w w p w o p h v u".split(),
        "x y y t l f c b n e r s y w w p w o p n y g".split(),
        "k y e f f f c n b t ? k s p w p w o e w v d".split(),
        "x f g t n f c b w t b s s w g p w o p k y d".split(),
        "f s n f s f c n b t ? s k p p p w o e w v l".split(),
        "f y p t n f c b w e ? s s w e p w t e w c w".split(),
        "f y w t p f c n k e e s s w w p w o p n v u".split(),
        "x s n t p f c n k e e s s w w p w o p k s u".split(),
        "f f g t n f c b n t b s s g w p w o p k y d".split(),
        "x y n t n f c b w e ? s s w w p w t e w c w".split(),
        "x y y t l f c b w e c s s w w p w o p n s g".split(),
        "f y g f f f c b h e b k k p b p w o l h v d".split(),
        "x f g t n f c b p t b s s p p p w o p k y d".split(),
        "b f g f n f w b p e ? s k w w p w t p w n g".split(),
        "x s n t p f c n n e e s s w w p w o p k s g".split(),
        "x s y t l f c b k e c s s w w p w o p k n m".split(),
        "x y n f s f c n b t ? s k w w p w o e w v l".split(),
        "x f e t n f c b w t b s s p g p w o p k y d".split(),
        "b s g f n f w b g e ? k k w w p w t p w s g".split(),
        "x s w t a f c b n e c s s w w p w o p k s g".split(),
        "f y w f n f c n p e ? s f w w p w o f h y d".split(),
        "f y n f s f c n b t ? s s w w p w o e w v p".split(),
        "f f e t n f c b p t b s s p g p w o p k y d".split(),
        # ... (include all 36 training samples here)
    ]
    
    Ytrain = "e p e e e p p e p e e p e p e p e p e p p e e e p e e p e p e e e e p e".split()
    
    Xtest = [
        "x s w f n f w b n t e f f w w p w o e k s g".split(),
        "x f n t n f c b p t b s s w w p w o p k v d".split(),
        "x s w t f f c b h t b f f w w p w o p h s g".split(),
        "x f y f f f c b g e b k k p n p w o l h y p".split(),
        "k f w f n f w b p e ? k k w w p w t p w s g".split(),
        "f y n f f f c n b t ? s k w w p w o e w v p".split(),
        "x y g f f f c b p e b k k p n p w o l h y g".split(),
        "b f g f n f w b g e ? k s w w p w t p w s g".split(),
        "x y w t a f c b g e c s s w w p w o p n n g".split(),
        "f y y f f f c b h e b k k p p p w o l h y g".split(),
        "f y n t n f c b p t b s s w w p w o p k y d".split(),
        "x s e f y f c n b t ? s s w w p w o e w v p".split(),

        # ... (include all 12 test samples here)
    ]
    
    expected_outputs = "p e e e e p e p e e e p".split()
    
    # Run the classifier
    predictions = knn_classifier(Xtrain, Ytrain, Xtest, k_neighbors)
    
    # Compare predictions with expected outputs
    correct = sum(p == e for p, e in zip(predictions, expected_outputs))
    accuracy = correct / len(expected_outputs)
    
    print(f"Predictions: {predictions}")
    print(f"Expected:    {expected_outputs}")
    print(f"Accuracy: {accuracy:.2f}")
    
    # Print individual results
    for i, (pred, exp) in enumerate(zip(predictions, expected_outputs), 1):
        print(f"Sample {i}: Predicted = {pred}, Expected = {exp}, {'Correct' if pred == exp else 'Incorrect'}")
    
    return predictions, expected_outputs, accuracy

# Run the test
test_knn_classifier()