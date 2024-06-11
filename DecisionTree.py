import csv
from collections import Counter

# Global variable for the training set size
TRAIN_SIZE_PERCENTAGE = 0.02

class DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y):
        # Base case: if all labels are the same, return leaf node with that label
        if len(set(y)) == 1:
            return {'decision': y[0]}

        # Find the best feature and split point
        best_split = self._find_best_split(X, y)

        # Split the data based on the best split
        feature_index, split_value = best_split
        left_X, left_y, right_X, right_y = self._split_dataset(X, y, feature_index, split_value)

        # Recursive call to grow left and right subtrees
        left_subtree = self._grow_tree(left_X, left_y)
        right_subtree = self._grow_tree(right_X, right_y)

        # Construct the node
        return {'feature_index': feature_index,
                'split_value': split_value,
                'left': left_subtree,
                'right': right_subtree}

    def _find_best_split(self, X, y):
        num_features = len(X[0])
        best_gini = float('inf')
        best_split = None

        for feature_index in range(num_features):
            feature_values = set(X[i][feature_index] for i in range(len(X)))

            for value in feature_values:
                left_X, left_y, right_X, right_y = self._split_dataset(X, y, feature_index, value)
                gini = self._gini_impurity(left_y, right_y)

                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, value)

        return best_split

    def _split_dataset(self, X, y, feature_index, value):
        left_X, left_y, right_X, right_y = [], [], [], []

        for i in range(len(X)):
            if X[i][feature_index] < value:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return left_X, left_y, right_X, right_y

    def _gini_impurity(self, left_y, right_y):
        total_samples = len(left_y) + len(right_y)
        p_left = len(left_y) / total_samples
        p_right = len(right_y) / total_samples

        gini_left = 1 - sum((Counter(left_y).get(label, 0) / len(left_y))**2 for label in set(left_y))
        gini_right = 1 - sum((Counter(right_y).get(label, 0) / len(right_y))**2 for label in set(right_y))

        gini = p_left * gini_left + p_right * gini_right
        return gini

    def predict(self, X):
        return [self._predict_single(x, self.tree) for x in X]

    def _predict_single(self, x, tree):
        if 'decision' in tree:
            return tree['decision']

        feature_index = tree['feature_index']
        split_value = tree['split_value']

        if x[feature_index] < split_value:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])

# Load the CSV file
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        dataset = list(reader)
    return dataset

# Convert string data to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = {value: index for index, value in enumerate(unique)}
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Convert string data to integers
filename = 'Problem4DecisionTreeData.CSV'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_int(dataset, i)

# Split dataset into features and target
X = [list(map(int, row[:-1])) for row in dataset]
y = [row[-1] for row in dataset]

# Determine the size of the training set
train_size = int(len(X) * TRAIN_SIZE_PERCENTAGE)

# Split the dataset into training and testing sets
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Initialize and train the decision tree
clf = DecisionTree()
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Calculate accuracy
accuracy = sum(predictions[i] == y_test[i] for i in range(len(y_test))) / len(y_test)
print("Accuracy:", accuracy)
