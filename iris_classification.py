from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Step 1: Data Preparation - Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Model Selection - KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Step 3: Train the KNN Model
knn.fit(X_train, y_train)

# Step 4: Evaluate the Model Performance
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 5: Classify New Samples
def classify_iris(sample):
    sample = scaler.transform([sample])  # Standardize input
    prediction = knn.predict(sample)
    return iris.target_names[prediction[0]]

# Example: Classify a new sample
new_sample = [5.1, 3.5, 1.4, 0.2]  # Example measurements
predicted_class = classify_iris(new_sample)
print(f'Predicted Class: {predicted_class}')
