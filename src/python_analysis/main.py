import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("--- Titanic Survival Prediction (Python) ---")

# --- Phase 1: Load and Preprocess Training Data ---
print("\n[1] Loading and Preprocessing Training Data")
try:
    train_df = pd.read_csv("src/data/train.csv")
    print("Train.csv loaded successfully.")
except FileNotFoundError:
    print("Error: train.csv not found. Make sure it is in 'src/data/'.")
    exit()

data = train_df.copy()

# Preprocess training data
print("Preprocessing training data...")
print("- Filling missing 'Age' values with the median.")
data['Age'] = data['Age'].fillna(data['Age'].median())

print("- Converting 'Sex' column to numeric (male: 0, female: 1).")
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

print("- Filling missing 'Fare' values with the median.")
data['Fare'] = data['Fare'].fillna(data['Fare'].median())

print("- Filling missing 'Embarked' values with the most common port.")
most_common_port = data['Embarked'].mode()[0]
data['Embarked'] = data['Embarked'].fillna(most_common_port)

print("- Converting 'Embarked' column to numeric using one-hot encoding.")
embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked', dtype=int)
data = pd.concat([data, embarked_dummies], axis=1)
print("Training data preprocessing complete.")


# --- Phase 2: Train the Model ---
print("\n[2] Building and Training the Model")
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X_train = data[features]
y_train = data['Survived']

print(f"Starting model training with features: {features}")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("Model training is complete.")


# --- Phase 3: Measure Accuracy on Training Set ---
print("\n[3] Measuring Model Accuracy on Training Data")
train_predictions = model.predict(X_train)
accuracy = accuracy_score(y_train, train_predictions)
print(f"Accuracy of the model on the training set: {accuracy:.4f}")


# --- Phase 4: Load and Preprocess Test Data ---
print("\n[4] Loading and Preprocessing Test Data")
try:
    test_df = pd.read_csv("src/data/test.csv")
    print("Test.csv loaded successfully.")
except FileNotFoundError:
    print("Error: test.csv not found in 'src/data/'.")
    exit()

test_data = test_df.copy()

print("Preprocessing test data using values from the training data...")
test_data['Age'] = test_data['Age'].fillna(data['Age'].median()) 
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Fare'] = test_data['Fare'].fillna(data['Fare'].median()) 
test_most_common_port = data['Embarked'].mode()[0] 
test_data['Embarked'] = test_data['Embarked'].fillna(test_most_common_port)
test_embarked_dummies = pd.get_dummies(test_data['Embarked'], prefix='Embarked', dtype=int)
test_data = pd.concat([test_data, test_embarked_dummies], axis=1)
print("Test data preprocessing complete.")


# --- Phase 5: Make Predictions on the Test Set ---
print("\n[5] Making Predictions on the Test Set")
X_test = test_data[features]

print("Generating predictions...")
test_predictions = model.predict(X_test)
print("Predictions generated for the test set.")

# Format predictions for submission
submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_predictions
})

print("\nFirst 5 rows of the final prediction output:")
print(submission.head())
