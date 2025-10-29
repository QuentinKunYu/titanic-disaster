# main.R

# Load necessary libraries quietly
library(readr)
library(dplyr)

cat("--- Titanic Survival Prediction (R) ---\n")

# --- Phase 1: Load and Preprocess Training Data ---
cat("\n[1] Loading and Preprocessing Training Data\n")
tryCatch({
    train_df <- read_csv("src/data/train.csv", show_col_types = FALSE)
    cat("Train.csv loaded successfully.\n")
}, error = function(e) {
    cat("Error: train.csv not found. Make sure it is in 'src/data/'.\n")
    quit(status = 1) # Exit script if file not found
})

# Make a copy for preprocessing
data <- train_df

# Preprocess training data
cat("Preprocessing training data...\n")
cat("- Filling missing 'Age' values with the median.\n")
median_age <- median(data$Age, na.rm = TRUE)
data$Age[is.na(data$Age)] <- median_age

cat("- Converting 'Sex' column to numeric (male: 0, female: 1).\n")
data$Sex <- ifelse(data$Sex == "male", 0, 1) # Directly convert to 0/1

cat("- Filling missing 'Fare' values with the median.\n")
median_fare <- median(data$Fare, na.rm = TRUE)
data$Fare[is.na(data$Fare)] <- median_fare

cat("- Filling missing 'Embarked' values with the most common port.\n")

get_mode <- function(v) {
  uniqv <- unique(v[!is.na(v)])
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
most_common_port <- get_mode(data$Embarked)
data$Embarked[is.na(data$Embarked)] <- most_common_port

cat("- Converting 'Embarked' column to numeric using one-hot encoding.\n")
data$Embarked <- as.factor(data$Embarked) # Ensure it's a factor first
embarked_dummies <- model.matrix(~ Embarked - 1, data = data) # Create dummy vars (e.g., EmbarkedC)
# Rename columns to match Python's output for consistency
colnames(embarked_dummies) <- paste0("Embarked_", levels(data$Embarked))
data <- cbind(data, embarked_dummies) # Combine dummies with original data
cat("Training data preprocessing complete.\n")


# --- Phase 2: Train the Model ---
cat("\n[2] Building and Training the Model\n")
# Define features including the numeric Sex and dummy Embarked variables
features <- c('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S')

# Create the formula string for glm
formula_str <- paste("Survived ~", paste(features, collapse = " + "))
formula <- as.formula(formula_str)

cat(paste("Starting model training with formula:", formula_str, "\n"))

# Train the logistic regression model 
model <- glm(formula, data = data, family = binomial(link = "logit"), control = glm.control(maxit = 100))
cat("Model training is complete.\n")


# --- Phase 3: Measure Accuracy on Training Set ---
cat("\n[3] Measuring Model Accuracy on Training Data\n")
# Predict probabilities on the training data
train_probabilities <- predict(model, newdata = data, type = "response")
# Convert probabilities to class predictions (0 or 1) using a 0.5 threshold
train_predictions <- ifelse(train_probabilities > 0.5, 1, 0)
# Calculate accuracy
accuracy <- mean(train_predictions == data$Survived, na.rm = TRUE)
cat(sprintf("Accuracy of the model on the training set: %.4f\n", accuracy))


# --- Phase 4: Load and Preprocess Test Data ---
cat("\n[4] Loading and Preprocessing Test Data\n")
tryCatch({
    test_df <- read_csv("src/data/test.csv", show_col_types = FALSE)
    cat("Test.csv loaded successfully.\n")
}, error = function(e) {
    cat("Error: test.csv not found in 'src/data/'.\n")
    quit(status = 1)
})

test_data <- test_df

# Preprocess test data using values derived *from the training data*
cat("Preprocessing test data using training data statistics...\n")
test_data$Age[is.na(test_data$Age)] <- median_age 
test_data$Sex <- ifelse(test_data$Sex == "male", 0, 1) 
test_data$Fare[is.na(test_data$Fare)] <- median_fare 
test_data$Embarked[is.na(test_data$Embarked)] <- most_common_port

# Convert Embarked to factor and ensure levels match training data before creating dummies
test_data$Embarked <- factor(test_data$Embarked, levels = levels(data$Embarked))
test_embarked_dummies <- model.matrix(~ Embarked - 1, data = test_data)
colnames(test_embarked_dummies) <- paste0("Embarked_", levels(data$Embarked))
test_data <- cbind(test_data, test_embarked_dummies)
cat("Test data preprocessing complete.\n")


# --- Phase 5: Make Predictions on the Test Set ---
cat("\n[5] Making Predictions on the Test Set\n")
cat("Generating predictions...\n")

X_test <- test_data[, features, drop = FALSE]

# Predict probabilities on the test set
test_probabilities <- predict(model, newdata = X_test, type = "response")
# Convert probabilities to class predictions
test_predictions <- ifelse(test_probabilities > 0.5, 1, 0)
cat("Predictions generated for the test set.\n")

# Format predictions for submission
submission <- data.frame(PassengerId = test_data$PassengerId, Survived = test_predictions)

print(submission)

