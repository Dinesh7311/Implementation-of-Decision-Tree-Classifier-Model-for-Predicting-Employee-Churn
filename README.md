# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required packages
2. load and print the present data
3. print placement and salary data
4. check for null and duplicate values
5. perform logistic regression
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Dinesh s
RegisterNumber: 24900464
# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Employee.csv")

# Display initial rows and dataset information
print("Dataset Preview:")
print(data.head())
print("\nDataset Information:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())
print("\nValue Counts for 'left':")
print(data["left"].value_counts())

# Encode categorical variables (if applicable)
le = LabelEncoder()
if "salary" in data.columns:
    data["salary"] = le.fit_transform(data["salary"])
else:
    print("Warning: 'salary' column not found in dataset.")

# Define Features (X) and Target (y)
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours"]]
y = data["left"]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Initialize and train the Decision Tree Classifier
dt = DecisionTreeClassifier(criterion="entropy", random_state=100)
dt.fit(x_train, y_train)

# Predict on the test set
y_pred = dt.predict(x_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Make a sample prediction (adjusted for correct feature count)
sample_data = [[0.5, 0.8, 9, 260]]  # Ensure this matches the feature set used in 'x'
sample_prediction = dt.predict(sample_data)
print(f"\nPrediction for sample data {sample_data}: {sample_prediction}")

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=x.columns, class_names=['Stayed', 'Left'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

## Output:
![ex 8-1](https://github.com/user-attachments/assets/59c5a090-fa6a-4414-8be2-24b4f9d4fef3)
![ex 8-2](https://github.com/user-attachments/assets/7bebd641-6cde-4bd7-bd6b-cfa8955bf577)
![Uploading ex-8-3.jpg…]()





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
