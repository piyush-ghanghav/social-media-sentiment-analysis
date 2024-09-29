import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from tkinter import messagebox

# Function to run the classifier and display results
def run_classifier(classifier, model_name):
    df = pd.read_csv('Sigmoid_output/sigmoid_output.csv')
    X = df.drop(['Sentiment_Binary'], axis=1)
    y = df['Sentiment_Binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
    # Set zero_division to handle undefined precision/recall for labels with no predicted samples
    report = classification_report(y_test, y_pred, zero_division=0)

    result_message = f"{model_name} Accuracy: {accuracy:.2f}\n\nClassification Report:\n{report}"
    messagebox.showinfo(f"{model_name} Result", result_message)

# SVM model with class_weight to handle class imbalance
def apply_svm(app):
    classifier = SVC(class_weight='balanced')  # Adjust for class imbalance
    run_classifier(classifier, "SVM")

# Decision Tree model
def apply_decision_tree(app):
    classifier = DecisionTreeClassifier()
    run_classifier(classifier, "Decision Tree")

# Random Forest model
def apply_random_forest(app):
    classifier = RandomForestClassifier()
    run_classifier(classifier, "Random Forest")

# Naive Bayes model
def naive_bayes(app):
    classifier = GaussianNB()
    run_classifier(classifier, "Naive Bayes")
