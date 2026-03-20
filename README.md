# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Data Collection
Collect a dataset of emails labeled as Spam or Not Spam (Ham).

Step 2: Data Preprocessing
Remove unwanted characters (punctuation, symbols).
Convert all text to lowercase.
Remove stop words (e.g., “the”, “is”, “and”).
Perform stemming or lemmatization.

Step 3: Feature Extraction
Convert text data into numerical format using:
TF-IDF (Term Frequency–Inverse Document Frequency) or
Bag of Words model.

Step 4: Dataset Splitting
Split dataset into:
Training set (e.g., 80%)
Testing set (e.g., 20%)

Step 5: Model Selection
Choose Support Vector Machine (SVM) classifier.
Select kernel function:
Linear / Polynomial / RBF (commonly Linear for text classification).

Step 6: Model Training
Train the SVM model using training data:
model.fit(X_train, y_train)

Step 7: Model Prediction
Predict labels for test data:
y_pred = model.predict(X_test)

Step 8: Model Evaluation
Evaluate performance using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix

Step 9: Output Result
Classify emails as:
Spam
Not Spam (Ham)
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SUDARSAN.A
RegisterNumber:  212224220111
*/
# Detect file encoding
import chardet

with open('spam.csv', 'rb') as file:
    encoding = chardet.detect(file.read(100000))
print(encoding)


# Load dataset
import pandas as pd

data = pd.read_csv('spam.csv', encoding='Windows-1252')

print(data.head())
print(data.isnull().sum())


# Define features and labels
X = data['v2']   # email text
y = data['v1']   # spam / ham


# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# Convert text to numerical data
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Train model
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)

print("Predicted values:", y_pred)


# Accuracy
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
```

## Output:
<img width="929" height="45" alt="image" src="https://github.com/user-attachments/assets/41376462-c48c-4799-a023-5491e26d0261" />

<img width="958" height="493" alt="Screenshot 2026-03-20 133033" src="https://github.com/user-attachments/assets/ae200329-734c-4f67-8432-735974fed6a9" />

<img width="704" height="63" alt="image" src="https://github.com/user-attachments/assets/dcd379d9-1848-4738-94ac-9dcb06f9343c" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
