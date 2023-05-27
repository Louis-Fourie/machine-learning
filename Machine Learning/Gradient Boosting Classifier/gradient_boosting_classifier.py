import pandas as pd
import gzip
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# DATA IMPORT

# open the JSON.gz file & read the compressed JSON data
with gzip.open('poland-bankruptcy-data-2009.json.gz', 'rt') as f:
    json_str = f.read()

# parse the JSON data
json_obj = json.loads(json_str)

# extract the "data" field from the JSON data
extracted_json = json_obj["data"]

# create a DataFrame from the list of dictionaries
df = pd.DataFrame(extracted_json)

# DATA PREPARATION

# feature/target & train/test split
X = df.drop(['bankrupt', 'company_id'], axis=1)
y = df['bankrupt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# preprocess data
imputer = SimpleImputer() # impute missing values
scaler = StandardScaler() # standardize features
resampler = SMOTE(random_state=42) # address class imbalance

def preprocess_train(X, y):
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    X_resampled, y_resampled = resampler.fit_resample(X_scaled, y)
    return X_resampled, y_resampled

X_train_preprocessed, y_train_resampled = preprocess_train(X_train, y_train)

def preprocess_test(X):
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    return X_scaled

X_test_preprocessed = preprocess_test(X_test)

# MODEL BUILDING

# tune hyperparameters
param_grid={'learning_rate': [0.1, 0.05, 0.01], 'n_estimators': [100, 500, 1000], 'max_depth': [3, 5, 7]}

tuner = GridSearchCV(GradientBoostingClassifier(), param_grid, scoring='accuracy', n_jobs=16, verbose=3)
tuner.fit(X_train_preprocessed, y_train_resampled)

# extract the model with the best hyperparameters
classifier = tuner.best_estimator_

# save the trained model to a file
joblib.dump(classifier, 'classifier.pkl')

# MODEL EVALUATION (X_train_preprocessed, y_train_resampled, X_test_preprocessed, y_test)

classifier = joblib.load('classifier.pkl')

# TRAIN evaluation metrics
train_y_pred = classifier.predict(X_train_preprocessed)
train_accuracy = accuracy_score(y_train_resampled, train_y_pred)
train_precision = precision_score(y_train_resampled, train_y_pred)
train_recall = recall_score(y_train_resampled, train_y_pred)
train_f1 = f1_score(y_train_resampled, train_y_pred)

print(f'Train Accuracy: {train_accuracy:.3f}')
print(f'Train Precision: {train_precision:.3f}')
print(f'Train Recall: {train_recall:.3f}')
print(f'Train F1-score: {train_f1:.3f}')

# TEST evaluation metrics
test_y_pred = classifier.predict(X_test_preprocessed)
test_accuracy = accuracy_score(y_test, test_y_pred)
test_precision = precision_score(y_test, test_y_pred)
test_recall = recall_score(y_test, test_y_pred)
test_f1 = f1_score(y_test, test_y_pred)

print(f'Test Accuracy: {test_accuracy:.3f}')
print(f'Test Precision: {test_precision:.3f}')
print(f'Test Recall: {test_recall:.3f}')
print(f'Test F1-score: {test_f1:.3f}')