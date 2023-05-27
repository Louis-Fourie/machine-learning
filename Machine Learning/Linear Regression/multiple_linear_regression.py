import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# IMPORT
path = 'Datasets/50_Startups.csv'
data = pd.read_csv(path)

# SPLIT
X = data.drop('Profit', axis=1)
y = data['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STANDARDIZE
numerical_columns = ['R&D Spend', 'Administration', 'Marketing Spend']
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# ENCODE
categorical_columns = ['State']
encoder = OneHotEncoder(cols=categorical_columns, use_cat_names=True)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# HEATMAP
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
data = pd.concat([train_data, test_data], axis=0)

def corr_matrix(data, set):
    corr_matrix = data.corr()
    corr_matrix = corr_matrix.drop(corr_matrix.index[6])
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix[['Profit']], annot=True, cmap="coolwarm", fmt=".2f")
    if set == 'train':
        plt.title("Correlation Heatmap - Training Set")
        plt.savefig('Heatmaps/train.png')
    elif set == 'test':
        plt.title("Correlation Heatmap - Testing Set")
        plt.savefig('Heatmaps/test.png')
    elif set == 'combined':
        plt.title("Correlation Heatmap - Training Set & Testing Set COMBINED")
        plt.savefig('Heatmaps/combined.png')
    

corr_matrix(train_data, 'train')
corr_matrix(test_data, 'test')
corr_matrix(data, 'combined')

# DROP FEATURES
X_train = X_train.drop(['State_Florida', 'State_California', 'State_New York'], axis=1)
X_test = X_test.drop(['State_Florida', 'State_California', 'State_New York'], axis=1)

# TRAIN
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training R-squared score:", train_r2)
print("Testing R-squared score:", test_r2)