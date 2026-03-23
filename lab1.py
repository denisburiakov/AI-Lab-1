#Lab1--------------------------------------------------------

#1-----------------------------------------------------------
import pandas as pd

heart_df = pd.read_csv('heart.csv')
#2-----------------------------------------------------------
df = heart_df.copy()
print("-" * 100)
print("First 5 columns: ")
print("-" * 100)
print(df.head())
print("-" * 100)
print("Dataset info: ")
print("-" * 100)
df.info()
print("-" * 100)
print("Dataset statistics: ")
print("-" * 100)
print(df.describe())
print("-" * 100)

#3-----------------------------------------------------------
Nan_matrix = df.isnull()
print(Nan_matrix.sum())
print("-" * 100)

#4-----------------------------------------------------------
df['age'] = df['age'].fillna(df['age'].median())
df['trestbps'] = df['trestbps'].fillna(df['trestbps'].median())
df['chol'] = df['chol'].fillna(df['chol'].median())

print(df.isnull().sum())
print("-" * 100)

#5-----------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = scaler.fit_transform(df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])

#6-----------------------------------------------------------
df = pd.get_dummies(df, columns=['cp', 'thal', 'slope'], drop_first=True)

#Final
print("Final dataset:")
print(df.head())
print("-" * 100)

#Lab2--------------------------------------------------------

#1-----------------------------------------------------------
from sklearn.model_selection import train_test_split

# РЕГРЕССИЯ: предсказываем thalach (макс. пульс)
# Убираем таргет классификации (target) и саму целевую переменную регрессии
X = df.drop(['thalach', 'target'], axis=1)
Y = df['thalach']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# Валидационная выборка
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.4, random_state=42)

#2-----------------------------------------------------------
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
Y_pred_test = linear_model.predict(X_test)

#3-----------------------------------------------------------
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

MSE = mean_squared_error(Y_test, Y_pred_test)
RMSE = root_mean_squared_error(Y_test, Y_pred_test)
MAE = mean_absolute_error(Y_test, Y_pred_test)

print(f"MSE = {MSE}")
print(f"RMSE = {RMSE}")
print(f"MAE = {MAE}")
print("-" * 100)

#4-----------------------------------------------------------
from sklearn.linear_model import LogisticRegression

X2 = df.drop(['target'], axis=1)
Y2 = df['target']

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.4, random_state=42)
X_test2, X_val2, Y_test2, Y_val2 = train_test_split(X_test2, Y_test2, test_size=0.4, random_state=42)

logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train2, Y_train2)
y_pred_test2 = logreg_model.predict(X_test2)

#5-----------------------------------------------------------
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(Y_test2, y_pred_test2)
print(f"Accuracy = {accuracy}")
print("-" * 100)

cm = confusion_matrix(Y_test2, y_pred_test2)
print("Confusion Matrix:")
print(cm)
print("-" * 100)