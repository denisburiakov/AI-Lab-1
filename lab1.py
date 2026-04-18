# Lab1--------------------------------------------------------

# 1-----------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

heart_df = pd.read_csv('heart.csv')
# 2-----------------------------------------------------------
df = heart_df.copy()
print("-" * 100)
print("First 5 columns: ")
print("-" * 100)
print(df.head())
print("-" * 100)
print("Data set info: ")
print("-" * 100)
df.info()
print("-" * 100)
print("Dataset statistics: ")
print("-" * 100)
print(df.describe())
print("-" * 100)

# 3-----------------------------------------------------------
Nan_matrix = df.isnull()
print(Nan_matrix.sum())
print("-" * 100)

# 4-----------------------------------------------------------
df['age'] = df['age'].fillna(df['age'].median())
df['trestbps'] = df['trestbps'].fillna(df['trestbps'].median())
df['chol'] = df['chol'].fillna(df['chol'].median())

print(df.isnull().sum())
print("-" * 100)

# 5-----------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = scaler.fit_transform(
    df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])

# 6-----------------------------------------------------------
df = pd.get_dummies(df, columns=['cp', 'thal', 'slope'], drop_first=True)

# Final
print("Final dataset:")
print(df.head())
print("-" * 100)

# Lab2--------------------------------------------------------

# 1-----------------------------------------------------------
from sklearn.model_selection import train_test_split

X = df.drop(['thalach', 'target'], axis=1)
Y = df['thalach']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.4, random_state=42)

# 2-----------------------------------------------------------
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
Y_pred_test = linear_model.predict(X_test)

# 3-----------------------------------------------------------
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

MSE = mean_squared_error(Y_test, Y_pred_test)
RMSE = root_mean_squared_error(Y_test, Y_pred_test)
MAE = mean_absolute_error(Y_test, Y_pred_test)

print(f"MSE = {MSE}")
print(f"RMSE = {RMSE}")
print(f"MAE = {MAE}")
print("-" * 100)

# 4-----------------------------------------------------------
from sklearn.linear_model import LogisticRegression

X2 = df.drop(['target'], axis=1)
Y2 = df['target']

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.4, random_state=42)
X_test2, X_val2, Y_test2, Y_val2 = train_test_split(X_test2, Y_test2, test_size=0.4, random_state=42)

logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train2, Y_train2)
y_pred_test2 = logreg_model.predict(X_test2)

# 5-----------------------------------------------------------
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(Y_test2, y_pred_test2)
print(f"Accuracy = {accuracy}")
print("-" * 100)

cm = confusion_matrix(Y_test2, y_pred_test2)
print("Confusion Matrix:")
print(cm)
print("-" * 100)

# Lab3--------------------------------------------------------
# 2-----------------------------------------------------------

from sklearn.tree import DecisionTreeRegressor

dt_regressor_model = DecisionTreeRegressor()
dt_regressor_model.fit(X_train, Y_train)

Y_pres_dt = dt_regressor_model.predict(X_test)

mse_dt = mean_squared_error(Y_test, Y_pres_dt)
rmse_dt = root_mean_squared_error(Y_test, Y_pres_dt)
mae_dt = mean_absolute_error(Y_test, Y_pres_dt)
print("Decision Tree Regression Results:")
print("-" * 100)
print(f"MSE = {mse_dt}")
print(f"RMSE = {rmse_dt}")
print(f"MAE = {mae_dt}")
print("-" * 100)

# 3-----------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc

dt_classifier_model = DecisionTreeClassifier()
dt_classifier_model.fit(X_train2, Y_train2)

y_proba = dt_classifier_model.predict_proba(X_test2)

fpr, tpr, thresholds = roc_curve(Y_test2, y_proba[:, 1])
auc_metric = auc(fpr, tpr)
print("Decision Tree Classification Results:")
print("-" * 100)
print(f"ROC AUC Score: {auc_metric:.4f}")
print("-" * 100)
plt.plot(fpr, tpr, marker='o')
plt.ylim([0, 1.1])
plt.xlim([0, 1.1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')
plt.show()
