#Lab1--------------------------------------------------------

#1-----------------------------------------------------------
import pandas as pd

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
#2-----------------------------------------------------------
df = train_df.copy()
print("-" * 100)
print("First 5 colums: ")
print("-" * 100)
print(df.head())
print("-" * 100)
print("Dataset info: ")
print("-" * 100)
print(df.info)
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
df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Cabin'] = df['Cabin'].fillna('U')

print(df.isnull().sum())
print("-" * 100)

#5-----------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

#6-----------------------------------------------------------
df = pd.get_dummies(df, columns =['Sex', 'Embarked'], drop_first = True)

#Final
print("Final dataset:")
print(df.head())
print("-" * 100)
print("dataset types:")
print(df.dtypes)

#Lab2--------------------------------------------------------

'''1. Разделить датасет, подготовленный на первой лабораторной работе, на
обучающую и тестовую выборки
2. Решить задачу регрессии для одного из непрерывных признаков в
датасете
3. Оценить работу регрессионной модели. При плохих результатах
подумать как можно его улучшить
4. Решить задачу классификации
5. Оценить работу классификационной модели. При плохих результатах
подумать как можно его улучшить
6. Выгрузить результат работы на Github'''

#1-----------------------------------------------------------
from sklearn.model_selection import train_test_split

X = df.drop(['Fare', 'Name', 'Ticket', 'Cabin'], axis = 1)
Y = df['Fare']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 42)

X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size = 0.4, random_state = 42)
print("-" * 100)
print("Test info: ")
print(X_train.head())
print("-" * 100)
print(X_test.head())
print("-" * 100)
print(Y_train.head())
print("-" * 100)
print(Y_test.head())
print("-" * 100)
print(Y_val.head())
print("-" * 100)

#2-----------------------------------------------------------
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
Y_pred_test = linear_model.predict(X_test)

print("Y predict: ")
print(Y_pred_test)
print("-" * 100)

#3-----------------------------------------------------------
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(Y_test, Y_pred_test)
print(F"MSE = {MSE} ")
print("-" * 100)

from sklearn.metrics import root_mean_squared_error
RMSE = root_mean_squared_error(Y_test, Y_pred_test)
print(f"RMSE = {RMSE} ")
print("-" * 100)

from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(Y_test, Y_pred_test)
print(f"MAE = {MAE} " )
print("-" * 100)

#4-----------------------------------------------------------
from sklearn.linear_model import LogisticRegression

X2 = df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis = 1)
Y2 = df['Survived']
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size = 0.4, random_state = 42)
X_test2, X_val2, Y_test2, Y_val2 = train_test_split(X_test2, Y_test2, test_size = 0.4, random_state = 42)
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train2, Y_train2)
y_pred_test2 = logreg_model.predict(X_test2)
print("Y predict: ")
print(y_pred_test2)
print("-" * 100)

#5-----------------------------------------------------------
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test2, y_pred_test2)
print(f"Accuracy = {accuracy} ")
print("-" * 100)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test2, y_pred_test2)
print("Confusion Matrix:")
print(cm)
print("-" * 100)






