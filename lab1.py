#Lab1--------------------------------------------------------

#1-----------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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








