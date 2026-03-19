
'''1. Загрузить данные с Kaggle – выбрать датасет (например, Титаник)
2. Вывести с помощью python данные из датасета на экран
3. Получить количество пропущенных значений для каждого столбца в
датасетах
4. Заполнить пропущенные значения в датасете модой/медианой/средним
значением и показать, что они действительно были заполнены
5. Провести нормализацию данных
6. Провести преобразование категориальных данных так, чтобы в
будущем это не привело к переобучению модели машинного обучения
и датасет имел удобоваримый вид
7. Выгрузить код и результаты на гитхаб (сбросить ссылку
преподавателю)'''
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

