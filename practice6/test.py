import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.model_selection as sms
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

#import sklearn as skl

def set_graph_size(width=8, height=6):
  """Установка размеров изображения"""
  plt.rcParams["figure.figsize"] = width, height

#1
boston_dataframe = pd.DataFrame(pd.read_csv('boston.csv'))

#2-3

boston_dataframe.head()

boston_dataframe.info()

#У всех данных тип - float64 (числовой), пропущенных значений вроде нет

#4
boston_dataframe.corr()

#5
plt.figure(figsize=(10,10))
sns.heatmap(boston_dataframe.corr())

set_graph_size(20,20)

#-0.737663
plt.subplot(3,2,1)
plt.scatter(x='MEDV', y='LSTAT',
            data=boston_dataframe)
plt.title('MEDV/LSTAT. cor=-0.737663', fontsize=30)
plt.xlabel('Медианная цена недвижимости (тыс. $)', fontsize=20)
plt.ylabel('% населения с низким соц.статусом', fontsize=12)

#0.695360
plt.subplot(3,2,2)
plt.scatter(x='MEDV', y='RM',
            data=boston_dataframe)
plt.title('MEDV/RM. cor=0.695360', fontsize=30)
plt.xlabel('Медианная цена недвижимости (тыс. $)', fontsize=20)
plt.ylabel('Кол-во комнат в доме', fontsize=12)

#-0.483725
plt.subplot(3,2,3)
plt.scatter(x='MEDV', y='INDUS',
            data=boston_dataframe)
plt.title('MEDV/INDUS. cor=-0.483725', fontsize=30)
plt.xlabel('Медианная цена недвижимости (тыс. $)', fontsize=20)
plt.ylabel('% площадей, не связанных с розничной торговлей', fontsize=12)

#-0.427321
plt.subplot(3,2,4)
plt.scatter(x='MEDV', y='NOX',
            data=boston_dataframe)
plt.title('MEDV/NOX. cor=-0.427321', fontsize=30)
plt.xlabel('Медианная цена недвижимости (тыс. $)', fontsize=20)
plt.ylabel('Качество воздуха (концентрация оксидов азота)', fontsize=12)

#-0.468536
plt.subplot(3,2,5)
plt.scatter(x='MEDV', y='TAX',
            data=boston_dataframe)
plt.title('MEDV/TAX. cor=-0.468536', fontsize=30)
plt.xlabel('Медианная цена недвижимости (тыс. $)', fontsize=20)
plt.ylabel('Налоги (ставка налога на 10 000 долларов США)', fontsize=12)

#-0.507787
plt.subplot(3,2,6)
plt.scatter(x='MEDV', y='PTRATIO',
            data=boston_dataframe)
plt.title('MEDV/PTRATIO. cor=-0.507787', fontsize=30)
plt.xlabel('Медианная цена недвижимости (тыс. $)', fontsize=20)
plt.ylabel('Соотношение кол-ва учеников и учителей', fontsize=12)

plt.tight_layout()

# Используем для стандартизации значений, поскольку у нас есть большой разброс по значениям, а также мы измеряем в разных единицах измерения
#scaler = StandardScaler()

factors_array = pd.DataFrame({'LSTAT': boston_dataframe['LSTAT'],
                            'RM': boston_dataframe['RM'],
                            'INDUS': boston_dataframe['INDUS'],
                            'NOX': boston_dataframe['NOX'],
                            'TAX': boston_dataframe['TAX'],
                            'PTRATIO': boston_dataframe['PTRATIO']})

target_array = pd.DataFrame({'MEDV': boston_dataframe['MEDV']})

X_train, X_test, y_train, y_test = sms.train_test_split(factors_array, target_array, test_size = 0.2, random_state=42)

print(f"Train shape: {X_train.shape[0]}")
print(f"Test shape: {X_test.shape[0]} ")

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)

print(f"Coefficient (slope): {lin_reg_model.coef_}")
print(f"Intercept: {lin_reg_model.intercept_}")

y_train_pred = lin_reg_model.predict(X_train)
y_test_pred = lin_reg_model.predict(X_test)

#print(y_train_pred, y_test_pred)
#print(y_test)

r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f'r2_train: {r2_train}, r2_test: {r2_test}')
print(f'rmse_train: {rmse_train}, rmse_test: {rmse_test}')

plt.figure(figsize=(10, 4))
sns.boxplot(x=boston_dataframe['MEDV'])
plt.title('Boxplot for MEDV')
plt.xlabel('MEDV')
plt.show()

boston_dataframe_f = boston_dataframe[boston_dataframe['MEDV'] < 50.0]

factors_array = pd.DataFrame({'LSTAT': boston_dataframe_f['LSTAT'],
                            'RM': boston_dataframe_f['RM'],
                            'INDUS': boston_dataframe_f['INDUS'],
                            'NOX': boston_dataframe_f['NOX'],
                            'TAX': boston_dataframe_f['TAX'],
                            'PTRATIO': boston_dataframe_f['PTRATIO']})

target_array = pd.DataFrame({'MEDV': boston_dataframe_f['MEDV']})

print(target_array, factors_array)
#display(target_array.info())
#display(factors_array.info())

X_train_f, X_test_f, y_train_f, y_test_f = sms.train_test_split(factors_array, target_array, test_size = 0.2, random_state=42)

print(f"Train shape: {X_train_f.shape[0]}")
print(f"Test shape: {X_test_f.shape[0]} ")

lin_reg_model_f = LinearRegression()
lin_reg_model_f.fit(X_train_f, y_train_f)

print(f"Coefficients: {lin_reg_model_f.coef_}")
print(f"Intercept: {lin_reg_model_f.intercept_}")

y_train_pred_f = lin_reg_model_f.predict(X_train_f)
y_test_pred_f = lin_reg_model_f.predict(X_test_f)

#print(y_train_pred, y_test_pred)
#print(y_test)

r2_train_f = r2_score(y_train_f, y_train_pred_f)
rmse_train_f = np.sqrt(mean_squared_error(y_train_f, y_train_pred_f))
r2_test_f = r2_score(y_test_f, y_test_pred_f)
rmse_test_f = np.sqrt(mean_squared_error(y_test_f, y_test_pred_f))

print(f'r2_train_f: {r2_train_f}, r2_test_f: {r2_test_f}')
print(f'rmse_train_f: {rmse_train}, rmse_test_f: {rmse_test_f}')

plt.figure(figsize=(10, 4))
sns.boxplot(x=boston_dataframe_f['MEDV'])
plt.title('Boxplot for MEDV')
plt.xlabel('MEDV')
plt.show()
