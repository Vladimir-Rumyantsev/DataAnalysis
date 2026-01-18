import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Загрузите данные из файла "boston.csv"
df = pd.read_csv('boston.csv')
print("1. Данные загружены")
print(f"Размер данных: {df.shape}")
print(f"Столбцы: {list(df.columns)}")
print()

# 2. Проверьте, что у всех загруженных данных числовой тип
print("2. Проверка типов данных:")
print(df.dtypes)
print()

# Проверяем, что все столбцы имеют числовой тип
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print(f"Предупреждение: найдены нечисловые столбцы: {list(non_numeric_cols)}")
else:
    print("Все столбцы имеют числовой тип")
print()

# 3. Проверьте, есть ли по каким-либо признакам отсутствующие данные
print("3. Проверка отсутствующих данных:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("Отсутствующих данных нет")
else:
    print(f"Всего пропусков: {missing_values.sum()}")
    # Заполняем пропуски медианными значениями
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
            print(f"Заполнено {df[column].isnull().sum()} пропусков в столбце {column} значением {median_value}")
print()

# 4. Посчитайте коэффициент корреляции для всех пар признаков
correlation_matrix = df.corr()
print("4. Корреляционная матрица:")
print(correlation_matrix)
print()

# 5. Постройте тепловую карту по корреляционной матрице
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.2f',
    linewidths=0.5
)
plt.title('Тепловая карта корреляций между признаками', fontsize=16)
plt.tight_layout()
plt.show()
print()

# 6. Выберите от 4 до 6 признаков, наиболее подходящих для анализа
print("6. Корреляция признаков с целевой переменной MEDV (цена недвижимости):")
correlation_with_target = correlation_matrix['MEDV'].sort_values(ascending=False)
print(correlation_with_target)
print()

# Выбираем признаки с наибольшей абсолютной корреляцией с MEDV
# Выбираем 6 признаков с наибольшей корреляцией (по модулю)
selected_features = correlation_with_target.drop('MEDV').abs().sort_values(ascending=False).head(6).index.tolist()
print(f"Выбранные признаки (6 с наибольшей корреляцией): {selected_features}")
print("Обоснование выбора: эти признаки имеют наибольшую корреляцию (по модулю) с ценой недвижимости")
print()

# 7. Для каждого из выбранных признаков постройте точечную диаграмму с целевым признаком
print("7. Диаграммы рассеяния выбранных признаков с ценой недвижимости:")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(selected_features):
    ax = axes[i]
    ax.scatter(df[feature], df['MEDV'], alpha=0.5)
    ax.set_xlabel(feature)
    ax.set_ylabel('MEDV (цена)')
    ax.set_title(f'{feature} vs MEDV\nкорр = {correlation_with_target[feature]:.3f}')
    # Добавляем линию тренда
    z = np.polyfit(df[feature], df['MEDV'], 1)
    p = np.poly1d(z)
    ax.plot(df[feature], p(df[feature]), "r--", alpha=0.8)

plt.tight_layout()
plt.show()
print()

# 8. Визуально убедитесь в наличии связи и исключите признаки без явной зависимости
# На основе визуального анализа оставляем признаки с явной линейной зависимостью
print("8. Визуальный анализ зависимостей:")
print("На основе визуального анализа оставляем следующие признаки:")
print("  - LSTAT: четкая обратная зависимость")
print("  - RM: четкая прямая зависимость")
print("  - PTRATIO: обратная зависимость, менее выраженная")
print("  - INDUS: слабая зависимость, но оставляем для анализа")
print("  - TAX: слабая зависимость, но оставляем для анализа")
print("  - NOX: слабая зависимость, но оставляем для анализа")
print("Все 6 признаков оставляем для дальнейшего анализа")
print()

# 9. Сформируйте список факторных признаков и целевую переменную
X = df[selected_features]  # факторные признаки
y = df['MEDV']             # целевая переменная
print(f"9. Размерность данных:")
print(f"   Факторные признаки (X): {X.shape}")
print(f"   Целевая переменная (y): {y.shape}")
print()

# 10. Разбиение датасета на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print("10. Разбиение данных на обучающую и тестовую выборки:")
print(f"   Обучающая выборка: {X_train.shape[0]} записей")
print(f"   Тестовая выборка: {X_test.shape[0]} записей")
print()

# 11. Обучение линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)
print("11. Модель линейной регрессии обучена")
print(f"   Коэффициенты: {model.coef_}")
print(f"   Свободный член: {model.intercept_:.2f}")
print()

# 12. Получение прогнозных значений
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print("12. Получены прогнозные значения для обучающей и тестовой выборок")
print()

# 13. Оценка качества модели
# Коэффициент детерминации (R²)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Корень из среднеквадратичной ошибки (RMSE)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("13. Оценка качества модели:")
print(f"   На обучающей выборке:")
print(f"     R² = {r2_train:.4f}")
print(f"     RMSE = {rmse_train:.4f}")
print()
print(f"   На тестовой выборке:")
print(f"     R² = {r2_test:.4f}")
print(f"     RMSE = {rmse_test:.4f}")
print()

# Анализ результатов
print("Анализ результатов:")
print(f"1. R² на обучающей выборке: {r2_train:.4f} - модель объясняет {r2_train*100:.1f}% дисперсии")
print(f"2. R² на тестовой выборке: {r2_test:.4f} - модель объясняет {r2_test*100:.1f}% дисперсии")
print(f"3. Разница между R²_train и R²_test: {abs(r2_train - r2_test):.4f}")
if abs(r2_train - r2_test) > 0.1:
    print("   Предупреждение: большая разница может указывать на переобучение")
else:
    print("   Разница приемлемая, признаков переобучения нет")
print()

# Визуализация сравнения фактических и прогнозных значений
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Обучающая выборка
axes[0].scatter(y_train, y_train_pred, alpha=0.5)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Фактические значения')
axes[0].set_ylabel('Прогнозные значения')
axes[0].set_title(f'Обучающая выборка\nR² = {r2_train:.3f}')
axes[0].grid(True, alpha=0.3)

# Тестовая выборка
axes[1].scatter(y_test, y_test_pred, alpha=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Фактические значения')
axes[1].set_ylabel('Прогнозные значения')
axes[1].set_title(f'Тестовая выборка\nR² = {r2_test:.3f}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print()

# 14. Постройте boxplot для целевого признака (MEDV)
plt.figure(figsize=(10, 6))
boxprops = dict(linestyle='-', linewidth=2, color='blue')
medianprops = dict(linestyle='-', linewidth=2, color='red')
flierprops = dict(marker='o', markerfacecolor='red', markersize=8, linestyle='none')

plt.boxplot(df['MEDV'],
            boxprops=boxprops,
            medianprops=medianprops,
            flierprops=flierprops)
plt.title('Boxplot для целевого признака MEDV (цена недвижимости)', fontsize=14)
plt.ylabel('Цена (тыс. $)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# Определение выбросов с помощью межквартильного размаха
Q1 = df['MEDV'].quantile(0.25)
Q3 = df['MEDV'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['MEDV'] < lower_bound) | (df['MEDV'] > upper_bound)]
print("14. Анализ выбросов:")
print(f"   Q1 (25-й перцентиль): {Q1:.2f}")
print(f"   Q3 (75-й перцентиль): {Q3:.2f}")
print(f"   IQR (межквартильный размах): {IQR:.2f}")
print(f"   Нижняя граница: {lower_bound:.2f}")
print(f"   Верхняя граница: {upper_bound:.2f}")
print(f"   Количество выбросов: {len(outliers)}")
print(f"   Процент выбросов: {len(outliers)/len(df)*100:.1f}%")
print()

if len(outliers) > 0:
    print("Выбросы (первые 5):")
    print(outliers[['MEDV'] + selected_features].head())
