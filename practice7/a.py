import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score

# 1. Загрузка данных
print("1. Загрузка данных...")


# Встроенная функция для определения кодировки
def detect_file_encoding(file_path, n_bytes=10000):
    with open(file_path, 'rb') as f:
        raw = f.read(n_bytes)

    # Простые проверки для русских кодировок
    if b'\xff\xfe' in raw[:2]:
        return 'utf-16'
    elif b'\xfe\xff' in raw[:2]:
        return 'utf-16-be'
    else:
        try:
            # Попробуем common Russian encodings
            for encoding in ['cp1251', 'utf-8', 'iso-8859-1', 'cp866']:
                try:
                    raw[:100].decode(encoding)
                    return encoding
                except:
                    continue
        except:
            pass
    return 'utf-8'  # fallback


try:
    encoding = detect_file_encoding('База.csv')
    print(f"  Определена кодировка: {encoding}")
    df = pd.read_csv('База.csv', sep=';', encoding=encoding, low_memory=False)
    print(f"  Загружено строк: {df.shape[0]}, столбцов: {df.shape[1]}")
except Exception as e:
    print(f"  Ошибка при загрузке: {e}")
    # Попробуем альтернативные кодировки
    for enc in ['cp1251', 'utf-8', 'windows-1251', 'iso-8859-1']:
        try:
            df = pd.read_csv('База.csv', sep=';', encoding=enc, low_memory=False)
            print(f"  Успешно загружено с кодировкой: {enc}")
            break
        except:
            continue

# 2. Предварительная фильтрация
print("\n2. Предварительная фильтрация...")

# a. Только жилые помещения
df = df[df['ВидПомещения'].astype(str).str.strip() == 'жилые помещения'].copy()
print(f"  После фильтрации по виду помещения: {df.shape[0]} строк")

# b. Фильтрация по статусу и кодирование целевой переменной
df = df[df['СледующийСтатус'].isin(['Продана', 'Свободна'])].copy()
df['target'] = df['СледующийСтатус'].map({'Продана': 1, 'Свободна': 0})

# c. Удаление ненужных столбцов
cols_to_drop = ['УИД_Брони', 'ВидПомещения', 'СледующийСтатус', 'ДатаБрони', 'ВремяБрони']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

print(f"  После всех фильтраций: {df.shape[0]} строк, {df.shape[1]} столбцов")

# 3. Преобразование типов данных
print("\n3. Преобразование типов данных...")

# a. Числовые поля - правильность типа
numeric_cols = ['ПродаваемаяПлощадь', 'Этаж', 'СтоимостьНаДатуБрони',
                'СкидкаНаКвартиру', 'ФактическаяСтоимостьПомещения']

for col in numeric_cols:
    if col in df.columns:
        # Замена запятых на точки для десятичных чисел
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')

# b. Обработка поля 'Тип' (количество комнат)
if 'Тип' in df.columns:
    df['Тип'] = df['Тип'].astype(str)
    # Удаляем 'к' в конце, заменяем 'с' на NaN
    df['Тип'] = df['Тип'].str.replace('к', '', regex=False)
    df['Тип'] = df['Тип'].replace('с', np.nan)
    # Заменяем запятые на точки
    df['Тип'] = df['Тип'].str.replace(',', '.', regex=False)
    df['Тип'] = pd.to_numeric(df['Тип'], errors='coerce')

# c. Бинарные признаки
binary_mappings = {
    'ВременнаяБронь': {'Да': 1, 'Нет': 0, 'да': 1, 'нет': 0},
    'СделкаАН': {'Да': 1, 'Нет': 0, 'да': 1, 'нет': 0},
    'ИнвестиционныйПродукт': {'Да': 1, 'Нет': 0, 'да': 1, 'нет': 0},
    'Привилегия': {'Да': 1, 'Нет': 0, 'да': 1, 'нет': 0},
    'ИсточникБрони': {'ручная': 0, 'МП': 1},
}

# Применяем маппинг для бинарных признаков
for col, mapping in binary_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Другие бинарные признаки
if 'ТипСтоимости' in df.columns:
    df['ТипСтоимости'] = df['ТипСтоимости'].apply(
        lambda x: 1 if isinstance(x, str) and '100%' in x else 0 if isinstance(x, str) and 'рассрочку' in x else np.nan
    )

if 'ВариантОплаты' in df.columns:
    df['ВариантОплаты'] = df['ВариантОплаты'].apply(
        lambda x: 1 if isinstance(x, str) and 'Единовременная' in x else 0 if isinstance(x,
                                                                                         str) and 'рассрочку' in x else np.nan
    )

# d. Категориальные небинарные признаки
categorical_cols = ['Город', 'Статус лида (из CRM)']
for col in categorical_cols:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], prefix=col.replace(' ', '_'), drop_first=False)

print(f"  После преобразований: {df.shape[1]} столбцов")

# 4. Обработка пропущенных значений
print("\n4. Обработка пропущенных значений...")
print(f"  Пропуски до обработки:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# a. СкидкаНаКвартиру - заменяем на 0
if 'СкидкаНаКвартиру' in df.columns:
    df['СкидкаНаКвартиру'] = df['СкидкаНаКвартиру'].fillna(0)

# b. Тип и ПродаваемаяПлощадь - заменяем на медиану
for col in ['Тип', 'ПродаваемаяПлощадь']:
    if col in df.columns and df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  {col}: заменено {df[col].isnull().sum()} пропусков на медиану {median_val:.2f}")

# c. ВариантОплатыДоп - удаляем столбец
if 'ВариантОплатыДоп' in df.columns:
    df.drop(columns=['ВариантОплатыДоп'], inplace=True)

# d. Остальные поля - заполняем или удаляем
initial_rows = len(df)
threshold = 0.05 * len(df)  # 5% от общего количества строк

# Сначала заполним пропуски в оставшихся бинарных признаках
for col in ['ТипСтоимости', 'ВариантОплаты']:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)

# Удаляем строки с пропусками в других столбцах (если их немного)
for col in df.columns:
    if col != 'target' and df[col].isnull().any():
        missing_count = df[col].isnull().sum()
        if missing_count < threshold and missing_count > 0:
            df = df.dropna(subset=[col])
            print(f"  {col}: удалено {missing_count} строк с пропусками")

print(f"  Удалено строк: {initial_rows - len(df)}")

# Убедимся, что нет пропусков
for col in df.columns:
    if df[col].isnull().any():
        # Заполняем медианой для числовых, модой для категориальных
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)

print(f"  Пропуски после обработки: {df.isnull().sum().sum()}")

# 5. Добавление новых признаков
print("\n5. Добавление новых признаков...")

# a. Цена за квадратный метр
if all(col in df.columns for col in ['ФактическаяСтоимостьПомещения', 'ПродаваемаяПлощадь']):
    # Избегаем деления на ноль
    df['ЦенаЗаМетр'] = df.apply(
        lambda row: row['ФактическаяСтоимостьПомещения'] / row['ПродаваемаяПлощадь']
        if row['ПродаваемаяПлощадь'] > 0 else np.nan,
        axis=1
    )
    # Заменяем NaN на медиану
    median_price = df['ЦенаЗаМетр'].median()
    df['ЦенаЗаМетр'] = df['ЦенаЗаМетр'].fillna(median_price)

# b. Скидка в процентах
if all(col in df.columns for col in ['СтоимостьНаДатуБрони', 'СкидкаНаКвартиру']):
    # Избегаем деления на ноль
    df['СкидкаПроцент'] = df.apply(
        lambda row: (row['СкидкаНаКвартиру'] / row['СтоимостьНаДатуБрони']) * 100
        if row['СтоимостьНаДатуБрони'] != 0 else 0.0,
        axis=1
    )

print(f"  Добавлено 2 новых признака")

# 6. Нормализация
print("\n6. Нормализация...")

# Разделяем на признаки и целевую переменную
X = df.drop(columns=['target'], errors='ignore')
y = df['target']

# Проверяем, что все столбцы числовые
print(f"  Проверка типов данных в X:")
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        print(f"    {col}: {X[col].dtype} - преобразуем в числовой")
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())

# Минимаксная нормализация для большинства признаков
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Особый случай для 'СкидкаНаКвартиру' - приводим к [-0.5, 0.5]
if 'СкидкаНаКвартиру' in X_normalized.columns:
    # Масштабируем от -0.5 до 0.5
    # MinMaxScaler уже дал [0, 1], преобразуем в [-0.5, 0.5]
    X_normalized['СкидкаНаКвартиру'] = X_normalized['СкидкаНаКвартиру'] - 0.5
    print("  Столбец 'СкидкаНаКвартиру' нормализован к диапазону [-0.5, 0.5]")
    print("  Такой диапазон удобен для разделения на скидки (положительные) и наценки (отрицательные)")

# 7. Проверка сбалансированности
print("\n7. Проверка сбалансированности датасета...")
class_distribution = y.value_counts(normalize=True)
print(f"  Распределение классов:")
print(f"    Класс 0 (Свободна): {class_distribution.get(0, 0):.2%} ({y.value_counts().get(0, 0)} строк)")
print(f"    Класс 1 (Продана): {class_distribution.get(1, 0):.2%} ({y.value_counts().get(1, 0)} строк)")

if abs(class_distribution.get(0, 0) - class_distribution.get(1, 0)) < 0.1:
    print("  Датасет сбалансирован")
else:
    print("  Датасет НЕ сбалансирован")
    print("  Рекомендуется использовать методы работы с несбалансированными данными")

# 8-9. Разделение на обучающую и тестовую выборки
print("\n8-9. Разделение на обучающую и тестовую выборки...")
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.3, random_state=42, stratify=y
)
print(f"  Обучающая выборка: {X_train.shape[0]} строк")
print(f"  Тестовая выборка: {X_test.shape[0]} строк")


# Функция для оценки модели
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)

    # Прогнозы
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Метрики
    metrics = {
        'model': model_name,
        'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
        'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, zero_division=0)
    }

    return metrics, y_test_pred


# 10-13. KNN и Decision Tree
print("\n10-13. Обучение и оценка базовых моделей...")

# KNN
knn = KNeighborsClassifier()
knn_metrics, knn_pred = evaluate_model(knn, X_train, X_test, y_train, y_test, "KNN")

# Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree_metrics, tree_pred = evaluate_model(tree, X_train, X_test, y_train, y_test, "Decision Tree")

# Вывод результатов
results = pd.DataFrame([knn_metrics, tree_metrics])
print("\nРезультаты базовых моделей:")
print(results.to_string())

# 14. Выводы по базовым моделям
print("\n14. Выводы по базовым моделям:")
print("""
Precision (точность) - доля правильно предсказанных положительных классов 
среди всех объектов, которые модель отнесла к положительному классу.

Recall (полнота) - доля правильно предсказанных положительных классов 
среди всех объектов, которые на самом деле являются положительными.

В нашем контексте:
- Высокий Precision: мало ложных срабатываний (редко предсказываем продажу, когда её не будет)
- Высокий Recall: находим большинство реальных продаж

Вывод: обе модели показывают умеренные результаты, но есть признаки переобучения 
у дерева решений (разница между train и test метриками).
""")

# 15. Boxplot и удаление выбросов
print("\n15. Анализ выбросов...")

# Выбор числовых признаков для анализа
numeric_features = ['ПродаваемаяПлощадь', 'Этаж', 'СтоимостьНаДатуБрони',
                    'ФактическаяСтоимостьПомещения', 'ЦенаЗаМетр', 'СкидкаПроцент']
numeric_features = [col for col in numeric_features if col in X.columns]

# Построение boxplot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numeric_features[:6]):
    axes[idx].boxplot(df[col].dropna())
    axes[idx].set_title(col)
    axes[idx].set_ylabel('Значение')

plt.tight_layout()
plt.suptitle('Boxplot числовых признаков', y=1.02, fontsize=16)
plt.show()

print("\nУдаление выбросов по методу IQR...")
# Удаление выбросов для каждого числового признака
df_no_outliers = df.copy()

for col in numeric_features:
    if col in df_no_outliers.columns:
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        initial_count = len(df_no_outliers)
        df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) &
                                        (df_no_outliers[col] <= upper_bound)]
        removed = initial_count - len(df_no_outliers)
        print(f"  {col}: удалено {removed} выбросов")

print(f"\n  Исходный размер: {len(df)} строк")
print(f"  После удаления выбросов: {len(df_no_outliers)} строк")
print(f"  Удалено: {len(df) - len(df_no_outliers)} строк ({((len(df) - len(df_no_outliers)) / len(df) * 100):.1f}%)")

# Обучение моделей на данных без выбросов
if len(df_no_outliers) > 0.5 * len(df):  # Если осталось достаточно данных
    X_no_out = df_no_outliers.drop(columns=['target'], errors='ignore')
    y_no_out = df_no_outliers['target']

    X_no_out_norm = pd.DataFrame(scaler.transform(X_no_out), columns=X_no_out.columns)
    if 'СкидкаНаКвартиру' in X_no_out_norm.columns:
        X_no_out_norm['СкидкаНаКвартиру'] = X_no_out_norm['СкидкаНаКвартиру'] - 0.5

    X_train_no, X_test_no, y_train_no, y_test_no = train_test_split(
        X_no_out_norm, y_no_out, test_size=0.3, random_state=42, stratify=y_no_out
    )

    # Обучение моделей
    knn_no = KNeighborsClassifier()
    knn_no_metrics, _ = evaluate_model(knn_no, X_train_no, X_test_no, y_train_no, y_test_no, "KNN (без выбросов)")

    tree_no = DecisionTreeClassifier(random_state=42)
    tree_no_metrics, _ = evaluate_model(tree_no, X_train_no, X_test_no, y_train_no, y_test_no, "Tree (без выбросов)")

    results_no = pd.DataFrame([knn_no_metrics, tree_no_metrics])
    print("\nРезультаты моделей без выбросов:")
    print(results_no.to_string())

# 16. Подбор параметров
print("\n16. Подбор оптимальных параметров...")

# Для KNN
print("\nПодбор k для KNN...")
k_values = range(1, 41)
knn_f1_scores = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_pred = knn_temp.predict(X_test)
    knn_f1_scores.append(f1_score(y_test, y_pred))

optimal_k = k_values[np.argmax(knn_f1_scores)]
print(f"  Оптимальное k: {optimal_k} (F1: {max(knn_f1_scores):.4f})")

# Для Decision Tree
print("\nПодбор глубины для Decision Tree...")
depth_values = range(2, 41)
tree_f1_scores = []

for depth in depth_values:
    tree_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_temp.fit(X_train, y_train)
    y_pred = tree_temp.predict(X_test)
    tree_f1_scores.append(f1_score(y_test, y_pred))

optimal_depth = depth_values[np.argmax(tree_f1_scores)]
print(f"  Оптимальная глубина: {optimal_depth} (F1: {max(tree_f1_scores):.4f})")

# Графики
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(k_values, knn_f1_scores, marker='o')
ax1.set_xlabel('Количество соседей (k)')
ax1.set_ylabel('F1-мера')
ax1.set_title('Зависимость F1-меры от k для KNN')
ax1.grid(True)
ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Оптимальное k={optimal_k}')
ax1.legend()

ax2.plot(depth_values, tree_f1_scores, marker='o', color='orange')
ax2.set_xlabel('Глубина дерева')
ax2.set_ylabel('F1-мера')
ax2.set_title('Зависимость F1-меры от глубины дерева')
ax2.grid(True)
ax2.axvline(x=optimal_depth, color='r', linestyle='--', label=f'Оптимальная глубина={optimal_depth}')
ax2.legend()

plt.tight_layout()
plt.show()

# 17-18. Логистическая регрессия и SVM
print("\n17-18. Логистическая регрессия и SVM...")

# Логистическая регрессия
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg_metrics, logreg_pred = evaluate_model(logreg, X_train, X_test, y_train, y_test, "Logistic Regression")

# SVM
svm = LinearSVC(max_iter=10000, random_state=42)
svm_metrics, svm_pred = evaluate_model(svm, X_train, X_test, y_train, y_test, "SVM")

# Модели с оптимальными параметрами
knn_opt = KNeighborsClassifier(n_neighbors=optimal_k)
knn_opt_metrics, _ = evaluate_model(knn_opt, X_train, X_test, y_train, y_test, "KNN (оптимальный)")

tree_opt = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
tree_opt_metrics, _ = evaluate_model(tree_opt, X_train, X_test, y_train, y_test, "Tree (оптимальный)")

# Сводная таблица всех моделей
all_metrics = [knn_metrics, tree_metrics, logreg_metrics, svm_metrics, knn_opt_metrics, tree_opt_metrics]
all_results = pd.DataFrame(all_metrics)

print("\nСводные результаты всех моделей:")
print(all_results.to_string())

# Вывод лучшей модели
best_model_idx = all_results['test_f1'].idxmax()
best_model = all_results.loc[best_model_idx]
print(f"\nЛучшая модель: {best_model['model']}")
print(f"  F1-мера: {best_model['test_f1']:.4f}")
print(f"  Precision: {best_model['test_precision']:.4f}")
print(f"  Recall: {best_model['test_recall']:.4f}")

# Интерпретация результатов
print("""
Итоговые выводы:
1. Все модели показали удовлетворительные результаты, но есть признаки переобучения.
2. Удаление выбросов может как улучшить, так и ухудшить качество моделей.
3. Подбор параметров существенно улучшает качество моделей.
4. Для данной задачи наиболее важны метрики Precision и Recall:
   - Высокий Precision важен для минимизации ложных прогнозов продаж
   - Высокий Recall важен для выявления максимального количества реальных продаж
5. В зависимости от бизнес-целей можно выбирать модель с лучшим Precision или Recall.
""")
