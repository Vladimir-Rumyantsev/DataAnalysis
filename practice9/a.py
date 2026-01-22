import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. Загрузить данные в датафрейм и вывести несколько строк
print("\n" + "="*80 + f"\nЧасть 1. Загрузка данных...\n" + "="*80)
df = pd.read_csv('Mental_health_diagnosis_treatment.csv')
print(
    f"Данные загружены!"
    f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов"
    f"\nСодержание:"
    f"\n{df.head}"
)


# 2. Определите количество значений каждого из признаков в загруженных данных

# Вывод общей информации о данных
print("\n" + "="*80 + "\nЧасть 2. Общая информация о датафрейме\n" + "="*80)
df.info()

# Отсутствующих данных нет


# 3. Определите тип признаков
print(
    f"\n" + "="*80 + f"\nЧасть 3. Анализ и преобразование типов данных\n" + "="*80 +
    f"\nТипы данных каждого признака:"
    f"\n{df.dtypes}"
)

# Определяем числовые и категориальные признаки
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print(
    f"\nЧисловые признаки ({len(numeric_features)}): {numeric_features}"
    f"\nКатегориальные признаки ({len(categorical_features)}): {categorical_features}"
)

# Проверяем, нужны ли преобразования типов
# 'Treatment Start Date' хранится как object, нужно преобразовать в datetime
print(
    f"\nПроверка необходимости преобразования типов:"
    f"\nПризнак 'Treatment Start Date' текущий тип: {df['Treatment Start Date'].dtype}"
)
df['Treatment Start Date'] = pd.to_datetime(df['Treatment Start Date'])
print(f"Признак 'Treatment Start Date' новый тип: {df['Treatment Start Date'].dtype}")

# Обновляем списки признаков после преобразования
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print(
    f"\nОбновленные списки после преобразования:"
    f"\nЧисловые признаки ({len(numeric_features)}): {numeric_features}"
    f"\nКатегориальные признаки ({len(categorical_features)}): {categorical_features}"
)


# 4. Сформулируйте 2 вопроса с простой фильтрацией данных
print(f"\n" + "="*80 + f"\nЧасть 4. Фильтрация данных - 2 вопроса\n" + "="*80)

# Вопрос 1: Вывести всех пациентов с диагнозом 'Major Depressive Disorder'
print('\nВопрос 1: Все пациенты с диагнозом "Major Depressive Disorder"')
major_depression_patients = df[df['Diagnosis'] == 'Major Depressive Disorder']
print(
    f"Количество пациентов: {len(major_depression_patients)}"
    f"\nПациенты:"
    f"\n{major_depression_patients[['Patient ID', 'Age', 'Gender', 'Diagnosis']]}"
)

# Вопрос 2: Вывести всех пациентов, у которых оценка прогресса лечения выше 8
print("\nВопрос 2: Все пациенты с оценкой прогресса лечения > 8")
high_progress_patients = df[df['Treatment Progress (1-10)'] > 8]
print(
    f"Количество пациентов: {len(high_progress_patients)}"
    f"\nПациенты:"
    f"\n{high_progress_patients[['Patient ID', 'Treatment Progress (1-10)', 'Outcome']]}"
)


# 5. Сформулируйте 2 вопроса с агрегирующими функциями
print("\n" + "="*80 + "\nЧасть 5. Агрегирующие функции - 2 вопроса\n" + "="*80)

# Вопрос 1: Средний возраст пациентов с диагнозом 'Panic Disorder'
print("\nВопрос 1: Средний возраст пациентов с диагнозом 'Panic Disorder'")
avg_age_panic = df[df['Diagnosis'] == 'Panic Disorder']['Age'].mean()
print(f"Средний возраст: {round(avg_age_panic, 2)} лет")

# Вопрос 2: Средний уровень стресса у мужчин и женщин
print("\nВопрос 2: Средний уровень стресса по полу")
avg_stress_by_gender = df.groupby('Gender')['Stress Level (1-10)'].mean()
print(
    f"Женщины — {round(avg_stress_by_gender['Female'], 3)}"
    f"\nМужчины — {round(avg_stress_by_gender['Male'], 3)}"
)


# 6. Сформулируйте 2 вопроса на поиск максимальных/минимальных значений
print("\n" + "="*80 + "\nЧасть 6. Поиск максимальных/минимальных значений - 2 вопроса\n" + "="*80)

# Вопрос 1: 20 пациентов с максимальной комплаентностью
print("Вопрос 1: 20 пациентов с максимальной комплаентностью")
top_adherence = df.nlargest(20, 'Adherence to Treatment (%)')
print(top_adherence[['Patient ID', 'Adherence to Treatment (%)', 'Diagnosis']])

# Вопрос 2: 10% пациентов с минимальным качеством сна
print("\nВопрос 2: 10% пациентов с минимальным качеством сна")
num_patients = int(len(df) * 0.1)  # 10% от общего количества
worst_sleep = df.nsmallest(num_patients, 'Sleep Quality (1-10)')
print(
    f"Количество пациентов (10%): {num_patients}"
    f"\nПациенты с минимальным качеством сна:"
    f"\n{worst_sleep[['Patient ID', 'Sleep Quality (1-10)', 'Stress Level (1-10)']]}"
)


# 7. Категориальный признак для ленточной или столбчатой диаграммы
print("\n" + "="*80 + "\nЧасть 7. Столбчатая диаграмма для категориального признака\n" + "="*80)

# Выбираем признак 'Outcome' (3 значения) для столбчатой диаграммы
outcome_counts = df['Outcome'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(outcome_counts.index, outcome_counts.values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
plt.title('Распределение пациентов по результатам лечения', fontsize=16)
plt.xlabel('Результат лечения', fontsize=12)
plt.ylabel('Количество пациентов', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Добавляем подписи значений
for i, (index, value) in enumerate(outcome_counts.items()):
    plt.text(i, value + 2, str(value), ha='center', fontsize=12)

plt.tight_layout()
plt.show()

print(
    f"Распределение по результатам лечения:"
    f"\n{outcome_counts}"
)


# 8. Категориальный признак для круговой диаграммы
print("\n" + "="*80 + "\nЧасть 8. Круговая диаграмма для категориального признака\n" + "="*80)

# Выбираем признак 'Diagnosis' (4 значения) для круговой диаграммы
diagnosis_counts = df['Diagnosis'].value_counts()

plt.figure(figsize=(10, 8))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
plt.pie(
    diagnosis_counts.values,
    labels=diagnosis_counts.index,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 12}
)
plt.title('Распределение пациентов по диагнозам', fontsize=16)
plt.axis('equal')  # Чтобы круг был кругом
plt.tight_layout()
plt.show()

print(
    f"Распределение по диагнозам:"
    f"\n{diagnosis_counts}"
)


# 9. Группировка данных и агрегация
print("\n" + "="*80 + "\nЧасть 9. Группировка данных и агрегация\n" + "="*80)

# Группируем по диагнозу и вычисляем средний возраст и средний уровень стресса
grouped_stats = df.groupby('Diagnosis').agg({
    'Age': 'mean',
    'Stress Level (1-10)': 'mean'
}).round(2)

print(
    f"Средний возраст и средний уровень стресса по диагнозам:"
    f"\n{grouped_stats}"
)


# 10. Создание нового признака
print("\n" + "="*80 + "\nЧасть 10. Создание нового признака\n" + "="*80)

# Создаем бинарный признак: 1 - если лечение успешно (Outcome = 'Improved' и Treatment Progress >= 7), иначе 0
df['Successful Treatment'] = ((df['Outcome'] == 'Improved') &
                              (df['Treatment Progress (1-10)'] >= 7)).astype(int)

# Проверяем распределение нового признака
success_counts = df['Successful Treatment'].value_counts()
print(
    f"Распределение нового признака \"Successful Treatment\":"
    f"\n{success_counts}"
    f"\nПроцент успешных лечений: {round((success_counts[1] * 100) / len(df), 2)}%"
)

# Выводим несколько строк для проверки
print(
    f"\nНовый признак:"
    f"\n{df[['Patient ID', 'Outcome', 'Treatment Progress (1-10)', 'Successful Treatment']]}"
)


# 11. Исследование связи между числовыми признаками
print("\n" + "="*80 + "\n11. Исследование связи между числовыми признаками\n" + "="*80)

# Выбираем 3 числовых признака с большим диапазоном значений
selected_features = ['Age', 'Physical Activity (hrs/week)', 'Adherence to Treatment (%)']

print(f"Выбранные признаки: {selected_features}")

# Создаем фигуру с 3 подграфиками для попарного анализа
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# График 1: Возраст vs Физическая активность
axes[0].scatter(df['Age'], df['Physical Activity (hrs/week)'], alpha=0.6, color='blue')
axes[0].set_xlabel('Возраст (лет)')
axes[0].set_ylabel('Физическая активность (часы/неделю)')
axes[0].set_title('Возраст vs Физическая активность')
axes[0].grid(True, alpha=0.3)

# График 2: Возраст vs Приверженность лечению
axes[1].scatter(df['Age'], df['Adherence to Treatment (%)'], alpha=0.6, color='green')
axes[1].set_xlabel('Возраст (лет)')
axes[1].set_ylabel('Приверженность лечению (%)')
axes[1].set_title('Возраст vs Приверженность лечению')
axes[1].grid(True, alpha=0.3)

# График 3: Физическая активность vs Приверженность лечению
axes[2].scatter(df['Physical Activity (hrs/week)'], df['Adherence to Treatment (%)'], alpha=0.6, color='red')
axes[2].set_xlabel('Физическая активность (часы/неделю)')
axes[2].set_ylabel('Приверженность лечению (%)')
axes[2].set_title('Физическая активность vs Приверженность лечению')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Вычисляем корреляции для количественной оценки
correlation_matrix = df[selected_features].corr()

# Выводы
print(
    f"\nМатрица корреляций:"
    f"\n{correlation_matrix}"
    f"\n\nВыводы о зависимости между признаками на основе анализа диаграмм рассеяния:"
    f"\n1. Возраст и физическая активность: НЕТ значимой корреляции (r = 0.001)"
    f"\n2. Возраст и приверженность лечению: НЕТ значимой корреляции (r = -0.022)"
    f"\n3. Физическая активность и приверженность лечению: НЕТ значимой корреляции (r = -0.054)"
    f"\n\nЭто может означать, что:"
    f"\n- Данные факторы независимы друг от друга"
    f"\n- Влияние опосредовано другими переменными"
    f"\n- Связь имеет нелинейный характер"
)


# 12. Анализ выбросов с помощью boxplot
print("\n" + "="*80 + "\n12. Анализ выбросов с помощью boxplot\n" + "="*80)

# Выбираем 3 числовых признака для анализа
boxplot_features = ['Age', 'Physical Activity (hrs/week)', 'Adherence to Treatment (%)']

# Создаем boxplot для каждого признака
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, feature in enumerate(boxplot_features):
    axes[i].boxplot(df[feature], vert=True, patch_artist=True)
    axes[i].set_title(f'Boxplot для {feature}', fontsize=14)
    axes[i].set_ylabel('Значение', fontsize=12)
    axes[i].set_xlabel(feature, fontsize=12)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Анализ выбросов для каждого признака
print("\nАнализ выбросов по каждому признаку:")
for feature in boxplot_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    num_outliers = len(outliers)

    print(f"\n{feature}:")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Границы выбросов: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Количество выбросов: {num_outliers}")
    print(f"  Процент выбросов: {num_outliers / len(df) * 100:.2f}%")


# 13. Выбор типа задачи машинного обучения
print(
    "\n" + "="*80 + "\n13. Выбор типа задачи машинного обучения\n" + "="*80 +
    "\nТип задачи: \"КЛАССИФИКАЦИЯ\", поскольку рекомендованный тип задачи для данного датасета - классификация"
)

# 14. Факторные и целевые признаки
print("\n" + "="*80 + "\n14. Факторные и целевые признаки\n" + "="*80)

# Определяем целевой признак - выберем 'Outcome' для многоклассовой классификации
target_feature = 'Outcome'
print(f"Целевой признак (target): {target_feature}")

# Определяем факторные признаки (исключая идентификаторы и целевой признак)
excluded_features = ['Patient ID', 'Treatment Start Date', target_feature, 'Successful Treatment']
factor_features = [col for col in df.columns if col not in excluded_features]

print(f"Количество факторных признаков: {len(factor_features)}")
print("\nФакторные признаки:")
for i, feature in enumerate(factor_features, 1):
    print(f"{i:2}. {feature}")

# Разделяем признаки на категориальные и числовые
numeric_features = df[factor_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df[factor_features].select_dtypes(include=['object']).columns.tolist()

print(f"\nЧисловые факторные признаки ({len(numeric_features)}):")
print(numeric_features)
print(f"\nКатегориальные факторные признаки ({len(categorical_features)}):")
print(categorical_features)
