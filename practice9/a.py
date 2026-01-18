import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print(f"\nЧасть 1.\nЗагрузка данных...")

# 1. Загрузите данные в датафрейм и вывести несколько строк
df = pd.read_csv('Mental_health_diagnosis_treatment.csv')
print(
    f"Данные загружены!"
    f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов"
    f"\nСодержание:"
    f"\n{df.head}"
)


# 2. Определите количество значений каждого из признаков в загруженных данных

# Вывод общей информации о данных
print("\nЧасть 2.\nОбщая информация о датафрейме:")
df.info()

# Отсутствующих данных нет


# 3. Определите тип признаков
print(
    f"\nЧасть 3."
    f"\nАнализ и преобразование типов данных"
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
print(
    f"\nЧасть 4."
    f"\nФильтрация данных - 2 вопроса"
)

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
print("\nЧасть 5.\nАгрегирующие функции - 2 вопроса")

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
print("\nЧасть 6.\nПоиск максимальных/минимальных значений - 2 вопроса")

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
print("\nЧасть 7.\nСтолбчатая диаграмма для категориального признака")

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
print("\nЧасть 8.\nКруговая диаграмма для категориального признака")

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

print("\nЧасть 9.\nГруппировка данных и агрегация")

# Группируем по диагнозу и вычисляем средний возраст и средний уровень стресса
grouped_stats = df.groupby('Diagnosis').agg({
    'Age': 'mean',
    'Stress Level (1-10)': 'mean'
}).round(2)

print(
    f"Средний возраст и средний уровень стресса по диагнозам:"
    f"\n{grouped_stats}"
)
