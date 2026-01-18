import pandas as pd


# 1. Загрузите данные из файла
print(
    f"\nЧасть 1."
    f"\nИдёт загрузка данных..."
)
telecom_df = pd.read_csv('telecom_churn.csv')
print(
    f"Данные загружены! Размер данных: {telecom_df.shape}"
)

# Выведите общую информацию о датафрейме
print(f"Общая информация о данных:")
telecom_df.info()

# Проверка отсутствующих данных
print(f"\nПроверка отсутствующих данных:")
missing_values = telecom_df.isnull().sum()
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("Отсутствующих данных нет")

# 2. Определите, сколько клиентов активны, а сколько потеряно
churn_counts = telecom_df['Churn'].value_counts()
print(
    f"\nЧасть 2."
    f"\nРаспределение клиентов:"
    f"\n{churn_counts}"
)

# Вычисление процентов
churn_percent = telecom_df['Churn'].value_counts(normalize=True) * 100
print(
    f"\nПроцентное распределение:"
    f"\nАктивные клиенты: {round(churn_percent[False], 2)}%"
    f"\nПотерянные клиенты: {round(churn_percent[True], 2)}%"
)

# 3. Добавьте столбец со средней продолжительностью одного звонка
# Суммарная продолжительность всех звонков
total_minutes = (
        telecom_df['Total day minutes'] +
        telecom_df['Total eve minutes'] +
        telecom_df['Total night minutes']
)

# Суммарное количество всех звонков
total_calls = (
        telecom_df['Total day calls'] +
        telecom_df['Total eve calls'] +
        telecom_df['Total night calls']
)

telecom_df['Average call duration'] = total_minutes / total_calls

print(
    f"\nЧасть 3."
    f"\n10 клиентов с наибольшей средней продолжительностью звонка:"
    f"\n{telecom_df.sort_values(
        'Average call duration',
        ascending=False
    ).head(10)[['Average call duration', 'Churn']]}"
)

# 4. Средняя продолжительность звонка по категориям оттока
avg_duration_by_churn = telecom_df.groupby('Churn')['Average call duration'].mean()
print(
    f"\nЧасть 4."
    f"\nСредняя продолжительность звонка по категориям оттока: {avg_duration_by_churn}"
)

# Проверка существенной разницы
diff_duration = abs(avg_duration_by_churn[True] - avg_duration_by_churn[False])
print(
    f"Разница в средней продолжительности: {round(diff_duration, 2)} минут"
)
print(
    f"Разница не является существенной" if diff_duration < 0.5 else "Разница является существенной"
)

# 5. Среднее количество звонков в службу поддержки по категориям оттока
avg_service_calls_by_churn = telecom_df.groupby('Churn')['Customer service calls'].mean()
print(
    f"\nЧасть 5."
    f"\nСреднее количество звонков в службу поддержки по категориям оттока: {avg_service_calls_by_churn}"
)

# Проверка существенной разницы
diff_service_calls = abs(avg_service_calls_by_churn[True] - avg_service_calls_by_churn[False])
print(
    f"Разница в среднем количестве звонков: {round(diff_service_calls, 2)}"
)
print(
    f"Разница является существенной" if diff_service_calls > 0.5 else "Разница не является существенной"
)

# 6. Таблица сопряженности Churn и Customer service calls
cross_tab = pd.crosstab(telecom_df['Customer service calls'], telecom_df['Churn'])
cross_tab['Total'] = cross_tab[True] + cross_tab[False]
cross_tab['Churn_rate'] = cross_tab[True] * 100 / cross_tab['Total']
print(
    f"\nЧасть 6."
    f"\nТаблица сопряженности Churn и Customer service calls:"
    f"\n{cross_tab}"
)

# Общий процент оттока
overall_churn_rate = churn_percent[True]
print(f"Общий процент оттока: {round(overall_churn_rate, 2)}%")

# Поиск, при каком количестве звонков процент оттока > 40%
high_churn_calls = cross_tab[cross_tab['Churn_rate'] > 40]
print("\nКоличество звонков, при котором процент оттока > 40%:")
if len(high_churn_calls) > 0:
    print(high_churn_calls.index.tolist())
    print(f"Минимальное количество: {high_churn_calls.index.min()}")
else:
    print("Нет значений с процентом оттока > 40%")

# 7. Таблица сопряженности Churn и International plan
cross_tab2 = pd.crosstab(telecom_df['International plan'], telecom_df['Churn'])
cross_tab2['Total'] = cross_tab2[True] + cross_tab2[False]
cross_tab2['Churn_rate'] = cross_tab2[True] * 100 / cross_tab2['Total']
print(
    f"\nЧасть 7."
    f"\nТаблица сопряженности Churn и International plan:"
    f"\n{cross_tab2}"
)

# Сравнение процентов оттока
churn_with_plan = cross_tab2.loc['Yes', 'Churn_rate']
churn_without_plan = cross_tab2.loc['No', 'Churn_rate']
print(
    f"\nПроцент оттока с международным роумингом: {round(churn_with_plan, 2)}%"
    f"\nПроцент оттока без международного роуминга: {round(churn_without_plan, 2)}%"
    f"\nРазница: {round(abs(churn_with_plan - churn_without_plan), 2)}%"
)
if churn_with_plan > churn_without_plan + 10:
    print("Процент оттока среди клиентов с международным роумингом СУЩЕСТВЕННО ВЫШЕ")
elif churn_with_plan + 10 < churn_without_plan:
    print("Процент оттока среди клиентов с международным роумингом СУЩЕСТВЕННО НИЖЕ")
else:
    print("Проценты оттока среди клиентов с международным роумингом и без него СОПОСТАВИМЫ")

# 8. Прогнозируемый отток и оценка ошибок
# Создаем прогноз: если >=4 звонков в поддержку ИЛИ есть международный роуминг
telecom_df['Predicted churn'] = (
        (telecom_df['Customer service calls'] >= 4) | (telecom_df['International plan'] == 'Yes')
)

# Сравнение с фактическим оттоком
print("\nЧасть 8.\nОценка качества прогноза:")

# Создаем confusion matrix
true_positive = ((telecom_df['Predicted churn'] == True) & (telecom_df['Churn'] == True)).sum()
true_negative = ((telecom_df['Predicted churn'] == False) & (telecom_df['Churn'] == False)).sum()
false_positive = ((telecom_df['Predicted churn'] == True) & (telecom_df['Churn'] == False)).sum()
false_negative = ((telecom_df['Predicted churn'] == False) & (telecom_df['Churn'] == True)).sum()

print(
    f"Истинно положительные (TP): {true_positive}"
    f"\nИстинно отрицательные (TN): {true_negative}"
    f"\nЛожноположительные (FP): {false_positive}"
    f"\nЛожноотрицательные (FN): {false_negative}"
)

# Вычисляем проценты ошибок
total_actual_negative = (telecom_df['Churn'] == False).sum()  # Всех активных клиентов
total_actual_positive = (telecom_df['Churn'] == True).sum()   # Всех потерянных клиентов

if total_actual_negative > 0:
    false_positive_rate = false_positive * 100 / total_actual_negative
    print(f"Процент ошибок I рода (ложноположительных): {round(false_positive_rate, 2)}%")
else:
    print("Нет активных клиентов для вычисления ошибок I рода")

if total_actual_positive > 0:
    false_negative_rate = false_negative * 100 / total_actual_positive
    print(f"Процент ошибок II рода (ложноотрицательных): {round(false_negative_rate, 2)}%")
else:
    print("Нет потерянных клиентов для вычисления ошибок II рода")

# Общая точность
accuracy = (true_positive + true_negative) * 100 / len(telecom_df)
print(f"\nОбщая точность прогноза: {round(accuracy, 2)}%")
