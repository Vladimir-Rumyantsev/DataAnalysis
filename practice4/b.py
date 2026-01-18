import pandas as pd
import matplotlib.pyplot as plt


# 1. Загрузите данные из файла
print(
    f"\nЧасть 1."
    f"\nИдёт загрузка данных..."
)
athletes_df = pd.read_csv('athlete_events.csv')
print(
    f"Данные загружены! Размер данных: {athletes_df.shape}"
)

# 2. Определите количество значений каждого из признаков
print(
    f"\nЧасть 2."
    f"\nКоличество непустых значений по каждому признаку:"
    f"\n{athletes_df.count()}"
    f"\n\nОбщая информация о данных:"
)
athletes_df.info()

# Определение признаков с наибольшим количеством пропусков
missing_values = athletes_df.isnull().sum()
print(
    f"\nКоличество пропущенных значений по каждому признаку:"
    f"\n{missing_values}"
)
max_missing_feature = missing_values.idxmax()
print(
    f"\nНаибольшее количество пропусков в признаке: "
    f"'{max_missing_feature}' ({missing_values[max_missing_feature]} пропусков)"
)

# 3. Выведите статистическую информацию
print(
    f"\nЧасть 3."
    f"\nСтатистика по возрастным и физическим характеристикам:"
    f"\n{athletes_df[['Age', 'Height', 'Weight']].describe()}"
)

# 4. Ответы на вопросы

# 4.1) Сколько лет было самому молодому участнику олимпийских игр в 1992 году?
athletes_1992 = athletes_df[athletes_df['Year'] == 1992]
youngest_1992 = athletes_1992.loc[athletes_1992['Age'].idxmin()]
print(
    f"\nЧасть 4.1."
    f"\nСамый молодой участник ОИ 1992 года:"
    f"\n  Возраст: {youngest_1992['Age']} лет"
    f"\n  Имя: {youngest_1992['Name']}"
    f"\n  Дисциплина: {youngest_1992['Event']}"
)

# 4.2) Список всех видов спорта
all_sports = athletes_df['Sport'].unique()
print(
    f"\nЧасть 4.2."
    f"\nВсего уникальных видов спорта: {len(all_sports)}"
    f"\nСписок всех видов спорта:"
    f"\n{all_sports}"
)

# 4.3) Средний рост теннисисток в 2000 году
female_tennis_2000 = athletes_df[
    (athletes_df['Year'] == 2000) &
    (athletes_df['Sex'] == 'F') &
    (athletes_df['Sport'] == 'Tennis')
]
average_height = female_tennis_2000['Height'].mean()
print(
    f"\nЧасть 4.3."
    f"\nСредний рост теннисисток в 2000 году: {round(average_height, 2)} см"
)

# 4.4) Золотые медали Китая в настольном теннисе в 2008 году
china_gold_table_tennis_2008 = athletes_df[
    (athletes_df['Year'] == 2008) &
    (athletes_df['NOC'] == 'CHN') &
    (athletes_df['Sport'] == 'Table Tennis') &
    (athletes_df['Medal'] == 'Gold')
]
gold_count = len(china_gold_table_tennis_2008)
print(
    f"\nЧасть 4.4."
    f"\nКитай выиграл {gold_count} золотых медалей в настольном теннисе в 2008 году"
)

# 4.5) Изменение количества видов спорта между 1988 и 2004 годами
summer_1988 = athletes_df[
    (athletes_df['Year'] == 1988) &
    (athletes_df['Season'] == 'Summer')
]
summer_2004 = athletes_df[
    (athletes_df['Year'] == 2004) &
    (athletes_df['Season'] == 'Summer')
]

sports_1988 = summer_1988['Sport'].nunique()
sports_2004 = summer_2004['Sport'].nunique()
change = sports_2004 - sports_1988
print(
    f"\nЧасть 4.5."
    f"\nКоличество видов спорта на летних ОИ:"
    f"\n  1988 год: {sports_1988}"
    f"\n  2004 год: {sports_2004}")
print(
      f"  Изменение: {change} (увеличилось на {abs(change)} видов)" if change >= 0 else
      f"  Изменение: {change} (уменьшилось на {abs(change)} видов)"
)

# 4.6) Гистограмма распределения возраста мужчин-керлингистов в 2014 году
curling_male_2014 = athletes_df[
    (athletes_df['Year'] == 2014) &
    (athletes_df['Sport'] == 'Curling') &
    (athletes_df['Sex'] == 'M')
]

plt.figure(figsize=(10, 6))
plt.hist(curling_male_2014['Age'].dropna(), bins=15, edgecolor='black', alpha=0.7)
plt.title('Распределение возраста мужчин-керлингистов на ОИ 2014 года', fontsize=14)
plt.xlabel('Возраст (лет)', fontsize=12)
plt.ylabel('Количество спортсменов', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print(
    f"\nЧасть 4.6."
    f"\nГистограмма построена"
)

# 4.7) Данные по зимней олимпиаде 2006 года: страны с медалями и средний возраст
winter_2006 = athletes_df[
    (athletes_df['Year'] == 2006) &
    (athletes_df['Season'] == 'Winter')
]
winter_2006_with_medals = winter_2006[winter_2006['Medal'].notna()]
grouped = winter_2006_with_medals.groupby('NOC').agg(
    total_medals=('Medal', 'count'),
    average_age=('Age', 'mean')
).reset_index()
print(
    f"\nЧасть 4.7."
    f"\nСтраны, завоевавшие медали на зимней ОИ 2006 года:"
    f"\n{grouped.sort_values('total_medals', ascending=False).head(10)}"
)

# 4.8) Сводная таблица медалей по достоинствам для зимней олимпиады 2006 года
medalists_2006 = winter_2006[winter_2006['Medal'].notna()]
pivot_table = medalists_2006.pivot_table(
    index='NOC',
    columns='Medal',
    values='ID',
    aggfunc='count',
    fill_value=0
)
print(
    f"\nЧасть 4.8."
    f"\nСводная таблица медалей по странам (зимняя ОИ 2006):"
    f"\n{pivot_table.head(15)}"
)
