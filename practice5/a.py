import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
Альтернативное задание.

1. Создайте серии:
а) Серия X, содержащая числа от 0 до 99 (np.arange(100))
б) Серия Y1, содержащая 100 случайных чисел с нормальным распределением с мат. ожиданием 0 и дисперсией 1
   (np.random.normal(0,1,100))
в) Серия Y2, содержащая 100 чисел, вычисляемых по формуле y2[i]=x[i]+e[i], где e[i] – случайное целое число
   из диапазона [-2;2] с равномерным распределением (y2 = x + np.random.randint(-2,3,100)).
2. Создайте датафрейм из построенных на предыдущем шаге серий.
3. Постройте точечную диаграмму (диаграмму рассеяния) по признакам X и Y2.
4. Постройте линейную диаграмму (график) зависимости Y1 от X.
5. Постройте гистограмму частот для признака Y1. На гистограмме должно быть 10 диапазонов значений.
6. Разбейте данные на 3 группы по значению разности X-Y2 (X-Y2<0; X-Y2=0; X-Y2>0).
   В одной области для каждой группы постройте boxplot (диаграмму «ящик с усами») для признака Y1.
"""


# 1. Создание серий
np.random.seed(42)  # Для воспроизводимости результатов
X = pd.Series(np.arange(100))  # а) Серия X от 0 до 99
Y1 = pd.Series(np.random.normal(0, 1, 100))  # б) Серия Y1 ~ N(0,1)
Y2 = X + np.random.randint(-2, 3, 100)  # в) Серия Y2 = X + e

# 2. Создание датафрейма
df = pd.DataFrame(
    {
        "X": X,
        "Y1": Y1,
        "Y2": Y2
    }
)

print(
    f"\nПостроенный датафрейм:\n{df}"
    f"\n\nОсновная информация о данных:"
)
df.info()

# 3. Точечная диаграмма X и Y2
plt.figure(figsize=(10, 6))
plt.scatter(df['X'], df['Y2'], alpha=0.7, edgecolors='black', linewidth=0.5)
plt.title('Точечная диаграмма X и Y2')
plt.xlabel('X')
plt.ylabel('Y2')
plt.grid(True, alpha=0.3)
plt.show()

# 4. Линейная диаграмма Y1 от X
plt.figure(figsize=(10, 6))
plt.plot(df['X'], df['Y1'], color='green', linewidth=2)
plt.title('Линейная диаграмма Y1 от X')
plt.xlabel('X')
plt.ylabel('Y1')
plt.grid(True, alpha=0.3)
plt.show()

# 5. Гистограмма для Y1 (10 диапазонов)
plt.figure(figsize=(10, 6))
plt.hist(df['Y1'], bins=10, edgecolor='black', alpha=0.7)
plt.title('Гистограмма частот для Y1 (10 интервалов)')
plt.xlabel('Y1')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# 6. Разбиение на группы и boxplot для Y1
# Вычисляем разность X-Y2
df['X-Y2'] = df['X'] - df['Y2']

# Создаем группы
df['Group'] = '>0'  # значение по умолчанию
df.loc[df['X-Y2'] < 0, 'Group'] = '<0'
df.loc[df['X-Y2'] == 0, 'Group'] = '=0'

# Определяем порядок групп для boxplot
groups_order = ['<0', '=0', '>0']
box_data = [df[df['Group'] == group]['Y1'] for group in groups_order]

plt.figure(figsize=(10, 6))
plt.boxplot(
    box_data,
    tick_labels=groups_order,
    patch_artist=True,
    boxprops=dict(facecolor='lightblue', color='blue'),
    medianprops=dict(color='red')
)
plt.title('Boxplot Y1 для групп (X-Y2)')
plt.xlabel('Группа (X-Y2)')
plt.ylabel('Y1')
plt.grid(True, alpha=0.3, axis='y')
plt.show()
