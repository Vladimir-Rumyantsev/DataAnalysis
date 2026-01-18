import numpy as np
import pandas as pd


# 1. Сгенерировать массив и преобразовать в Series
M = 1.0
s = 1.0
start_numbers_series = pd.Series(np.random.normal(loc=M, scale=s, size=1000))
print(
    f"\nЧасть 1."
    f"\nСгенерированные значения:\n{start_numbers_series}"
)

# 2. Вычислить долю значений в диапазоне (M-s; M+s)
mask_1 = (start_numbers_series >= M - s) & (start_numbers_series <= M + s)
fraction_1 = mask_1.sum() / len(start_numbers_series)
print(
    f"\nЧасть 2."
    f"\nДоля значений в диапазоне (M-s; M+s): {round(fraction_1*100, 2)}%"
    f"\nТеоретическое значение (правило 68-95-99.7): 68.27%"
    f"\nРасхождение с теорией: {round(abs(fraction_1 - 0.6827)*100, 2)}%"
)

# 3. Вычислить долю значений в диапазоне (M-3s; M+3s)
mask_2 = (start_numbers_series > M - 3*s) & (start_numbers_series < M + 3*s)
fraction_2 = mask_2.sum() / len(start_numbers_series)
print(
    f"\nЧасть 3."
    f"\nДоля значений в диапазоне (M-3s; M+3s): {round(fraction_2*100, 2)}%"
    f"\nТеоретическое значение (правило 68-95-99.7): 99.73%"
    f"\nРасхождение с теорией: {round(abs(fraction_2 - 0.9973)*100, 2)}%"
)

# 4. Заменить каждое значение на квадратный корень
root_series = pd.Series(np.sqrt(start_numbers_series))
print(
    f"\nЧасть 4."
    f"\nЗначения после извлечения корня:"
    f"\n{root_series}"
    f"\nПредупреждение возникает при попытке извлечь квадратный корень из отрицательных чисел. "
    f"Эти значения заменяются на NaN (Not a Number)."
)

# 5. Посчитать среднее арифметическое (без учета NaN)
mean_root = root_series.mean()   # по умолчанию skipna=True
print(
    f"\nЧасть 5."
    f"\nСреднее арифметическое квадратных корней (без NaN): {round(mean_root, 2)}"
)

# 6. Создать DataFrame и вывести первые 6 строк
df = pd.DataFrame({
    "number": start_numbers_series,
    "root": root_series
})
print(
    f"\nЧасть 6."
    f"\nПервые 6 строк DataFrame:"
    f"\n{df.head(6)}"
)

# 7. Найти записи с корнем в диапазоне от 1.8 до 1.9
result = df.query('1.8 <= root <= 1.9')
print(
    f"\nЧасть 7."
    f"\nЗаписей с корнем в диапазоне [1.8, 1.9]: {len(result)}"
    f"\n{result}"
)
