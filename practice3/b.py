import numpy as np

# Генерация двух одномерных массивов
np.random.seed(42)  # для воспроизводимости
m1 = np.random.randint(1, 11, size=10)
m2 = np.random.randint(1, 11, size=10)

print("1. Исходные массивы:")
print(f"m1: {m1}")
print(f"m2: {m2}")

# 1. Симметрическая разность (элементы только в одном из массивов)
m3 = np.setxor1d(m1, m2)
print(f"\nm3 (симметрическая разность): {m3}")

# 2. Замена значений в m1 кратных 2 или 3 на 1
m1_modified = m1.copy()
mask = (m1_modified % 2 == 0) | (m1_modified % 3 == 0)
m1_modified[mask] = 1
print(f"\n2. Модифицированный m1: {m1_modified}")

# 3. Слияние и преобразование в матрицу 4x5
merged = np.concatenate([m1, m2])
matrix = merged.reshape(4, 5)
print(f"\n3. Матрица 4x5:\n{matrix}")

# 4. Удаление 1-го и 4-го столбцов (индексы 0 и 3)
matrix_reduced = np.delete(matrix, [0, 3], axis=1)
print(f"\n4. Матрица после удаления 1 и 4 столбцов:\n{matrix_reduced}")

# 5. Транспонирование
matrix_transposed = matrix_reduced.T
print(f"\n5. Транспонированная матрица:\n{matrix_transposed}")
