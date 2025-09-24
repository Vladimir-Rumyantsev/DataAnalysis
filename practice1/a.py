"""
Школьнику на уроке математики задали задачу по нахождению суммы первых N членов числового ряда,
имеющего вид: S = 2/3 + 1 + 6/5 + ... + (2*i)/(i+2) + .... Напишите программу, которая посчитает
эту сумму для заданного значения N. Ответ представьте в виде десятичной дроби с точностью до 0.001.

Формат ввода:
Во входных данных - единственное целое положительное число N, 1 ≤ N ≤ 100.

Формат вывода:
Выведите единственное вещественное число — ответ на задачу.
"""


import numbers


def input_number(
        prompt: str = "Введите число: ",
        min_val: numbers = 0,
        max_val: numbers = float("inf"),
        data_type: type = int
) -> numbers:
    while True:
        try:
            value = data_type(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"Error: Value must be between {min_val} and {max_val}")
        except ValueError:
            print("Error: Invalid input. Expected number.")


def main():
    n: int = input_number(
        prompt="",
        min_val=1,
        max_val=100,
        data_type=int
    )
    result: float = 0
    for i in range(1, n + 1):
        result += (2 * i) / (i + 2)
    print(round(result, 3))


if __name__ == '__main__':
    main()
