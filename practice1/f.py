"""
Даны два натуральных числа N и S. Найдите минимальное N-значное натуральное число,
сумма цифр которого равна S. Не забывайте, что ведущих нулей в числе быть не может.

Формат ввода:
В первой строке записано одно целое число N – количество разрядов числа (1 ≤ N ≤ 20).
Во второй строке записано одно целое число S - сумма цифр числа (1 ≤ S ≤ 100).

Формат вывода:
Выведите искомое натуральное число. Если такого числа не существует, выведите "NO".
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
        max_val=20,
        data_type=int
    )
    s: int = input_number(
        prompt="",
        min_val=1,
        max_val=100,
        data_type=int
    )

    if s < 1 or s > 9 * n:
        print("NO")
        return

    result: list = []
    remaining_sum: int = s

    for i in range(n):
        digits_after = n - i - 1
        if i == 0:
            digit = max(1, remaining_sum - 9 * digits_after)
        else:
            digit = max(0, remaining_sum - 9 * digits_after)

        result.append(str(digit))
        remaining_sum -= digit

    print(''.join(result))


if __name__ == '__main__':
    main()
