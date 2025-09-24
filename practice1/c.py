"""
Второкурсник ИКНТ Вова точно знает, что у него самая обворожительная улыбка.
Он старательно заботится о ней и каждый день тщательно чистит зубы.
У Вовы 2 ряда белоснежных зубов по X зубов в каждом.
На чистку одного зуба второкурсник тратит Y минут.
Каждое утро он начинает чистить зубы в тот момент, когда на его любимых электронных часах ровно 8:00.
Сколько будут показывать часы в тот момент, когда второкурсник Вова закончит чистить зубы?

Формат ввода:
Входные данные содержат целое число X – количество зубов в одном ряду (верхнем или нижнем)
во рту у второкурсника Вовы (0 < X ≤ 20), целое число Y – количество минут,
которое тратит Вова на чистку одного зуба (0 < Y ≤ 22).

Формат вывода:
Требуется вывести два числа – количество часов и минут, которые будут показывать
электронные часы Вовы, когда он дочистит зубы.
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
    x: int = input_number(
        prompt="",
        min_val=1,
        max_val=20,
        data_type=int
    )
    y: int = input_number(
        prompt="",
        min_val=1,
        max_val=22,
        data_type=int
    )

    hours: int = 8
    minutes: int = 0

    minutes += x * y * 2
    hours += minutes // 60
    minutes %= 60
    hours %= 24

    print(hours)
    print(minutes)


if __name__ == '__main__':
    main()
