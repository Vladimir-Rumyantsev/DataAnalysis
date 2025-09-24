"""
Студент ИКНТ Андрей сидел в библиотеке и читал книжку по математическому анализу.
Книжка была очень интересная, с огромными формулами и загадочными выкладками, в ней было ровно Y страниц.
Андрей не мог оторваться, он читал и читал. За час Андрей с наслаждением прочитал X страниц книги.
На следующий день он читал за час на две страницы больше, и читал уже не один час, а два.
На третий день Андрей снова читал за час на две страницы больше и читал еще на час дольше.
И так каждый новый день он читал за час на две страницы больше и читал он на час дольше.
За сколько дней Андрей прочитает всю книгу?

Формат ввода:
Входные данные содержат целое число X – количество страниц, которые Андрей читал за один час
в первый день (0 < X ≤ 100000), целое число Y – количество страниц в книге (0 < Y ≤ 100000).

Формат вывода:
Требуется вывести одно число – количество дней, за которые Андрей прочитает книгу.
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
        max_val=100000,
        data_type=int
    )
    y: int = input_number(
        prompt="",
        min_val=1,
        max_val=100000,
        data_type=int
    )
    day: int = 1
    hours_of_reading: int = x
    already_read: int = x
    while already_read < y:
        day += 1
        hours_of_reading += 2
        already_read += day * hours_of_reading
    print(day)


if __name__ == '__main__':
    main()
