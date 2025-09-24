"""
На заводе открывается производство нового вида деталей, которые вытачиваются из цилиндрических
заготовок, сделанных из специального сплава. Для производства одной детали необходима заготовка
массой a грамм. При этом после обработки остаются опилки массой b грамм. Опилки можно собрать,
и если их в общей сложности накопилось a грамм, то их можно переплавить в новую заготовку.
На завод поступило n заготовок. Сколько деталей из них получится сделать?

Формат ввода:
В первой строке содержится целое положительное число a — масса одной заготовки.
Во второй строке — целое неотрицательное число b — масса опилок от изготовления одной детали. 0 ≤ b < a ≤ 10^7.
В третьей строке содержится целое неотрицательное число n — начальное количество заготовок (0 ≤ n ≤ 10^9).

Формат вывода:
Выведите одно число — максимальное количество деталей, которое можно изготовить из имеющихся заготовок.
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
    a: int = input_number(
        prompt="",
        min_val=1,
        max_val=10**7,
        data_type=int
    )
    b: int = input_number(
        prompt="",
        min_val=0,
        max_val=a-1,
        data_type=int
    )
    n: int = input_number(
        prompt="",
        min_val=0,
        max_val=10**9,
        data_type=int
    )

    if n == 0 or b == 0:
        print(n)
        return

    print(((n*a)-b)//(a-b))


if __name__ == '__main__':
    main()
