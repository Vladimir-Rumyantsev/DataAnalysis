"""
Задача 5 (1 балл). Начинающий предприниматель Александр открыл свою первую пиццерию.
Для учета заказов он использует максимально простой инструмент – записывает в блокнот информацию о дате заказа,
названии пиццы и стоимости заказа (стоимость одной и той же пиццы даже в один и тот же день может быть разной
– это зависит от дополнительных ингредиентов, которые пожелал добавить клиент, но которые Александр в своем
блокноте никак не учитывает). По прошествии нескольких дней Александр хочет извлечь из своих записей
какую-нибудь полезную информацию. Напишите программу, которая будет выводить:

а) список всех пицц с указанием, сколько раз их заказывали; список должен быть отсортирован по убыванию
   количества заказов, то есть первой в списке должна оказаться самая популярная пицца;
б) список всех дат с указанием суммарной стоимости проданных в этот день пицц;
   список должен быть отсортирован хронологически;
в) информацию о самом дорогом заказе;
г) среднюю стоимость заказа (среднее арифметическое по всем стоимостям).

Формат входных и выходных данных определите самостоятельно.
"""


import numbers
from datetime import datetime, date


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
        prompt="\nВведите количество входных данных о пиццах: ",
        min_val=1,
        max_val=1000,
        data_type=int
    )

    pizza_orders: dict[str, list[tuple[date, int]]] = {}

    print(
        f'\nВведите информацию об {n} покупок, по одной в каждой строчке '
        f'в формате "<ДД.ММ.ГГГГ> <название пиццы> <рубли.копейки>".'
    )
    for i in range(n):
        while True:
            try:
                new_data: list[str] = input(f"\n{i + 1}.: ").split()
                if len(new_data) == 3 and len(new_data[1]) >= 2:
                    purchase_date: date = datetime.strptime(new_data[0], "%d.%m.%Y").date()
                    pizza_name: str = f"{new_data[1][0].upper()}{new_data[1][1:].lower()}"
                    cost: int = int(float(new_data[2]) * 100)
                    if cost > 0:
                        if pizza_name in pizza_orders:
                            pizza_orders[pizza_name].append((purchase_date, cost))
                        else:
                            pizza_orders[pizza_name] = [(purchase_date, cost)]
                        break
                    else:
                        print("Ошибка. Вы указали неположительную цену пиццы.")
                else:
                    print("Ошибка. Проверьте формат ввода.")
            except (ValueError, TypeError, OverflowError):
                print("Ошибка. Проверьте формат ввода.")

    if not pizza_orders:
        print("Пицц нет")
        return

    data_for_a: list[tuple[str, int]] = []
    unsorted_data_for_b: dict[date, int] = {}
    data_for_c: list[tuple[str, date, int]] = []
    data_for_d: float = 0.0
    number_of_orders: int = 0

    for pizza_name in pizza_orders:
        data_for_a.append((pizza_name, len(pizza_orders[pizza_name])))

        for purchase_tuple in pizza_orders[pizza_name]:
            if purchase_tuple[0] in unsorted_data_for_b:
                unsorted_data_for_b[purchase_tuple[0]] += 1
            else:
                unsorted_data_for_b[purchase_tuple[0]] = 1

            pizza_date = purchase_tuple[0]
            pizza_cost = purchase_tuple[1]
            if (not data_for_c) or (pizza_cost == data_for_c[0][2]):
                data_for_c.append((pizza_name, pizza_date, pizza_cost))
            elif pizza_cost > data_for_c[0][2]:
                data_for_c = [(pizza_name, pizza_date, pizza_cost)]

            data_for_d += purchase_tuple[1]
            number_of_orders += 1

    data_for_a.sort(key=lambda x: x[1])
    data_for_b: list[tuple[date, int]] = sorted(unsorted_data_for_b.items(), key=lambda x: x[0])
    data_for_d /= number_of_orders * 100

    # print(f"\nа) список всех пицц:", *data_for_a, sep="\n")
    # print(f"\n————————————————————————————————————————————————————————————————\n"
    #       f"\nб) список всех дат:", *data_for_b, sep="\n")
    # print(f"\n————————————————————————————————————————————————————————————————\n"
    #       f"\nв) информация о самом дорогом заказе:", *data_for_c, sep="\n")
    # print(f"\n————————————————————————————————————————————————————————————————\n"
    #       f"\nг) средняя стоимость заказа: {round(data_for_d, 2)}")
    print(f"\n————————————————————————————————————————————————————————————————\n"
          f"\nа) список всех пицц:")
    for i in data_for_a:
        print(f"{i[0]}: {i[1]}")

    print(f"\n————————————————————————————————————————————————————————————————\n"
          f"\nб) список всех дат:")
    for i in data_for_b:
        print(f"{i[0].strftime("%d.%m.%Y")}: {i[1]}")

    print(f"\n————————————————————————————————————————————————————————————————\n"
          f"\nв) информация о самом дорогом заказе:")
    for i in data_for_c:
        print(f"{i[0]}: {i[1].strftime("%d.%m.%Y")} {round(i[2] / 100, 2)}")

    print(f"\n————————————————————————————————————————————————————————————————\n"
          f"\nг) средняя стоимость заказа: {round(data_for_d, 2)}₽")


if __name__ == '__main__':
    main()
