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
        for _ in range(10):
            input_parts: list[str] = input(f"\n{i + 1}.: ").strip().split()

            if len(input_parts) < 3:
                print("Ошибка: недостаточно данных")
                continue

            date_str: str = input_parts[0]
            price_str: str = input_parts[-1]
            pizza_name: str = ' '.join(input_parts[1:-1])

            if not pizza_name:
                print("Ошибка: отсутствует название пиццы")
                continue

            try:
                purchase_date: date = datetime.strptime(date_str, "%d.%m.%Y").date()
            except ValueError:
                print("Ошибка: неверно указана дата покупки пиццы")
                continue

            price_parts: list[str] = price_str.split('.')
            if not ((len(price_parts) == 1) or (len(price_parts) == 2 and len(price_parts[1]) <= 2)):
                print("Ошибка: неверно указана цена пиццы")
                continue
            try:
                cost: int = int(float(price_str) * 100)
            except (ValueError, OverflowError):
                print("Ошибка: неверно указана цена пиццы")
                continue
            if cost <= 0:
                print("Ошибка: цена должна быть положительной")
                continue

            pizza_name = pizza_name.capitalize()

            if pizza_name in pizza_orders:
                pizza_orders[pizza_name].append((purchase_date, cost))
            else:
                pizza_orders[pizza_name] = [(purchase_date, cost)]

            break

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
            pizza_date = purchase_tuple[0]
            pizza_cost = purchase_tuple[1]

            if pizza_date in unsorted_data_for_b:
                unsorted_data_for_b[pizza_date] += pizza_cost
            else:
                unsorted_data_for_b[pizza_date] = pizza_cost

            if (not data_for_c) or (pizza_cost == data_for_c[0][2]):
                data_for_c.append((pizza_name, pizza_date, pizza_cost))
            elif pizza_cost > data_for_c[0][2]:
                data_for_c = [(pizza_name, pizza_date, pizza_cost)]

            data_for_d += pizza_cost
            number_of_orders += 1

    data_for_a.sort(key=lambda x: -x[1])
    data_for_b: list[tuple[date, int]] = sorted(unsorted_data_for_b.items(), key=lambda x: x[0])
    data_for_d /= (number_of_orders * 100)

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
