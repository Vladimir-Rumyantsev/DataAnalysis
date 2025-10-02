"""
Компания друзей собралась пойти в поход. Забот и затрат при подготовке похода оказалось много: кто-то закупал еду,
кто-то брал в аренду снаряжение, кто-то заказывал транспорт. Когда всё было готово, друзья решили подсчитать,
кто сколько денег потратил и, соответственно, кто кому сколько денег должен перевести.
Статей расходов оказалось очень много, участников похода было тоже много, поэтому сделать все расчеты вручную
оказалось затруднительно. Напишите программу, которая по информации о том, кто сколько денег потратил,
определит: кто, кому и сколько денег должен перевести, чтобы расходы всех участников похода оказались одинаковыми
(с точностью до копейки). Количество переводов при этом должно быть как можно меньше.

Входные данные:
В первой строке через пробел записаны имена всех участников похода. Имена уникальны,
каждое имя состоит из латинских букв. Длина каждого имени – не более 20 символов, количество имен – не более 100.
Во второй строке записано одно целое число N – количество покупок, которое было сделано при подготовке похода.
Далее следует N строк, каждая из которых описывает одну покупку и содержит имя того, кто эту покупку оплачивал,
и одно целое число – сумму покупки. Имя и число разделены пробелом. Гарантируется, что имя есть в общем списке
участников похода.

Выходные данные:
В первой строке выведите одно число M – минимальное количество переводов, которые нужно совершить.
Далее выведите M строк, в каждой указав два имени и вещественное число через пробел:
кто, кому и сколько должен перевести. Все суммы переводов должны быть округлены до 2 знаков после запятой.
Если существует несколько вариантов переводов, то выведите любой из них – главное, чтобы их количество было минимальным.

Пример:

Входные данные:
Ivan Aleksej Igor
3
Ivan 500
Aleksej 100
Ivan 200

Выходные данные:
2
Aleksej Ivan 166.67
Igor Ivan 266.67

Комментарий к примеру:
Иван потратил в сумме 700, но двое остальных переведут ему в сумме 433.34, итого затраты Ивана составят 266.66.
Алексей потратил 100 и должен перевести Ивану 166.67, в итоге его затраты составят 266.67.
Игорь ничего не потратил на покупки, и все его затраты – это перевод 266.67 Ивану. Таким образом,
затраты всех трех друзей окажутся одинаковыми с точностью до копейки (затраты Ивана на 1 копейку меньше,
чем у остальных). Очевидно, что одним переводом добиться равных затрат не получится,
поэтому 2 – это минимальное количество переводов. Заметим, что такой набор переводов – не единственно возможный.
Например, Алексей мог бы перевести 166.67 Игорю, а Игорь перевести Ивану 433.34.
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


def validate_names(names):
    if len(names) > 100:
        print("\nОшибка. Слишком много имён. Их должно быть не более ста.")
        return False

    if len(names) != len(set(names)):
        print("\nОшибка. Есть повторяющиеся имена.")
        return False

    for name in names:
        if not (1 <= len(name) <= 20):
            print(f'\nОшибка. Имя "{name}" некорректной длины (должно быть от 1 символа до 20 включительно).')
            return False

        if not name.isalpha():
            print(f'\nОшибка. Имя "{name}" состоит не только из латинских букв.')
            return False

        if not name[0].isupper() or not name[1:].islower():
            print(f'\nОшибка. Имя "{name}" не начинается с большой буквы или не продолжается строчными буквами.')
            return False

    return True


def main():
    names: list[str] = input("\nВведите через пробел имена всех участников: ").split()
    if not validate_names(names): return
    n: int = input_number(
        prompt="\nВведите количество покупок N (число от 1 до 1000 включительно): ",
        min_val=1,
        max_val=1000,
        data_type=int
    )

    expenses: dict = {name: 0 for name in names}

    print(f'Введите {n} покупок, по одной в каждой строчке в формате "<Имя> <Сумма покупки>".')
    for i in range(n):
        while True:
            try:
                new_data: list[str] = input(f"{i + 1}.: ").split()
                if len(new_data) == 2 and new_data[0] in names:
                    cost: int = int(float(new_data[1]) * 100)
                    if cost > 0:
                        expenses[new_data[0]] += cost
                        break
                    else:
                        print("Ошибка. Цена покупки не может быть <= 0.")
                else:
                    print("Ошибка. Проверьте формат ввода и имя участника.")
            except (ValueError, TypeError, OverflowError):
                print("Ошибка. Сумма должна быть числом.")

    total: int = sum(expenses.values())
    average: float = total / len(names)

    participants: list = []
    for name in names:
        spent = expenses[name]
        balance = spent - average
        participants.append((name, balance))

    participants.sort(key=lambda x: x[1])

    transfers = []
    left = 0
    right = len(participants) - 1

    while left < right:
        debtor, debt = participants[left]
        creditor, credit = participants[right]

        if abs(debt) < 1 and abs(credit) < 1:
            break

        amount = min(-debt, credit)

        transfers.append((debtor, creditor, amount / 100))

        participants[left] = (debtor, debt + amount)
        participants[right] = (creditor, credit - amount)

        if abs(participants[left][1]) < 1:
            left += 1
        if abs(participants[right][1]) < 1:
            right -= 1

    print(f"\n{len(transfers)}")
    for transfer in transfers:
        print(f"{transfer[0]} {transfer[1]} {transfer[2]:.2f}")
    print("\n————————————————————————————————————————————————————————————————\n")


if __name__ == '__main__':
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print("\nПрограмма завершила работу.")
            break
        except Exception as e:
            print(f"\nПроизошла ошибка: {e}")
            break
