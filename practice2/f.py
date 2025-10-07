"""
Файл mbox.txt содержит метаданные почтового сервера. Мы знаем, что строка с адресом автора письма
начинается с "From ". Найти адреса всех авторов сообщений и найти того из них, кто пишет больше всех писем.
Исходный файл можно взять по ссылке: https://www.py4e.com/code3/mbox.txt
Для создания локальной программы вы можете загрузить файл себе на компьютер.
Либо можно выполнять это задание в Jupyter Notebook. Для загрузки данных можно использовать следующий код:
# импортируем библиотеку для доступа к файлам в интернете
import requests
# в переменной mbox хранится текст для работы
mbox = requests.get(' https://www.py4e.com/code3/mbox.txt').text

# преобразуем текст в список, где каждый объект — это одна строка в файле
all_lines = mbox.split('\n')

"""


import re
import requests
from collections import Counter


def download_data() -> list[str]:
    return requests.get('https://www.py4e.com/code3/mbox.txt').text.split('\n')


def main() -> None:
    counter: Counter = Counter()

    for line in download_data():
        if re.match(r"[Ff]rom:.+@", line) is not None:
            line_split: list = line.split()
            if len(line_split) >= 2:
                counter[line_split[1].lower()] += 1

    max_count: int = 0
    most_common: list[tuple[str, int]] = []

    for email, count in counter.items():
        if count > max_count:
            most_common = [(email, count)]
            max_count = count
        elif count == max_count:
            most_common.append((email, count))

    if not most_common:
        print(f'\nНе найден ни один почтовый отправитель.')
        return
    elif len(most_common) == 1:
        key, count = most_common[0]
        print(f'\nСамый частый почтовый отправитель: "{key}" встречается {count} раз(а).')
    else:
        print(
            f'\nНайдено несколько самых частых почтовых отправителей, '
            f'каждый из которых встречается {most_common[0][1]} раз(а):'
        )
        for i in most_common:
            print(f'Отправитель: "{i[0]}".')

    keys_list = list(counter.keys())
    print("\nВсе почтовые отправители:", *keys_list, sep="\n")


if __name__ == '__main__':
    main()
