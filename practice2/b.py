"""
Пароль считается надежным, если его длина составляет не менее 12 символов,
при этом он должен содержать хотя бы одну заглавную букву, хотя бы одну строчную букву,
хотя бы одну цифру и хотя бы один спецсимвол. Любые другие символы в пароле запрещены.
Напишите программу, которая по указанному списку паролей определяет, какие из них являются надежными, а какие – нет.

Допустимые заглавные буквы: A, B, …, Z (все символы латинского алфавита).
Допустимые строчные буквы: a, b, …, z (все символы латинского алфавита).
Допустимые спецсимволы: !, @, #, $, %, &, *, +.

Формат ввода:
В первой строке записано одно целое число N – количество паролей, которые нужно проверить, 1 ≤ N ≤ 100.
Далее следует N строк, в каждой из которых записан один пароль.
Гарантируется, что в паролях нет недопустимых символов, а длина каждого – не больше 100 символов.

Формат вывода:
Выведите N строк, по одной для каждого пароля из входных данных. В каждой выведите “Valid” (без кавычек),
если пароль надежный, и “Invalid” (без кавычек), если ненадежный.

Пример:

Ввод:
2
123456789aA!
123IsAVeryDifficultPassword

Вывод:
Valid
Invalid
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


def is_password_valid(password: str):
    if len(password) < 12: return False

    has_digit: bool = False
    has_uppercase: bool = False
    has_lowercase: bool = False
    has_special_char: bool = False
    allowed_special_chars: set = {'!', '@', '#', '$', '%', '&', '*', '+'}

    for char in password:
        if char.isdigit():
            has_digit = True
        elif char.isupper():
            has_uppercase = True
        elif char.islower():
            has_lowercase = True
        elif char in allowed_special_chars:
            has_special_char = True
        else:
            return False

    return has_digit and has_uppercase and has_lowercase and has_special_char


def main():
    num_passwords: int = input_number(
        prompt="",
        min_val=1,
        max_val=100,
        data_type=int
    )

    results: list = []

    for _ in range(num_passwords):
        if is_password_valid(input()):
            results.append("Valid")
        else:
            results.append("Invalid")

    print(*results, sep='\n')


if __name__ == '__main__':
    main()
