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
        data_type=int
    )
    b: int = input_number(
        prompt="",
        data_type=int
    )
    n: int = input_number(
        prompt="",
        data_type=int
    )

    if 0 <= b < a <= 10 ** 7 and 0 <= n <= 10 ** 9:
        if a <= 10 ** 5 or n <= 10 ** 7:
            details = n
            refuse = n * b
            n = 0
            while refuse >= a:
                n += refuse // a
                refuse = refuse % a
                details += n
                refuse += n * b
                n = 0
        else:
            x = (n * b) // a
            details = ((n * a) - b) // (a - b)
        print(details)


if __name__ == '__main__':
    main()
