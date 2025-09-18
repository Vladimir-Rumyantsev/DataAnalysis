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
        max_val=100,
        data_type=int
    )
    y: int = input_number(
        prompt="",
        min_val=0,
        max_val=x-1,
        data_type=int
    )
    n: int = input_number(
        prompt="",
        min_val=1,
        max_val=10**10,
        data_type=int
    )

    days: int = (n - y) // (x - y) + 1

    if days < 1:
        days = 1

    print(days)


if __name__ == '__main__':
    main()
