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
