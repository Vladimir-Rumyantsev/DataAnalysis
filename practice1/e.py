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
