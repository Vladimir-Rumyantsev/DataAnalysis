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
    n: int = input_number(
        prompt="",
        min_val=1,
        max_val=100000,
        data_type=int
    )
    for num in range(n, 0, -1):
        if len(str(num)) == len(set(str(num))):
            print(num)
            return


if __name__ == '__main__':
    main()
