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
        max_val=20,
        data_type=int
    )
    s: int = input_number(
        prompt="",
        min_val=1,
        max_val=100,
        data_type=int
    )

    if s < 1 or s > 9 * n:
        print("NO")
        return

    result: list = []
    remaining_sum: int = s

    for i in range(n):
        digits_after = n - i - 1
        if i == 0:
            digit = max(1, remaining_sum - 9 * digits_after)
        else:
            digit = max(0, remaining_sum - 9 * digits_after)

        result.append(str(digit))
        remaining_sum -= digit

    if remaining_sum != 0:
        print("NO")
    else:
        print(''.join(result))


if __name__ == '__main__':
    main()
