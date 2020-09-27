import time


def func(x):
    return (1 / 6) * (x ** 6) - (5 / 4) * (x ** 4) + 2 * (x ** 2)


def d1_func(x):
    return (x ** 5) - 5 * (x ** 3) + 4 * x


def d2_func(x):
    return 5 * (x ** 4) - 15 * (x ** 2) + 4


x_prev = 0.65

while True:
    d1 = d1_func(x_prev)
    d2 = d2_func(x_prev)

    x_next = x_prev - (d1 / d2)

    print(f"First Derivative d1: {d1}")
    print(f"First Derivative d2: {d2}")
    print(f"X_Prev: {x_prev}")
    print(f"Func at X_Prev: {func(x_prev)}")
    print(f"X_Next: {x_next}")
    print(f"Func at X_Next: {func(x_next)}")

    print("=" * 50)

    if func(x_next) == func(x_prev):
        break
    else:
        x_prev = x_next

    #time.sleep(1)
