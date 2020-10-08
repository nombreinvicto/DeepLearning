def depression_degree(x):
    if 0 <= x < 4.5:
        y = 1
    elif 4.5 <= x < 10.5:
        y = 2
    elif 10.5 <= x < 14.5:
        y = 3
    elif 14.5 <= x < 19.5:
        y = 4
    elif 19.5 <= x:
        y = 5

    return y