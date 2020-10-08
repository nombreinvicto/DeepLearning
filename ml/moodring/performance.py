def performance(pred_degree, test_y_degree):
    if pred_degree == test_y_degree:
        perform = 'good'
    elif pred_degree > test_y_degree:
        perform = 'higher'
    elif pred_degree < test_y_degree:
        perform = 'lower'

    return perform