from typing import List, Tuple

def linearRegression(x: List[float], y: List[float]) -> Tuple[float, float]:
    n = len(x)

    if n != len(y):
        raise ValueError("x and y must have the same length")

    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(xi ** 2 for xi in x)
    sum_xy = sum(x[i] * y[i] for i in range(n))

    numerator = (n * sum_xy) - (sum_x * sum_y)
    denominator = (n * sum_x2) - (sum_x ** 2)

    if denominator == 0:
        raise ValueError("Cannot compute linear regression")

    m = numerator / denominator
    c = (sum_y - (m * sum_x)) / n

    return m, c


if __name__ == "__main__":
    dist = [0, 1, 2, 3, 4]
    fare = [5, 10, 15, 20, 25]

    slope, intercept = linearRegression(dist, fare)
    print(f"The slope of the data set is: {slope}")
    print(f"The intercept is: {intercept}")
    print(f"Linear Eq: y = {slope}x + {intercept}")