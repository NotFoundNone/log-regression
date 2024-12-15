import numpy as np

def sigmoid(z):
    """
    Вычисляет сигмоиду от входного значения z.
    Формула: sigmoid(z) = 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    """
    Вычисляет значение функции стоимости (log loss).
    """
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -(1/m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    return cost

def gradient_descent(X, y, theta, alpha, num_iter):
    """
    Выполняет градиентный спуск для минимизации функции стоимости.
    """
    m = len(y)
    cost_history = []

    for i in range(num_iter):
        gradient = (1 / m) * (X.T @ (sigmoid(X @ theta) - y))
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        # Прерывание при обнаружении числовой нестабильности
        if np.isnan(cost) or np.isinf(cost):
            print(f"Прерывание: alpha={alpha}, итерация={i}")
            return theta, cost_history

    return theta, cost_history

def predict(X, theta):
    """
    Предсказывает классы (0 или 1) для данных на основе обученной модели.
    """
    return sigmoid(X @ theta) >= 0.5
