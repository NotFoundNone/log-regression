from processing_data import load_data_from_file, normalization_func, add_bias_column, add_nonlinear_features
from logistic_regression import sigmoid, compute_cost, gradient_descent, predict
import numpy as np
import matplotlib.pyplot as plt

def test_learning_rates(X, y, alphas, num_iter):
    """
    Тестирует разные скорости обучения (alphas) для заданного набора данных (X, y).
    """
    costs = {}
    thetas = {}

    for alpha in alphas:
        theta = np.zeros(X.shape[1])
        optimized_theta, cost_history = gradient_descent(X, y, theta, alpha, num_iter)

        if len(cost_history) == num_iter:
            costs[alpha] = cost_history
            thetas[alpha] = optimized_theta
        else:
            print(f"Прерывание на alpha={alpha} из-за больших градиентов.")

    return costs, thetas

def plot_cost_vs_alpha(costs, alphas, title):
    """
    Строит график изменения функции стоимости для каждого alpha.
    """
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        if alpha in costs:
            plt.plot(range(1, len(costs[alpha]) + 1), costs[alpha], label=f'alpha={alpha}')
    plt.xlabel("Итерации")
    plt.ylabel("Функция стоимости")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Визуализация сравнения моделей
def plot_comparison(X_original, X_nonlinear, y, theta_original, theta_nonlinear):
    """
    Сравнивает границы решений для модели с исходными и нелинейными признаками.
    """
    plt.figure(figsize=(14, 6))

    # График для модели без нелинейных признаков
    plt.subplot(1, 2, 1)
    plt.title("Без нелинейных признаков")
    plot_decision_boundary(X_original, y, theta_original, "Vibration", "Irregularity")

    # График для модели с нелинейными признаками
    plt.subplot(1, 2, 2)
    plt.title("С нелинейными признаками")
    plot_decision_boundary(X_nonlinear, y, theta_nonlinear, "Vibration", "Irregularity")

    plt.tight_layout()
    plt.show()

def plot_decision_boundary(X, y, theta, x_label, y_label):
    """
    Строит границу решения и отображает данные.
    """
    # Разделяем данные по классам
    positives = X[y == 1]
    negatives = X[y == 0]

    # График данных
    plt.scatter(positives[:, 1], positives[:, 2], c='g', label='Fault = 1 (Positive)', edgecolor='k')
    plt.scatter(negatives[:, 1], negatives[:, 2], c='r', label='Fault = 0 (Negative)', edgecolor='k')

    # Граница решения (только для первых двух признаков)
    x_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    y_values = -(theta[0] + theta[1] * x_values) / theta[2]
    plt.plot(x_values, y_values, label='Decision Boundary', color='blue')

    # Настройки графика
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

def main():
    # Путь к файлу с данными
    file_path = "ex2data1.txt"

    # Загрузка данных
    data = load_data_from_file(file_path)

    # Нормализация данных
    normalized_data, mean_std = normalization_func(data, exclude_columns=['Fault'])

    # Разделение на признаки и метки
    X_original = normalized_data[['Vibration', 'Irregularity']]
    y = normalized_data['Fault']

    # Добавление нелинейных признаков
    data_with_nonlinear = add_nonlinear_features(normalized_data, ['Vibration', 'Irregularity'])
    X_nonlinear = data_with_nonlinear.drop(columns=['Fault'])

    # Добавление столбца единиц
    X_original = add_bias_column(X_original)
    X_nonlinear = add_bias_column(X_nonlinear)

    # Преобразование в NumPy массивы
    X_original = X_original.to_numpy()
    X_nonlinear = X_nonlinear.to_numpy()
    y = y.to_numpy()

    # Настройки
    alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    iterations = 10000

    # Тестирование для линейных признаков
    print("=== Тестирование для линейных признаков ===")
    costs_linear, thetas_linear = test_learning_rates(X_original, y, alphas, iterations)
    best_alpha_linear = min(costs_linear, key=lambda a: costs_linear[a][-1])
    theta_original = thetas_linear[best_alpha_linear]
    print(f"Лучшее alpha для линейных признаков: {best_alpha_linear}")

    # Тестирование для нелинейных признаков
    print("\n=== Тестирование для нелинейных признаков ===")
    costs_nonlinear, thetas_nonlinear = test_learning_rates(X_nonlinear, y, alphas, iterations)
    best_alpha_nonlinear = min(costs_nonlinear, key=lambda a: costs_nonlinear[a][-1])
    theta_nonlinear = thetas_nonlinear[best_alpha_nonlinear]
    print(f"Лучшее alpha для нелинейных признаков: {best_alpha_nonlinear}")

    # Вычисление точности моделей
    y_pred_original = predict(X_original, theta_original)
    y_pred_nonlinear = predict(X_nonlinear, theta_nonlinear)

    accuracy_original = np.mean(y_pred_original == y) * 100
    accuracy_nonlinear = np.mean(y_pred_nonlinear == y) * 100

    # Вывод точности
    print("\n=== Точность моделей ===")
    print(f"Линейные признаки: точность = {accuracy_original:.2f}%")
    print(f"Нелинейные признаки: точность = {accuracy_nonlinear:.2f}%")

    # Визуализация сравнения моделей
    print("\n=== Визуализация сравнения моделей ===")
    plot_comparison(X_original, X_nonlinear, y, theta_original, theta_nonlinear)

    # Итоговые значения функции стоимости
    print("\n=== Итоговые значения функции стоимости ===")
    print("Линейные признаки:")
    for alpha, cost_history in costs_linear.items():
        print(f"alpha={alpha}, конечная стоимость={cost_history[-1]:.4f}")

    print("\nНелинейные признаки:")
    for alpha, cost_history in costs_nonlinear.items():
        print(f"alpha={alpha}, конечная стоимость={cost_history[-1]:.4f}")

    # Визуализация сходимости
    plot_cost_vs_alpha(costs_linear, alphas, "Сходимость: Линейные признаки")
    plot_cost_vs_alpha(costs_nonlinear, alphas, "Сходимость: Нелинейные признаки")


# Запуск программы
if __name__ == "__main__":
    main()