from processing_data import load_data_from_file, normalization_func, add_bias_column, add_nonlinear_features
from logistic_regression import sigmoid, compute_cost, gradient_descent, predict
import numpy as np
import matplotlib.pyplot as plt

# Путь к файлу с данными
file_path = "ex2data1.txt"

# Загрузка данных
data = load_data_from_file(file_path)

# Нормализация данных
# Исключаем столбец 'Fault' из нормализации
normalized_data, mean_std = normalization_func(data, exclude_columns=['Fault'])

# Разделение на признаки и метки
X_original = normalized_data[['Vibration', 'Irregularity']]  # Только исходные признаки
y = normalized_data['Fault']                                # Метки

# Добавление нелинейных признаков
data_with_nonlinear = add_nonlinear_features(normalized_data, ['Vibration', 'Irregularity'])
X_nonlinear = data_with_nonlinear.drop(columns=['Fault'])  # Все признаки с нелинейными дополнениями

# Добавление столбца единиц
X_original = add_bias_column(X_original)
X_nonlinear = add_bias_column(X_nonlinear)

# Преобразование в NumPy массивы
X_original = X_original.to_numpy()
X_nonlinear = X_nonlinear.to_numpy()
y = y.to_numpy()

# Инициализация параметров
theta_original = np.zeros(X_original.shape[1])
theta_nonlinear = np.zeros(X_nonlinear.shape[1])

# Гиперпараметры
alpha = 0.01
num_iter = 10000

# Обучение моделей
theta_original = gradient_descent(X_original, y, theta_original, alpha, num_iter)
theta_nonlinear = gradient_descent(X_nonlinear, y, theta_nonlinear, alpha, num_iter)

# Оценка моделей
y_pred_original = predict(X_original, theta_original)
y_pred_nonlinear = predict(X_nonlinear, theta_nonlinear)

accuracy_original = np.mean(y_pred_original == y) * 100
accuracy_nonlinear = np.mean(y_pred_nonlinear == y) * 100

# Вывод результатов
print("=== Результаты без нелинейных признаков ===")
print("Параметры модели (theta):", theta_original)
print(f"Точность модели: {accuracy_original:.2f}%\n")

print("=== Результаты с нелинейными признаками ===")
print("Параметры модели (theta):", theta_nonlinear)
print(f"Точность модели: {accuracy_nonlinear:.2f}%\n")

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

# Построение сравнения
plot_comparison(X_original, X_nonlinear, y, theta_original, theta_nonlinear)