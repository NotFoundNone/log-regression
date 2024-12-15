import pandas as pd
import numpy as np

def load_data_from_file(file_path):
    """
    Загружает данные из текстового файла в DataFrame.
    """
    df = pd.read_csv(file_path, header=None, names=['Vibration', 'Irregularity', 'Fault'])
    return df

def normalization_func(df, exclude_columns=None):
    """
    Нормализует данные, кроме указанных столбцов.
    Возвращает нормализованный DataFrame и словарь с параметрами нормализации.
    """
    if exclude_columns is None:
        exclude_columns = []

    # Словарь для хранения средних значений и стандартных отклонений
    mean_std_values = {}

    for column in df.columns:
        if column not in exclude_columns:
            mean_val = df[column].mean()  # Среднее значение
            std_val = df[column].std()    # Стандартное отклонение

            # Сохраняем среднее и стандартное отклонение
            mean_std_values[column] = {'mean': mean_val, 'std': std_val}

            # Стандартизация столбца
            df[column] = (df[column] - mean_val) / std_val

    return df, mean_std_values

def add_bias_column(df):
    """
    Добавляет столбец bias (с единицами) к DataFrame.
    """
    df.insert(0, 'Bias', 1)
    return df

def add_nonlinear_features(df, columns):
    """
    Добавляет нелинейные признаки (квадратичные и произведения) в DataFrame.
    """
    for col in columns:
        df[f"{col}^2"] = df[col] ** 2  # Квадрат признака

    # Добавляем произведение признаков
    if len(columns) > 1:
        df["Interaction"] = df[columns[0]] * df[columns[1]]  # Произведение двух признаков

    return df

