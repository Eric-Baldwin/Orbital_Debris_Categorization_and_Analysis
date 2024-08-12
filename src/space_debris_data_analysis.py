import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def ensure_directory_exists(directory):
    """
    Ensure the specified directory exists; if not, create it.

    Parameters:
    directory (str): Path to the directory.

    Returns:
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

def generate_histogram(df, column, file_path='img/histogram.png'):
    """
    Generate and save a histogram of the specified column.

    Parameters:
    df (pd.DataFrame): The dataframe.
    column (str): The column to plot.
    file_path (str): Path to save the image file.

    Returns:
    None
    """
    ensure_directory_exists(os.path.dirname(file_path))
    plt.figure(figsize=(10, 7))
    sns.histplot(df[column], bins=30, kde=True)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')
    plt.savefig(file_path)
    plt.close()
    print(f"Histogram saved as '{file_path}'.")

def generate_bar_plot(df, column, file_path='img/bar_plot.png'):
    """
    Generate and save a bar plot of the specified column.

    Parameters:
    df (pd.DataFrame): The dataframe.
    column (str): The column to plot.
    file_path (str): Path to save the image file.

    Returns:
    None
    """
    ensure_directory_exists(os.path.dirname(file_path))
    plt.figure(figsize=(10, 7))
    sns.countplot(x=column, data=df)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Bar Plot of {column}')
    plt.savefig(file_path)
    plt.close()
    print(f"Bar plot saved as '{file_path}'.")

def generate_heatmap(df, file_path='img/heatmap.png'):
    """
    Generate and save a heatmap of the correlation matrix.

    Parameters:
    df (pd.DataFrame): The dataframe.
    file_path (str): Path to save the image file.

    Returns:
    None
    """
    ensure_directory_exists(os.path.dirname(file_path))
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap of Correlation Matrix')
    plt.savefig(file_path)
    plt.close()
    print(f"Heatmap saved as '{file_path}'.")

def generate_scatter_plot(df, x_column, y_column, hue_column, file_path='img/scatter_plot.png'):
    """
    Generate and save a scatter plot of the specified columns.

    Parameters:
    df (pd.DataFrame): The dataframe.
    x_column (str): The column for the x-axis.
    y_column (str): The column for the y-axis.
    hue_column (str): The column for color encoding.
    file_path (str): Path to save the image file.

    Returns:
    None
    """
    ensure_directory_exists(os.path.dirname(file_path))
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=x_column, y=y_column, hue=hue_column, data=df)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Scatter Plot of {x_column} vs. {y_column}')
    plt.savefig(file_path)
    plt.close()
    print(f"Scatter plot saved as '{file_path}'.")

def main():
    # Load data
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'space_decay.csv'))
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Preprocess data
    df = df.drop(labels=['DECAY_DATE'], axis=1)
    df['OBJECT_TYPE'] = df['OBJECT_TYPE'].replace({'DEBRIS': 'Debris', 'PAYLOAD': 'Payload', 'TBA': 'Unknown', 'ROCKET BODY': 'Rocket'})
    df['COUNTRY_CODE'] = df['COUNTRY_CODE'].replace(to_replace={'TBD': 'Unknown', np.nan: 'Unknown'})
    df['RCS_SIZE'] = df['RCS_SIZE'].replace(to_replace={np.nan: 'Unknown'})
    df['PERIOD_HOURS'] = df['PERIOD'] / 60
    df['ALTITUDE_MI'] = (df['SEMIMAJOR_AXIS'] - 6371) * 0.6213
    le_country_code = LabelEncoder()
    df['COUNTRY_CODE'] = le_country_code.fit_transform(df['COUNTRY_CODE'])
    le_rcs_size = LabelEncoder()
    df['RCS_SIZE'] = le_rcs_size.fit_transform(df['RCS_SIZE'])
    imputer = SimpleImputer(strategy='mean')
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # Generate and save the images
    generate_histogram(df, 'ALTITUDE_MI', 'img/altitude_histogram.png')
    generate_bar_plot(df, 'OBJECT_TYPE', 'img/object_type_bar_plot.png')
    generate_heatmap(df, 'img/correlation_heatmap.png')
    generate_scatter_plot(df, 'INCLINATION', 'PERIOD_HOURS', 'OBJECT_TYPE', 'img/inclination_vs_period_scatter.png')


if __name__ == "__main__":
    main()
