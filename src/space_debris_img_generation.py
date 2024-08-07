# src/space_debris_img_generation.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE


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


def generate_confusion_matrix(y_true, y_pred, file_path='img/confusion_matrix.png'):
    """
    Generate and save the confusion matrix.

    Parameters:
    y_true (pd.Series): True labels.
    y_pred (np.ndarray): Predicted labels.
    file_path (str): Path to save the image file.

    Returns:
    None
    """
    ensure_directory_exists(os.path.dirname(file_path))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(file_path)
    plt.close()
    print(f"Confusion matrix saved as '{file_path}'.")


def generate_roc_curve(model, X_test, y_test, file_path='img/roc_curve.png'):
    """
    Generate and save the ROC curve.

    Parameters:
    model (LogisticRegression): The trained model.
    X_test (np.ndarray): Test feature matrix.
    y_test (pd.Series): True labels for the test set.
    file_path (str): Path to save the image file.

    Returns:
    None
    """
    ensure_directory_exists(os.path.dirname(file_path))
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(file_path)
    plt.close()
    print(f"ROC curve saved as '{file_path}'.")


def generate_precision_recall_curve(model, X_test, y_test, file_path='img/precision_recall_curve.png'):
    """
    Generate and save the Precision-Recall curve.

    Parameters:
    model (LogisticRegression): The trained model.
    X_test (np.ndarray): Test feature matrix.
    y_test (pd.Series): True labels for the test set.
    file_path (str): Path to save the image file.

    Returns:
    None
    """
    ensure_directory_exists(os.path.dirname(file_path))
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(file_path)
    plt.close()
    print(f"Precision-Recall curve saved as '{file_path}'.")


def main():
    # Load your data
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'space_decay.csv'))
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Preprocess your data
    df = df.drop(labels=['DECAY_DATE'], axis=1)
    df['OBJECT_TYPE'] = df['OBJECT_TYPE'].replace({'DEBRIS': 'Debris', 'PAYLOAD': 'Payload', 'TBA': 'Unknown', 'ROCKET BODY': 'Rocket'})
    df['COUNTRY_CODE'] = df['COUNTRY_CODE'].replace(to_replace={'TBD': 'Unknown', np.nan: 'Unknown'})
    df['RCS_SIZE'] = df['RCS_SIZE'].replace(to_replace={np.nan: 'Unknown'})
    df['PERIOD_HOURS'] = df['PERIOD'] / 60
    df['ALTITUDE_MI'] = (df['SEMIMAJOR_AXIS'] - 6371) * 0.6213
    df['OBJECT_TYPE'] = df['OBJECT_TYPE'].apply(lambda x: 1 if x == 'Payload' else 0)
    le_country_code = LabelEncoder()
    df['COUNTRY_CODE'] = le_country_code.fit_transform(df['COUNTRY_CODE'])
    le_rcs_size = LabelEncoder()
    df['RCS_SIZE'] = le_rcs_size.fit_transform(df['RCS_SIZE'])
    imputer = SimpleImputer(strategy='mean')
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    features = ['INCLINATION', 'PERIOD_HOURS', 'ALTITUDE_MI', 'ECCENTRICITY', 'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'APOAPSIS', 'PERIAPSIS']
    X = df[features]
    y = df['OBJECT_TYPE']
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    scaler = StandardScaler()
    X_poly = scaler.fit_transform(X_poly)
    smote = SMOTE(random_state=42)
    X_poly_resampled, y_resampled = smote.fit_resample(X_poly, y)
    X_train, X_test, y_train, y_test = train_test_split(X_poly_resampled, y_resampled, test_size=0.3, random_state=42)
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    best_logreg = grid_search.best_estimator_
    y_pred_logreg = best_logreg.predict(X_test)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_logreg))

    # Generate and save the images
    generate_confusion_matrix(y_test, y_pred_logreg)
    generate_roc_curve(best_logreg, X_test, y_test)
    generate_precision_recall_curve(best_logreg, X_test, y_test)


if __name__ == "__main__":
    main()
