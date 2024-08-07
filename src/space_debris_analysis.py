import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


def load_data(file_path):
    df = pd.read_csv('/home/eric_baldwin/ddiMain/capstone/ddi_capstone_2/data/space_decay.csv')
    print("Data loaded successfully.")
    return df


def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(labels=['DECAY_DATE'], axis=1)
    print("Dropped unnecessary columns.")

    # Rename object type for better presentation
    df['OBJECT_TYPE'] = df['OBJECT_TYPE'].replace(
        {'DEBRIS': 'Debris', 'PAYLOAD': 'Payload', 'TBA': 'Unknown', 'ROCKET BODY': 'Rocket'})
    print("Renamed object types.")

    # Fill missing values for categorical columns with 'Unknown'
    df['COUNTRY_CODE'] = df['COUNTRY_CODE'].replace(
        to_replace={'TBD': 'Unknown', np.nan: 'Unknown'})
    df['RCS_SIZE'] = df['RCS_SIZE'].replace(to_replace={np.nan: 'Unknown'})
    print("Filled missing values for categorical columns.")

    # Create PERIOD_HOURS and ALTITUDE_MI columns
    df['PERIOD_HOURS'] = df['PERIOD'] / 60
    df['ALTITUDE_MI'] = (df['SEMIMAJOR_AXIS'] - 6371) * 0.6213
    print("Created PERIOD_HOURS and ALTITUDE_MI columns.")

    # Convert OBJECT_TYPE to binary classification
    df['OBJECT_TYPE'] = df['OBJECT_TYPE'].apply(
        lambda x: 1 if x == 'Payload' else 0)
    print("Converted OBJECT_TYPE to binary classification.")

    # Encode categorical variables
    le_country_code = LabelEncoder()
    df['COUNTRY_CODE'] = le_country_code.fit_transform(df['COUNTRY_CODE'])
    le_rcs_size = LabelEncoder()
    df['RCS_SIZE'] = le_rcs_size.fit_transform(df['RCS_SIZE'])
    print("Encoded categorical variables.")

    # Fill missing values for numerical columns
    imputer = SimpleImputer(strategy='mean')
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    print("Filled missing values for numerical columns.")

    # Feature selection
    features = ['INCLINATION', 'PERIOD_HOURS', 'ALTITUDE_MI', 'ECCENTRICITY', 'RA_OF_ASC_NODE',
                'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'APOAPSIS', 'PERIAPSIS']
    X = df[features]
    y = df['OBJECT_TYPE']
    print("Selected features for modeling.")

    return X, y


def add_polynomial_features(X, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    print("Added polynomial features.")
    return X_poly


def standardize_features(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    print("Standardized features.")
    return X_standardized


def handle_class_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Handled class imbalance using SMOTE.")
    return X_resampled, y_resampled


def split_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    print("Split data into training and test sets.")
    return X_train, X_test, y_train, y_test


def define_model(max_iter=1000, class_weight='balanced'):
    logreg = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
    print("Defined the Logistic Regression model.")
    return logreg


def tune_hyperparameters(model, X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    print("Completed grid search for hyperparameter tuning.")
    return grid_search


def evaluate_model(best_model, X_test, y_test):
    y_pred_logreg = best_model.predict(X_test)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_logreg))


def main():
    file_path = '/home/eric_baldwin/ddiMain/capstone/ddi_capstone_2/data/space_decay.csv'
    df = load_data(file_path)

    X, y = preprocess_data(df)
    X_poly = add_polynomial_features(X)
    X_standardized = standardize_features(X_poly)
    X_resampled, y_resampled = handle_class_imbalance(X_standardized, y)

    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)

    model = define_model(max_iter=1000, class_weight='balanced')
    grid_search = tune_hyperparameters(model, X_train, y_train)

    best_logreg = grid_search.best_estimator_
    print("Selected the best Logistic Regression model from grid search.")

    evaluate_model(best_logreg, X_test, y_test)
    print("Best Hyperparameters:", grid_search.best_params_)


if __name__ == "__main__":
    main()
