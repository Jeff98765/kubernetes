import os
import csv
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_dataset(df: pd.DataFrame):
    """Preprocess the Titanic dataset.

    Args:
        df (pd.DataFrame): Raw Titanic dataset.

    Returns:
        pd.DataFrame: Preprocessed Titanic dataset with imputed, encoded,
                      and scaled features.
    """
    # Step 1: Handle Missing Values
    if df['Embarked'].isnull().any():
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Impute 'Age' using KNN with 4 features
    if df['Age'].isnull().any():
        knn_data = df[['Age', 'Pclass', 'Fare', 'Sex']].copy()
        knn_data['Sex'] = knn_data['Sex'].apply(lambda x: 1 if x == 'male' else 0)
        
        imputer = KNNImputer(n_neighbors=5)
        knn_imputed = imputer.fit_transform(knn_data)
        df['Age'] = knn_imputed[:, 0]  # Update 'Age' with imputed values
    
    # Impute missing 'Fare' values with the mode (most frequent value)
    if df['Fare'].isnull().any():
        fare_mode = df['Fare'].mode()[0]
        df['Fare'] = df['Fare'].fillna(fare_mode)

    # Step 2: Drop Unnecessary Columns
    columns_to_drop = ['Ticket', 'Cabin', 'PassengerId', 'Name']
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Step 3: Feature Engineering
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    
    # Step 4: Encode Categorical Features (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype=int)
    
    # Step 5: Scale Numerical Columns
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Family_Size']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df


def save_processed_data(df: pd.DataFrame, filename: str):
    """Save the cleaned data to a CSV file in the processed data directory.

    Args:
        df (pd.DataFrame): Preprocessed Titanic dataset.
        filename (str): Name of the CSV file to save the data.

    Returns:
        None
    """
    # Define the directory path where processed data should be saved
    processed_data_path = os.path.join('data', '02_processed', filename)
    
    # Ensure the processed data directory exists
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    # Save the cleaned data to the specified path
    df.to_csv(processed_data_path, index=False, quoting=csv.QUOTE_NONE)
    print(f"Cleaned data saved to {processed_data_path}")


def split_data(df: pd.DataFrame):
    """Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): Preprocessed Titanic dataset.

    Returns:
        X_train (pd.DataFrame), X_test (pd.DataFrame), y_train (pd.Series),
        y_test (pd.Series): Train-test split data.
    """
    # Select Features and Target
    X = df.drop('Survived', axis=1)  # Features
    y = df['Survived']  # Target

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, test_size=0.15, random_state=42
    )
    
    return X_train, X_test, y_train, y_test
