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

    if df['Age'].isnull().any():
        knn_data = df[['Age', 'Pclass', 'Fare', 'Sex']].copy()
        knn_data['Sex'] = knn_data['Sex'].apply(lambda x: 1 if x == 'male' else 0)
        
        imputer = KNNImputer(n_neighbors=5)
        knn_imputed = imputer.fit_transform(knn_data)
        df['Age'] = knn_imputed[:, 0]  # Update 'Age' with imputed values
    
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
    """Save the cleaned data to a CSV file in the processed data directory."""
    processed_data_dir = 'data/02_processed/'
    processed_data_path = os.path.join(processed_data_dir, filename)  

    # Ensure only the directory exists, avoiding issues with file paths
    os.makedirs(processed_data_dir, exist_ok=True)

    df.to_csv(processed_data_path, index=False, quoting=csv.QUOTE_NONE)
    print(f"Cleaned data saved to {processed_data_path}")


if __name__ == '__main__':
    # Load dataset paths
    train_dataset_path = 'data/01_raw/train.csv'
    test_dataset_path = 'data/01_raw/test.csv'
    
    if os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path):
        # Load the training and test datasets
        train_df = pd.read_csv(train_dataset_path)
        test_df = pd.read_csv(test_dataset_path)
        print("Train and test datasets loaded successfully.")

        # Preprocess train and test dataset
        train_processed = preprocess_dataset(train_df)
        print("Train dataset preprocessed successfully.")
        test_processed = preprocess_dataset(test_df)
        print("Test dataset preprocessed successfully.")

        # Save processed train and test dataset
        save_processed_data(train_processed, 'train_processed.csv')
        save_processed_data(test_processed, 'test_processed.csv')
    
    else:
        print(f"Train or test dataset file not found. Please ensure both '{train_dataset_path}' and '{test_dataset_path}' exist.")
