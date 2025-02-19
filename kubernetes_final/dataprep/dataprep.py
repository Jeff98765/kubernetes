from flask import Flask
import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)

def extract_title(name):
    """Extract title from passenger's name and normalize it."""
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if title_search:
        title = title_search.group(1)
        title_mapping = {
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
            'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
            'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
        }
        return title_mapping.get(title, title)  # Replace if found in mapping
    return ""


def feature_engineering(df):
    """Add new features and clean up titles."""
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 for the passenger themselves
    df['Title'] = df['Name'].apply(extract_title)
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    df['Age'] = df['Age'].fillna(df['Age'].median())  # Filling missing Age with median
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Filling missing Embarked with mode
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # Handling Fare missing values
    return df


def handle_outliers(df):
    """Cap outliers for Age and Fare to reduce skewness."""
    df['Age'] = np.clip(df['Age'], a_min=df['Age'].quantile(0.01), a_max=df['Age'].quantile(0.99))
    df['Fare'] = np.clip(df['Fare'], a_min=df['Fare'].quantile(0.01), a_max=df['Fare'].quantile(0.99))
    return df


def drop_unnecessary_columns(df, columns_to_drop):
    """Drop unnecessary columns."""
    return df.drop(columns=columns_to_drop)


def encode_features(df):
    """Encode categorical features using OneHotEncoder."""
    categorical_features = ['Sex', 'Embarked', 'Title']
    encoder = OneHotEncoder(drop='first')
    encoded_data = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(categorical_features))
    
    df = df.drop(columns=categorical_features)
    df = pd.concat([df, encoded_df], axis=1)
    
    return df


def scale_features(df):
    """Scale numerical features using StandardScaler."""
    numerical_features = ['Age', 'Fare', 'FamilySize']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_features])
    scaled_df = pd.DataFrame(scaled_data, columns=numerical_features)
    
    df = df.drop(columns=numerical_features)
    df = pd.concat([df, scaled_df], axis=1)
    
    return df

def preprocess_data(input_file, output_file):
    """Main function to preprocess data."""
    # Load Titanic dataset
    df = pd.read_csv(input_file)

    # Apply transformations
    df = feature_engineering(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)

    # Drop unnecessary columns
    columns_to_drop = ['Cabin', 'PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket']
    df = drop_unnecessary_columns(df, columns_to_drop)
    
    # Encoding and Scaling (separately)
    df = encode_features(df)
    df = scale_features(df)

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Warning: There are still missing values in the dataset!")

    # Save processed data
    df.to_csv(output_file, index=False)

@app.route('/preprocess')
def preprocess_train_test():
    # Define file paths
    train_dataset_path = 'data/01_raw/train.csv'
    predict_dataset_path = 'data/01_raw/predict.csv'
    train_processed_path = 'data/02_processed/train_processed.csv'
    predict_processed_path = 'data/02_processed/predict_processed.csv'

    # Preprocess the data
    try:
        preprocess_data(train_dataset_path, train_processed_path)
        preprocess_data(predict_dataset_path, predict_processed_path)
    except Exception as e:
        return f"Error during data preprocessing: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)