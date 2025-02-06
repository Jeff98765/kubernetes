import pandas as pd
import yaml

# Local imports
from src.dataprep import preprocess_dataset, save_processed_data, split_data
from src.model import (train_and_evaluate_models, tune_random_forest, 
                       train_and_evaluate_model, save_model, 
                       make_predictions_and_save_to_csv)


def load_config(config_path='conf/catalog.yml'):
    """Load the YAML configuration file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Parsed configuration data.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    """Main function to run the data pipeline and model training process."""
    
    # Load configuration
    config = load_config('conf/catalog.yml')

    # Filepaths from config
    train_data_path = config['train']['filepath']
    test_data_path = config['test']['filepath']
    model_path = config['saved_model']['model_path']

    # Read dataset
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    # Process the training and test datasets
    df_train_processed = preprocess_dataset(df_train)
    df_test_processed = preprocess_dataset(df_test)

    # Save the processed datasets
    save_processed_data(df_train_processed, 'train_processed.csv')
    save_processed_data(df_test_processed, 'test_processed.csv')

    # Split training data
    X_train, X_test, y_train, y_test = split_data(df_train_processed)

    # Train and evaluate models
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    # Display results for each model
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

    # Tune Random Forest Classifier and evaluate
    best_rf_model, best_rf_params = tune_random_forest(X_train, y_train)
    results = train_and_evaluate_model(best_rf_model, X_train, y_train, X_test, y_test)

    # Save the trained Random Forest model
    save_model(best_rf_model, 'random_forest_model.pkl')

    # Make predictions and save them to a CSV file
    make_predictions_and_save_to_csv(df_test_processed, model_path, 'test_predictions.csv')


if __name__ == '__main__':
    main()
