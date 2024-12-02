# random_repair.py - Repair case IDs using random predictors.
import os

import numpy as np
import pandas as pd


class FullyRandomPredictor:
    """
    Predicts case IDs observed in df_cleaned completely at random.
    """

    def __init__(self, df_cleaned: pd.DataFrame) -> None:
        """
        Initializes the Fully Random Predictor.

        :param df_cleaned: DataFrame with cleaned data.
        """
        print("Initializing Fully Random Predictor...")
        self.df_cleaned = df_cleaned
        self.case_ids = df_cleaned['case_id'].unique()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts case IDs for the given dataframe using a fully random approach.

        :param df: DataFrame with missing case IDs.
        :return: DataFrame with predicted case IDs.
        """
        print("Predicting case IDs using Fully Random Predictor...")
        df['predicted_case_id'] = np.random.choice(self.case_ids, size=len(df))
        print("Fully Random Predictor prediction complete.")
        return df


class RandomPredictorWithDistribution:
    """
    Predicts case IDs given the frequencies of case IDs in df_cleaned.
    """

    def __init__(self, df_cleaned: pd.DataFrame) -> None:
        """
        Initializes the Random Predictor with Distribution.

        :param df_cleaned: DataFrame with cleaned data.
        """
        print("Initializing Random Predictor with Distribution...")
        self.df_cleaned = df_cleaned
        self.case_id_frequencies = df_cleaned['case_id'].value_counts(normalize=True)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts case IDs for the given dataframe based on the distribution of case IDs.

        :param df: DataFrame with missing case IDs.
        :return: DataFrame with predicted case IDs.
        """
        print("Predicting case IDs using Random Predictor with Distribution...")
        case_ids = self.case_id_frequencies.index
        case_id_probs = self.case_id_frequencies.values
        df['predicted_case_id'] = np.random.choice(case_ids, size=len(df), p=case_id_probs)
        print("Random Predictor with Distribution prediction complete.")
        return df


def prepare_data(data_path: str, log_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and preprocesses the data.

    :param data_path: Path to the data directory.
    :param log_name: Name of the log file.
    :return: Original and cleaned dataframes.
    """
    print(f"Loading data from {data_path}{log_name}...")
    df = pd.read_csv(os.path.join(data_path, log_name))
    resource_col = [col for col in df.columns if 'resource' in col.lower()][0]
    df = df[['Sorted Index', 'case:concept:name', 'concept:name', 'time:timestamp', resource_col]]
    df.columns = ['event_id', 'case_id', 'activity', 'timestamp', 'resource']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['case_id', 'timestamp'])

    # Remove rows with missing case IDs
    df_cleaned = df.dropna(subset=['case_id']).reset_index(drop=True).copy()
    print("Data loading and preprocessing complete.")
    return df, df_cleaned


def repair_case_ids(df: pd.DataFrame, predictor: FullyRandomPredictor) -> pd.DataFrame:
    """
    Repairs missing case IDs in the dataframe using the specified predictor.

    :param df: DataFrame with missing case IDs.
    :param predictor: Predictor to use for repairing case IDs.
    :return: DataFrame with repaired case IDs.
    """
    print("Starting case ID repair...")
    df_repaired = predictor.predict(df)
    print("Case ID repair complete.")
    return df_repaired


if __name__ == "__main__":
    # Example: Load data and prepare it
    data_path = '../Data/Elusive CSV Logs/renting/'
    log_name = 'renting_log_low_10.csv'
    df, df_cleaned = prepare_data(data_path, log_name)

    # Select erroneous rows (those where case_id is NaN)
    df_erroneous = df[df['case_id'].isna()]

    # Initialize predictors
    # predictor = RandomPredictorWithDistribution(df_cleaned)
    predictor = FullyRandomPredictor(df_cleaned)

    # Use random predictor with distribution to repair erroneous case IDs
    df_repaired_random_dist = repair_case_ids(df_erroneous, predictor)

    # Combine the repaired predictions with the original dataframe
    df_combined_random_dist = pd.concat([df[~df['case_id'].isna()], df_repaired_random_dist])
    # Replace NaN values in 'case_id' with corresponding 'predicted_case_id' values
    df_combined_random_dist['predicted_case_id'].fillna(df_combined_random_dist['case_id'], inplace=True)

    df_repaired = df_combined_random_dist.copy()

    # Load the ground truth data
    correct_data_path = os.path.join('../Data/Correct Logs', data_path.split('/')[-2] + '.csv')
    df_correct = pd.read_csv(correct_data_path)
    resource_col_corr = [col for col in df_correct.columns if 'resource' in col.lower()][0]
    df_correct = df_correct[['Sorted Index', 'case:concept:name', 'concept:name', 'time:timestamp', resource_col_corr]]
    df_correct.columns = ['event_id', 'case_id', 'activity', 'timestamp', 'resource']

    # Add the ground truth case ID to the repaired dataframe
    df_repaired = df_repaired.merge(df_correct[['event_id', 'case_id']], on='event_id', how='left',
                                    suffixes=('', '_ground_truth'))
    df_repaired.rename(columns={'case_id_ground_truth': 'Ground Truth Case ID'}, inplace=True)

    df_repaired['Determined Case ID'] = df_repaired['predicted_case_id'].astype(
        df_repaired['Ground Truth Case ID'].dtypes)
    df_repaired['Determination Probability'] = ''
    df_repaired['Determination Follow-up Probability'] = ''
    df_repaired['Original Case ID'] = df_repaired['case_id']
    df_repaired['Activity'] = df_repaired['activity']
    df_repaired['Timestamp'] = df_repaired['timestamp']
    df_repaired['Resource'] = df_repaired['resource']
    df_repaired['Sorted Index'] = df_repaired['event_id']

    df_out = df_repaired[
        ['Determined Case ID', 'Determination Probability', 'Determination Follow-up Probability', 'Original Case ID',
         'Activity', 'Timestamp', 'Resource', 'Sorted Index', 'Ground Truth Case ID']]
