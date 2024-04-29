# log_preparation.py - Prepare logs for training and testing the model.
import os

from dateutil import parser
import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter

from config import extract_log_name


def prepare_logs(file_path: str, elusive_percentage: float = 0.1) -> None:
    """
    Prepare logs for training and testing the model.
    
    :param file_path: Path to the XES file that contains the log.
    :param elusive_percentage: Percentage of rows to set empty in the elusive log. Default is 0.2.
    """
    try:
        if not os.path.isfile(file_path):
            raise FileNotFoundError("The provided path does not point to a valid file.")

        file_name = extract_log_name(file_path)
        result_folder = 'logs/created'
        result_folder_elusive = 'logs/created/elusive'
        result_csv_file = f'{file_name}.csv'
        result_xes_file = f'{file_name}.xes'

        file = pm4py.read_xes(file_path)
        log = log_converter.apply(file, variant=log_converter.Variants.TO_DATA_FRAME)
        log['time:timestamp'] = log['time:timestamp'].apply(lambda x: parser.isoparse(x) if isinstance(x, str) else x)
        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], utc=True)
        log = log.sort_values(['time:timestamp']).reset_index(drop=True)
        log['time:timestamp'] = log['time:timestamp'].dt.tz_localize(None)

        log['Sorted Index'] = log.index

        elusive_log = log.copy()
        num_rows_to_set_empty = int(len(elusive_log) * elusive_percentage)
        random_indices = np.random.choice(elusive_log.index, num_rows_to_set_empty, replace=False)
        elusive_log.loc[random_indices, 'case:concept:name'] = ""

        os.makedirs(result_folder, exist_ok=True)
        log.to_csv(os.path.join(result_folder, result_csv_file), index=False)
        pm4py.write_xes(log, os.path.join(result_folder, result_xes_file))

        os.makedirs(result_folder_elusive, exist_ok=True)
        elusive_log.to_csv(os.path.join(result_folder_elusive, result_csv_file), index=False)
        pm4py.write_xes(elusive_log, os.path.join(result_folder_elusive, result_xes_file))

        print("Logs prepared successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    path = input("Enter the path to the XES file that contains the log: ").strip('"')
    if path:
        prepare_logs(path)
