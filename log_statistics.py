# log_statistics.py - Calculate statistics of the log.
import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.filtering.log.variants import variants_filter


def get_log_statistics(log: pd.DataFrame) -> dict:
    """
    Calculates various statistics from the log DataFrame.

    :param log: DataFrame containing the log data, which should have columns including 'case:concept:name' to identify
     unique cases.
    :return: Dictionary with log statistics.
    """
    case_counts = log['case:concept:name'].value_counts()

    stats = {
        "number_of_events": len(log),
        "number_of_variants": len(variants_filter.get_variants(log)),
        "average_event_count_per_trace": case_counts.mean(),
        "number_of_cases": case_counts.size,
    }
    return stats


def read_log(file_path: str) -> pd.DataFrame:
    """
    Reads the XES log file and converts it to a DataFrame.

    :param file_path: Path to the XES file that contains the log.
    :return: DataFrame containing the log data.
    """
    log = pm4py.read_xes(file_path)
    return log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)


def run_log_statistics() -> None:
    """
    Calculates various log statistics and prints them to the console.
    """
    file_path = input("Enter the path to the XES file that contains the log: ").strip('"')
    try:
        log = read_log(file_path)
        stats = get_log_statistics(log)

        print(f"Number of events in the log: {stats['number_of_events']}")
        print(f"Number of variants in the log: {stats['number_of_variants']}")
        print(f"Average event count per trace: {stats['average_event_count_per_trace']:.2f}")
        print(f"Number of cases in the log: {stats['number_of_cases']}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The log file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    run_log_statistics()
