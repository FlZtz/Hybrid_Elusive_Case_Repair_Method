# quality_metrics.py - Evaluate the repaired logs using various metrics.
import os
import pickle

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from tqdm.auto import tqdm

memorized_distances: dict = {}  # Memorization dictionary to store distances between traces


def bigram_similarity(df: pd.DataFrame, ins_col: str = 'y_pred', true_col: str = 'y_true',
                      timestamp_col: str = 'timestamp', event_id_col: str = 'event_id') -> float:
    """
    Computes the bigram similarity (L2L2gram) between the predicted and true cases.

    :param df: A pandas DataFrame containing event log data.
    :param ins_col: The column name representing the predicted case ID. Default is 'y_pred'.
    :param true_col: The column name representing the true case ID. Default is 'y_true'.
    :param timestamp_col: The column name representing event timestamps. Default is 'timestamp'.
    :param event_id_col: The column name representing the event ID. Default is 'event_id'.
    :return: The bigram similarity value.
    """
    # Extract true and predicted cases
    true_cases = (
        df.sort_values(by=[true_col, timestamp_col])
        .groupby(true_col)[event_id_col]
        .apply(list)
    )
    pred_cases = (
        df.sort_values(by=[ins_col, timestamp_col])
        .groupby(ins_col)[event_id_col]
        .apply(list)
    )

    # Flatten predicted cases into a set of all bigrams in L'
    pred_bigrams = set(
        (pred[i], pred[i + 1])
        for pred in pred_cases
        for i in range(len(pred) - 1)
    )

    # Compute bigram similarity
    def case_bigram_similarity(case: list) -> float:
        """
        Compute the bigram similarity for a single case.

        :param case: A list of event IDs in the case.
        :return: The bigram similarity for the case.
        """
        bigrams = [(case[i], case[i + 1]) for i in range(len(case) - 1)]

        if not bigrams:
            return 0  # Avoid division by zero for single-event cases

        matching_bigrams = sum(1 for bigram in bigrams if bigram in pred_bigrams)

        return matching_bigrams / len(bigrams)

    total_similarity = sum(case_bigram_similarity(case) for case in true_cases)
    num_cases = len(true_cases)

    # Normalize over all cases
    if num_cases == 0:
        return 0.0

    return total_similarity / num_cases


def calculate_metrics_for_log(data_path: str, log_name: str, configuration_path: str) -> dict:
    """
    Calculate metrics for a given log.

    :param data_path: Path to the folder containing the log.
    :param log_name: Name of the log file.
    :param configuration_path: Path to the configuration file.
    :return: A dictionary containing the calculated metrics
    """
    # Load and preprocess data
    df, correct_predictions, wrong_predictions, config = prepare_data(data_path, log_name, configuration_path)
    df_completeness = df.copy()
    df['y_pred'] = df['y_pred'].fillna(-1)
    df = df[df['y_pred'] != -1].reset_index(drop=True)

    # Extract traces
    true_traces = df.sort_values(by=['y_true', 'timestamp']).groupby('y_true')['activity'].apply(tuple).to_list()
    repaired_traces = df.sort_values(by=['y_pred', 'timestamp']).groupby('y_pred')['activity'].apply(tuple).to_list()
    distinct_true_traces = list(set(true_traces))
    distinct_repaired_traces = list(set(repaired_traces))

    consistency_metrics = consistency(df, config)

    # Calculate metrics
    metrics = {
        "Completeness": completeness(df_completeness),
        **consistency_metrics,
        "Trace-to-Trace Similarity": trace2trace_similarity_parallel(distinct_true_traces, distinct_repaired_traces),
        "Trace-to-Trace Frequency Similarity": trace_to_trace_frequency_similarity_parallel(true_traces,
                                                                                            repaired_traces,
                                                                                            optimized_ins_del_distance),
        "Partial Case Similarity": partial_case_similarity_optimized(df),
        "Bigram Similarity": bigram_similarity(df),
        "Trigram Similarity": trigram_similarity(df),
        "Case Similarity": case_similarity(df),
        "Event-Time Deviation (SMAPEET)": event_time_deviation(df),
        "Case Cycle Time Deviation (SMAPECT)": case_cycle_time_deviation(df)
    }

    return metrics


def case_cycle_time_deviation(df: pd.DataFrame, ins_col: str = 'y_pred', true_col: str = 'y_true',
                              timestamp_col: str = 'timestamp', activity_col: str = 'activity'):
    """
    Computes the case cycle time deviation (SMAPECT) between the predicted and true logs.

    :param df: A pandas DataFrame containing event log data.
    :param ins_col: The column name representing the predicted case ID. Default is 'y_pred'.
    :param true_col: The column name representing the true case ID. Default is 'y_true'.
    :param timestamp_col: The column name representing timestamps of events. Default is 'timestamp'.
    :param activity_col: The column name representing the event activity. Default is 'activity'.
    :return: The case cycle time deviation (SMAPECT).
    """

    # Compute the first and last timestamps for each case in true and predicted logs
    def compute_cycle_time(df: pd.DataFrame, case_col: str) -> pd.DataFrame:
        """
        Helper function to calculate cycle times for each case.

        :param df: A pandas DataFrame containing event log data.
        :param case_col: The column name representing the case ID.
        :return: A DataFrame containing the cycle times for each case.
        """
        case_stats = (
            df.groupby(case_col)[timestamp_col]
            .agg(['min', 'max'])  # Min is start time, max is end time
            .reset_index()
        )
        case_stats['cycle_time'] = (case_stats['max'] - case_stats['min']).dt.total_seconds()
        return case_stats

    true_cycle_times = compute_cycle_time(df, true_col)
    pred_cycle_times = compute_cycle_time(df, ins_col)

    # Merge true and predicted cases on matching start events
    merged = (
        df.groupby(true_col).first().reset_index()[[true_col, activity_col]]  # True start events
        .merge(
            df.groupby(ins_col).first().reset_index()[[ins_col, activity_col]],  # Predicted start events
            on=activity_col,  # Match cases by start event
            suffixes=('_true', '_pred'),
        )
        .merge(true_cycle_times, on=true_col)
        .merge(pred_cycle_times, on=ins_col, suffixes=('_true', '_pred'))
    )

    # Calculate SMAPE for each matched case
    smape_numerator = (merged['cycle_time_true'] - merged['cycle_time_pred']).abs()
    smape_denominator = (merged['cycle_time_true'].abs() + merged['cycle_time_pred'].abs()).replace(0, 1e-10)
    smape_values = smape_numerator / smape_denominator

    # Compute average SMAPE across matched cases
    smapect = smape_values.mean()

    return smapect


def case_similarity(df: pd.DataFrame, ins_col: str = 'y_pred', true_col: str = 'y_true',
                    timestamp_col: str = 'timestamp', event_id_col: str = 'event_id') -> float:
    """
    Computes the case similarity (L2Lcase) between the predicted and true cases.

    :param df: A pandas DataFrame containing event log data.
    :param ins_col: The column name representing the predicted case ID. Default is 'y_pred'.
    :param true_col: The column name representing the true case ID. Default is 'y_true'.
    :param timestamp_col: The column name representing event timestamps. Default is 'timestamp'.
    :param event_id_col: The column name representing the event ID. Default is 'event_id'.
    :return: The case similarity value.
    """
    # Extract true and predicted cases as sorted tuples of activities
    true_cases = set(
        tuple(group[event_id_col])
        for _, group in df.sort_values(by=[true_col, timestamp_col]).groupby(true_col)
    )
    pred_cases = set(
        tuple(group[event_id_col])
        for _, group in df.sort_values(by=[ins_col, timestamp_col]).groupby(ins_col)
    )

    # Compute intersection and normalization
    intersection_count = len(true_cases & pred_cases)
    total_cases = len(true_cases)  # Assuming both logs have the same number of cases

    return intersection_count / total_cases if total_cases > 0 else 0.0


def completeness(df: pd.DataFrame, ins_col: str = 'y_pred', ori_col: str = 'is_erroneous') -> float:
    """
    Calculate the completeness improvement fraction.

    :param df: DataFrame containing the event log data.
    :param ins_col: Column name representing the predicted case ID. Default is 'y_pred'.
    :param ori_col: Column name representing whether the original case ID is erroneous. Default is 'is_erroneous'.
    :return: Fractional improvement in completeness.
    """
    # Calculate original completeness
    original_completeness = (~df[ori_col]).mean() * 100

    # Calculate predicted completeness
    predicted_completeness = (1 - df[ins_col].isna().mean()) * 100

    # Calculate fractional improvement
    max_possible_increase = 100 - original_completeness
    improvement_fraction = (predicted_completeness - original_completeness) / max_possible_increase

    return improvement_fraction


def compute_distance_matrix(log1: list, log2: list, distance_function: callable) -> np.ndarray:
    """
    Compute the distance matrix between traces in two logs using the given distance function.

    :param log1: List of traces from the first log.
    :param log2: List of traces from the second log.
    :param distance_function: Function to compute the distance between two traces.
    :return: The distance matrix.
    """
    distance_matrix = np.zeros((len(log1), len(log2)))

    for i, trace1 in tqdm(enumerate(log1), total=len(log1)):
        for j, trace2 in enumerate(log2):
            distance_matrix[i, j] = distance_function(trace1, trace2)

    return distance_matrix


# Optimized version of compute_distance_matrix
def compute_distance_matrix_parallel(log1: list, log2: list, distance_function: callable,
                                     n_jobs: int = -1) -> np.ndarray:
    """
    Compute the distance matrix between traces in two logs using the given distance function with parallel processing.

    :param log1: List of traces from the first log.
    :param log2: List of traces from the second log.
    :param distance_function: Function to compute the distance between two traces.
    :param n_jobs: Number of parallel jobs (default: -1, which uses all available cores).
    :return: The distance matrix.
    """

    def compute_row(i: int, trace1: list) -> list:
        """
        Compute the distances between a single trace and all traces in the second log.

        :param i: Index of the trace in the first log.
        :param trace1: The trace from the first log.
        :return: List of distances between the trace and all traces in the second log.
        """
        return [distance_function(trace1, trace2) for trace2 in log2]

    # Parallel computation of rows in the distance matrix
    distance_matrix = Parallel(n_jobs=n_jobs)(
        delayed(compute_row)(i, trace1) for i, trace1 in tqdm(enumerate(log1), total=len(log1))
    )

    return np.array(distance_matrix)


def consistency(df: pd.DataFrame, config: dict) -> dict:
    """
    Calculate the consistency metrics for the log.

    :param df: DataFrame containing the event log data.
    :param config: Configuration dictionary.
    :return: Dictionary containing the consistency metrics.
    """
    consistency_metrics = {
        "Consistency Start Activity": consistency_start_activity(df, config),
        "Consistency End Activity": consistency_end_activity(df, config),
        "Consistency Directly Following": consistency_directly_following(df, config)
    }

    # Remove key-value pairs where the value is -1
    consistency_metrics = {k: v for k, v in consistency_metrics.items() if v != -1}

    return consistency_metrics


def consistency_directly_following(df: pd.DataFrame, config: dict, ins_col: str = 'y_pred',
                                   act_col: str = 'activity') -> float:
    """
    Calculate the proportion of cases containing solely correct directly following activities.

    Solely correct directly following activities means that, for each case, it verifies whether the directly following
    activities occur in the correct order and have the same non-zero number of predecessors and successors as defined
    in the configuration.

    :param df: The DataFrame containing the log.
    :param config: The configuration dictionary.
    :param ins_col: Column name representing the predicted case ID. Default is 'y_pred'.
    :param act_col: Column name representing the activity. Default is 'activity'.
    :return: The proportion of correct directly following activities in all cases.
    """
    if 'Directly Following' not in config.get('complete_expert_attributes', []):
        return -1

    directly_following = config['complete_expert_values']['Directly Following']
    always_directly_following = [pair for pair, occurrence in zip(
        directly_following['values'], directly_following['occurrences']) if occurrence == 'always']

    if not always_directly_following:
        return -1

    num_correct_directly_following = 0

    for case_id, group in df.groupby(ins_col):
        is_consistent_case = True

        for predecessor, successor in always_directly_following:
            positions_predecessor = group[group[act_col] == predecessor].index.tolist()
            positions_successor = group[group[act_col] == successor].index.tolist()

            if len(positions_predecessor) != len(positions_successor):
                is_consistent_case = False
                break

            if not positions_predecessor or not positions_successor:
                continue

            first_predecessor, last_predecessor = positions_predecessor[0], positions_predecessor[-1]
            first_successor, last_successor = positions_successor[0], positions_successor[-1]

            if first_predecessor > first_successor or last_predecessor > last_successor:
                is_consistent_case = False
                break

            group = group.drop(positions_predecessor + positions_successor)

        if is_consistent_case:
            num_correct_directly_following += 1

    num_cases = len(df[ins_col].unique())

    return (num_correct_directly_following / num_cases) if num_cases else -1


def consistency_end_activity(df: pd.DataFrame, config: dict, ins_col: str = 'y_pred',
                             act_col: str = 'activity') -> float:
    """
    Calculate the proportion of cases containing correct end activities.

    A correct end activity is one that is present in the expert input values for the end activity.

    :param df: DataFrame containing the event log data.
    :param config: Configuration dictionary.
    :param ins_col: Column name representing the predicted case ID. Default is 'y_pred'.
    :param act_col: Column name representing the activity. Default is 'activity'.
    :return: The proportion of correct end activities in all cases.
    """
    if 'End Activity' not in config.get('complete_expert_attributes', []):
        return -1

    end_activities = set(config['complete_expert_values']['End Activity']['values'])

    # Group by the case ID and get the last activity for each case
    last_activities = df.groupby(ins_col)[act_col].last()

    # Count the number of cases with correct end activity
    num_correct_end = last_activities.isin(end_activities).sum()

    # Total number of cases
    num_cases = len(last_activities)

    return (num_correct_end / num_cases) if num_cases else -1


def consistency_start_activity(df: pd.DataFrame, config: dict, ins_col: str = 'y_pred',
                               act_col: str = 'activity') -> float:
    """
    Calculate the proportion of cases containing correct start activities.

    A correct start activity is one that is present in the expert input values for the start activity.

    :param df: DataFrame containing the event log data.
    :param config: Configuration dictionary.
    :param ins_col: Column name representing the predicted case ID. Default is 'y_pred'.
    :param act_col: Column name representing the activity. Default is 'activity'.
    :return: The proportion of correct start activities in all cases.
    """
    if 'Start Activity' not in config.get('complete_expert_attributes', []):
        return -1

    start_activities = set(config['complete_expert_values']['Start Activity']['values'])

    # Group by the case ID and get the first activity for each case
    first_activities = df.groupby(ins_col)[act_col].first()

    # Count the number of cases with correct start activity
    num_correct_start = first_activities.isin(start_activities).sum()

    # Total number of cases
    num_cases = len(first_activities)

    return (num_correct_start / num_cases) if num_cases else -1


def event_time_deviation(df: pd.DataFrame, ins_col: str = 'y_pred', true_col: str = 'y_true',
                         timestamp_col: str = 'timestamp'):
    """
    Computes the event-time deviation (SMAPEET) between the predicted and true logs.

    :param df: A pandas DataFrame containing event log data.
    :param ins_col: The column name representing the predicted case ID. Default is 'y_pred'.
    :param true_col: The column name representing the true case ID. Default is 'y_true'.
    :param timestamp_col: The column name representing timestamps of events. Default is 'timestamp'.
    :return: The event-time deviation (SMAPEET).
    """

    def compute_elapsed_times(df: pd.DataFrame, case_col: str, timestamp_col: str) -> pd.Series:
        """
        Helper function to calculate elapsed times for each event in a case.

        :param df: A pandas DataFrame containing event log data.
        :param case_col: The column name representing the case ID.
        :param timestamp_col: The column name representing timestamps of events.
        :return: Series containing the elapsed times for each event in a case.
        """
        elapsed_times = (
            df.sort_values(by=[case_col, timestamp_col])
            .groupby(case_col)[timestamp_col]
            .diff()
            .fillna(pd.Timedelta(seconds=0))
        )

        # Convert timedelta to seconds as float
        return elapsed_times.dt.total_seconds()

    df_tmp = df.copy()

    # Compute elapsed times for true and predicted logs
    df_tmp['ET_true'] = compute_elapsed_times(df_tmp, true_col, timestamp_col)
    df_tmp['ET_pred'] = compute_elapsed_times(df_tmp, ins_col, timestamp_col)

    # Compute SMAPE
    smape_numerator = (df_tmp['ET_true'] - df_tmp['ET_pred']).abs()
    smape_denominator = (df_tmp['ET_true'].abs() + df_tmp['ET_pred'].abs()).replace(0, 1e-10)  # Avoid division by zero
    smape_values = smape_numerator / smape_denominator

    # Aggregate over all events
    smapeet = smape_values.mean()

    return smapeet


def get_minimal_distances(repaired_traces: list, true_traces: list) -> tuple[list, list]:
    """
    Calculates the minimal distances between each repaired trace and the closest true trace.

    :param repaired_traces: List of repaired traces.
    :param true_traces: List of true traces.
    :return: Tuple containing the minimal distances and the closest true traces.
    """
    minimal_distances = []
    closest_traces = []

    for repaired_trace in tqdm(repaired_traces):
        min_distance = float('inf')
        closest_trace = None

        for true_trace in true_traces:
            distance = ins_del_distance(repaired_trace, true_trace)
            if distance < min_distance:
                min_distance = distance
                closest_trace = true_trace

        minimal_distances.append(min_distance)
        closest_traces.append(closest_trace)

    return minimal_distances, closest_traces


# Parallelized computation of minimal distances
def get_minimal_distances_parallel(repaired_traces: list, true_traces: list, n_jobs: int = -1) -> tuple[list, list]:
    """
    Calculates the minimal distances between each repaired trace and the closest true trace with parallel processing.

    :param repaired_traces: List of repaired traces.
    :param true_traces: List of true traces.
    :param n_jobs: Number of parallel jobs (default: -1, which uses all available cores).
    :return: Tuple containing the minimal distances and the closest true traces.
    """

    def compute_for_repaired_trace(repaired_trace: list) -> tuple[float, list]:
        """
        Helper function to compute the minimal distance for a single repaired trace.

        :param repaired_trace: The repaired trace.
        :return: Tuple containing the minimal distance and the closest true trace.
        """
        min_distance = float('inf')
        closest_trace = None

        for true_trace in true_traces:
            distance = optimized_ins_del_distance(repaired_trace, true_trace)
            if distance < min_distance:
                min_distance = distance
                closest_trace = true_trace

        return min_distance, closest_trace

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_for_repaired_trace)(repaired_trace)
        for repaired_trace in tqdm(repaired_traces)
    )

    minimal_distances, closest_traces = zip(*results)

    return list(minimal_distances), list(closest_traces)


def ins_del_distance(trace1: list, trace2: list) -> int:
    """
    Calculates the Levenshtein distance between two traces with only insertions and deletions.

    :param trace1: The first trace.
    :param trace2: The second trace.
    :return: The Levenshtein distance between the two traces.
    """
    global memorized_distances

    # Check if distance has already been computed
    if (trace1, trace2) in memorized_distances:
        return memorized_distances[(trace1, trace2)]

    len1, len2 = len(trace1), len(trace2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(1, len1 + 1):
        dp[i][0] = i
    for j in range(1, len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if trace1[i - 1] == trace2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1

    # Store result in memoization dictionary
    distance = dp[len1][len2]
    memorized_distances[(trace1, trace2)] = distance

    return distance


# Optimized ins_del_distance using two-row matrix
def optimized_ins_del_distance(trace1: list, trace2: list) -> int:
    """
    Calculates the Levenshtein distance between two traces with only insertions and deletions using optimized
    operations.

    :param trace1: The first trace.
    :param trace2: The second trace.
    :return: The Levenshtein distance between the two traces.
    """
    len1, len2 = len(trace1), len(trace2)
    previous_row = list(range(len2 + 1))
    current_row = [0] * (len2 + 1)

    for i in range(1, len1 + 1):
        current_row[0] = i
        for j in range(1, len2 + 1):
            if trace1[i - 1] == trace2[j - 1]:
                current_row[j] = previous_row[j - 1]
            else:
                current_row[j] = min(previous_row[j], current_row[j - 1]) + 1
        previous_row, current_row = current_row, previous_row

    return previous_row[len2]


def partial_case_similarity_optimized(df: pd.DataFrame, ins_col: str = 'y_pred', true_col: str = 'y_true',
                                      timestamp_col: str = 'timestamp', event_id_col: str = 'event_id'):
    """
    Computes the partial case similarity (L2Lfirst) directly from a DataFrame using optimized operations.

    :param df: A pandas DataFrame containing event log data.
    :param ins_col: The column name representing the predicted case ID. Default is 'y_pred'.
    :param true_col: The column name representing the true case ID. Default is 'y_true'.
    :param timestamp_col: The column name representing event timestamps. Default is 'timestamp'.
    :param event_id_col: The column name representing the event ID. Default is 'event_id'.
    :return: The partial case similarity value.
    """
    # Sort and extract cases
    true_cases = (
        df.sort_values(by=[true_col, timestamp_col])
        .groupby(true_col)[event_id_col]
        .apply(list)
    )
    pred_cases = (
        df.sort_values(by=[ins_col, timestamp_col])
        .groupby(ins_col)[event_id_col]
        .apply(list)
    )

    # Convert to DataFrame for fast merging
    true_starts = true_cases.apply(lambda x: x[0]).reset_index(name='start_event')
    pred_starts = pred_cases.apply(lambda x: x[0]).reset_index(name='start_event')

    # Merge on start events to find matching cases
    matched_cases = pd.merge(true_starts, pred_starts, on='start_event', suffixes=('_true', '_pred'))

    # Compute intersections efficiently
    def compute_intersection(row: pd.Series) -> tuple[int, int]:
        """
        Compute the intersection of events between the true and predicted cases.

        :param row: A row from the DataFrame containing the start events of matched cases.
        :return: Tuple containing the number of intersecting events and the total number of non-start events.
        """
        true_case = true_cases[row[true_col]]
        pred_case = pred_cases[row[ins_col]]

        return len(set(true_case[1:]).intersection(pred_case[1:])), len(true_case[1:])

    intersections = matched_cases.apply(compute_intersection, axis=1)

    # Extract intersection and non-start events
    total_intersections = sum(inter[0] for inter in intersections)
    total_non_start_events = sum(inter[1] for inter in intersections)

    # Avoid division by zero
    if total_non_start_events == 0:
        return 0.0

    similarity = total_intersections / total_non_start_events

    return similarity


def prepare_data(data_path: str, log_name: str,
                 config_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Load and prepare data for evaluation.

    :param data_path: Path to the folder containing the log.
    :param log_name: Name of the log file.
    :param config_path: Path to the configuration file.
    :return: Tuple containing the DataFrame, correct predictions, wrong predictions, and the model configuration.
    """
    file_path = os.path.join(data_path, log_name)

    df = pd.read_csv(file_path)
    df['is_erroneous'] = df['Original Case ID'].isnull()
    df = df[['Sorted Index', 'is_erroneous', 'Determined Case ID', 'Ground Truth Case ID', 'Activity', 'Timestamp',
             'Resource']]
    df.columns = ['event_id', 'is_erroneous', 'y_pred', 'y_true', 'activity', 'timestamp', 'resource']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['y_true', 'timestamp'])

    repaired_events = df[df.is_erroneous].copy()
    correct_predictions = repaired_events[repaired_events.y_true == repaired_events.y_pred].copy()
    wrong_predictions = repaired_events[repaired_events.y_true != repaired_events.y_pred].copy()

    model_configuration = {}
    for file in os.listdir(config_path):
        if file.endswith('.pkl'):
            with open(os.path.join(config_path, file), 'rb') as f:
                model_configuration = pickle.load(f)

            model_configuration = {key: value.to_dict() if isinstance(value, pd.DataFrame) else value for key, value in
                                   model_configuration.items()}

    return df, correct_predictions, wrong_predictions, model_configuration


def trace_to_trace_frequency_similarity(log1: list, log2: list, ins_del_distance: callable) -> float:
    """
    Compute the trace-to-trace frequency similarity between two event logs.

    :param log1: List of traces from the first log.
    :param log2: List of traces from the second log.
    :param ins_del_distance: Function to compute the insertion-deletion distance.
    :return: L2Lfreq: The trace-to-trace frequency similarity value.
    """
    # Step 1: Compute the distance matrix
    print("Computing distance matrix...")
    distance_matrix = compute_distance_matrix(log1, log2, ins_del_distance)

    # Step 2: Solve the assignment problem using the Hungarian Algorithm
    print("Solving assignment problem...")
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    minimal_total_distance = distance_matrix[row_ind, col_ind].sum()

    # Step 3: Compute total number of events in both logs
    print("Computing total number of events...")
    total_events = sum(len(trace) for trace in log1) + sum(len(trace) for trace in log2)

    # Step 4: Compute the similarity
    print("Computing similarity...")
    L2Lfreq = 1 - (minimal_total_distance / (2 * total_events))

    return L2Lfreq


# Optimized trace_to_trace_frequency_similarity
def trace_to_trace_frequency_similarity_parallel(log1: list, log2: list, ins_del_distance: callable,
                                                 n_jobs: int = -1) -> float:
    """
    Compute the trace-to-trace frequency similarity between two event logs with parallel processing.

    :param log1: List of traces from the first log.
    :param log2: List of traces from the second log.
    :param ins_del_distance: Function to compute the insertion-deletion distance.
    :param n_jobs: Number of parallel jobs (default: -1, which uses all available cores).
    :return: L2Lfreq: The trace-to-trace frequency similarity value.
    """
    # Step 1: Compute the distance matrix
    print("Computing distance matrix...")
    distance_matrix = compute_distance_matrix_parallel(log1, log2, ins_del_distance, n_jobs=n_jobs)

    # Step 2: Solve the assignment problem using the Hungarian Algorithm
    print("Solving assignment problem...")
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    minimal_total_distance = distance_matrix[row_ind, col_ind].sum()

    # Step 3: Compute total number of events in both logs
    print("Computing total number of events...")
    total_events = sum(len(trace) for trace in log1) + sum(len(trace) for trace in log2)

    # Step 4: Compute the similarity
    print("Computing similarity...")
    L2Lfreq = 1 - (minimal_total_distance / (2 * total_events))

    return L2Lfreq


def trace2trace_similarity(repaired_traces: list, true_traces: list) -> float:
    """
    Compute the trace-to-trace similarity.

    :param repaired_traces: List of repaired traces.
    :param true_traces: List of true traces.
    :return: The trace-to-trace similarity.
    """
    minimal_distances, closest_traces = get_minimal_distances(repaired_traces, true_traces)
    sum_minimal_distances = sum(minimal_distances)
    total_length = sum(len(trace) for trace in repaired_traces) + sum(len(trace) for trace in closest_traces)
    similarity = 1 - (sum_minimal_distances / total_length)

    return similarity


# Compute similarity with the parallelized function
def trace2trace_similarity_parallel(repaired_traces: list, true_traces: list) -> float:
    """
    Compute the trace-to-trace similarity.

    :param repaired_traces: List of repaired traces.
    :param true_traces: List of true traces.
    :return: The trace-to-trace similarity.
    """
    minimal_distances, closest_traces = get_minimal_distances_parallel(repaired_traces, true_traces)
    sum_minimal_distances = sum(minimal_distances)
    total_length = sum(len(trace) for trace in repaired_traces) + sum(len(trace) for trace in closest_traces)
    similarity = 1 - (sum_minimal_distances / total_length)

    return similarity


# Example Usage
def trigram_similarity(df: pd.DataFrame, ins_col: str = 'y_pred', true_col: str = 'y_true',
                       timestamp_col: str = 'timestamp', event_id_col: str = 'event_id') -> float:
    """
    Computes the trigram similarity (L2L3gram) between the predicted and true cases.

    :param df: A pandas DataFrame containing event log data.
    :param ins_col: The column name representing the predicted case ID. Default is 'y_pred'.
    :param true_col: The column name representing the true case ID. Default is 'y_true'.
    :param timestamp_col: The column name representing event timestamps. Default is 'timestamp'.
    :param event_id_col: The column name representing the event ID. Default is 'event_id'.
    :return: The trigram similarity value.
    """
    # Extract true and predicted cases
    true_cases = (
        df.sort_values(by=[true_col, timestamp_col])
        .groupby(true_col)[event_id_col]
        .apply(list)
    )
    pred_cases = (
        df.sort_values(by=[ins_col, timestamp_col])
        .groupby(ins_col)[event_id_col]
        .apply(list)
    )

    # Flatten predicted cases into a set of all trigrams in L'
    pred_trigrams = set(
        (pred[i - 1], pred[i], pred[i + 1])
        for pred in pred_cases
        for i in range(1, len(pred) - 1)
    )

    # Compute trigram similarity
    def case_trigram_similarity(case):
        trigrams = [
            (case[i - 1], case[i], case[i + 1]) for i in range(1, len(case) - 1)
        ]
        if not trigrams:
            return 0  # Avoid division by zero for cases with fewer than 3 events
        matching_trigrams = sum(1 for trigram in trigrams if trigram in pred_trigrams)
        return matching_trigrams / len(trigrams)

    total_similarity = sum(case_trigram_similarity(case) for case in true_cases)
    num_cases = len(true_cases)

    # Normalize over all cases
    if num_cases == 0:
        return 0.0
    return total_similarity / num_cases


if __name__ == "__main__":
    # Define base paths and folder structure
    base_paths = [
        '../Data/LSTM_Repair/Hospital Billing',
        '../Data/LSTM_Repair/Renting',
        '../Data/LSTM_Repair/Review',
        '../Data/Transformer_Repair/Hospital Billing/Configuration 1',
        '../Data/Transformer_Repair/Hospital Billing/Configuration 2',
        '../Data/Transformer_Repair/Renting/Configuration 1',
        '../Data/Transformer_Repair/Renting/Configuration 2',
        '../Data/Transformer_Repair/Review/Configuration 1',
        '../Data/Transformer_Repair/Review/Configuration 2',
        '../Data/RandomWithDist_Repair/Hospital Billing',
        '../Data/RandomWithDist_Repair/Renting',
        '../Data/RandomWithDist_Repair/Review',
        '../Data/Random_Repair/Hospital Billing',
        '../Data/Random_Repair/Renting',
        '../Data/Random_Repair/Review'
    ]

    configuration_paths = {
        'Hospital Billing': '../Data/Configuration/Hospital Billing',
        'Renting': '../Data/Configuration/Renting',
        'Review': '../Data/Configuration/Review'
    }

    # Path to the results CSV
    results_csv_path = "../Data/log_metrics_results.csv"

    # Load existing results if the CSV exists
    if os.path.exists(results_csv_path):
        existing_results_df = pd.read_csv(results_csv_path)
        processed_logs = set(existing_results_df["Log Name"])
        print(f"Loaded existing results. {len(processed_logs)} logs already processed.")
    else:
        existing_results_df = pd.DataFrame()
        processed_logs = set()
        print("No existing results found. Starting fresh.")

    # Initialize results list
    results = []

    # Iterate through each folder and process logs
    for base_path in base_paths:
        model_type = "LSTM_Repair" if "LSTM_Repair" in base_path \
            else "Transformer_Repair" if "Transformer_Repair" in base_path \
            else "RandomWithDist_Repair" if "RandomWithDist_Repair" in base_path else "Random_Repair"
        log_type = "Hospital Billing" if "Hospital Billing" in base_path \
            else "Renting" if "Renting" in base_path else "Review"
        configuration = "/" if "LSTM_Repair" in base_path \
            else "Configuration 1" if "Configuration 1" in base_path else "Configuration 2"

        config_path = configuration_paths.get(log_type, None)

        # Iterate over all logs in the folder
        for log_name in os.listdir(base_path):
            if log_name.endswith('.csv'):
                # Skip logs already processed
                if log_name in processed_logs:
                    print(f"Skipping {log_name}, already processed.")
                    continue

                print(f"Processing {log_name} in {base_path}...")
                try:
                    # Calculate metrics
                    log_metrics = calculate_metrics_for_log(base_path, log_name, config_path)

                    # Append results
                    result = {
                        "Model": model_type,
                        "Log": log_type,
                        "Configuration": configuration,
                        "Log Name": log_name,
                        **log_metrics
                    }
                    results.append(result)

                    # Save to CSV incrementally
                    temp_df = pd.DataFrame([result])
                    if existing_results_df.empty:
                        existing_results_df = temp_df
                    else:
                        existing_results_df = pd.concat([existing_results_df, temp_df], ignore_index=True)
                    existing_results_df.to_csv(results_csv_path, index=False)

                    print(f"\n{'=' * 50}")
                    print(f"Results for Log: {log_name}")
                    print(f"Model: {model_type}")
                    print(f"Log: {log_type}")
                    print(f"Configuration: {configuration}")
                    print(f"{'-' * 50}")

                    for metric, value in log_metrics.items():
                        print(f"{metric}: {value:.2f}")

                    print(f"\n{'=' * 50}\n")

                except Exception as e:
                    print(f"Error processing {log_name}: {e}")

    print("Processing complete. Final results saved to log_metrics_results.csv.")
