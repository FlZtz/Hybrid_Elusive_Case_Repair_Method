# lstm_repair.py - Repair logs with missing case IDs using an LSTM model.
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


def build_lstm_model(input_shape: tuple, num_classes: int) -> Sequential:
    """
    Build an LSTM model for sequence classification.

    :param input_shape: Shape of the input data.
    :param num_classes: Number of classes.
    :return: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(128))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def evaluate_model(model_path: str, X_test: np.ndarray) -> np.ndarray:
    """
    Evaluate the model on the test data.

    :param model_path: Path to the trained model.
    :param X_test: Test data.
    :return: Predictions on the test data.
    """
    model = load_model(model_path)
    y_pred = model.predict(X_test)

    return y_pred  # Return predictions for further evaluation


def prepare_data_with_sliding_window(data_path: str, log_name: str, window_size: int = 5, test_size: float = 0.2,
                                     random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
pd.DataFrame, LabelEncoder, OneHotEncoder, OneHotEncoder, MinMaxScaler]:
    """
    Prepare the data for training and testing the LSTM model using sliding windows.

    :param data_path: Path to the data.
    :param log_name: Name of the log file.
    :param window_size: Size of the sliding window. Default is 5.
    :param test_size: Fraction of the data to use for testing. Default is 0.2.
    :param random_state: Random state for reproducibility. Default is 42.
    :return: Tuple containing the training and testing data, the cleaned DataFrame, and the encoders.
    """
    # Load and preprocess the data as before
    df = pd.read_csv(os.path.join(data_path, log_name))
    resource_col = [col for col in df.columns if 'resource' in col.lower()][0]
    df = df[['Sorted Index', 'case:concept:name', 'concept:name', 'time:timestamp', resource_col]]
    df.columns = ['event_id', 'case_id', 'activity', 'timestamp', 'resource']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['case_id', 'timestamp'])

    # Remove rows with missing case IDs
    df_cleaned = df.dropna(subset=['case_id']).reset_index(drop=True)

    # One-hot encode categorical variables (activity and resource)
    onehot_encoder_activity = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
    activity_encoded = onehot_encoder_activity.fit_transform(df_cleaned[['activity']])
    onehot_encoder_resource = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
    resource_encoded = onehot_encoder_resource.fit_transform(df_cleaned[['resource']])

    # Scale timestamps
    df_cleaned['time_since_start'] = (df_cleaned['timestamp'] - df_cleaned['timestamp'].min()).dt.total_seconds()
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_cleaned['time_scaled'] = scaler.fit_transform(df_cleaned[['time_since_start']])

    # Concatenate all encoded features
    X_data = np.hstack([activity_encoded, resource_encoded, df_cleaned[['time_scaled']].values])

    # Encode target case IDs
    label_encoder = LabelEncoder()
    case_id_labels = label_encoder.fit_transform(df_cleaned['case_id'])

    # Group data by case ID to create sliding windows
    sequences = []
    targets = []
    for case_id, indices in df_cleaned.groupby('case_id').indices.items():
        X_case = X_data[indices]
        y_case = case_id_labels[indices[0]]  # Target label for this case ID

        # Generate sliding windows
        for i in range(len(X_case) - window_size + 1):
            window = X_case[i:i + window_size]
            sequences.append(window)
            targets.append(y_case)

    # Cap padding at the 95th percentile of sequence lengths
    max_seq_length = min(int(np.percentile([len(seq) for seq in sequences], 95)), window_size)
    X_padded = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype='float32')
    y_targets = np.array(targets)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y_targets, test_size=test_size,
                                                        random_state=random_state)

    # Convert y_train and y_test to categorical
    y_train = to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_test = to_categorical(y_test, num_classes=len(label_encoder.classes_))

    return (X_train, X_test, y_train, y_test, df_cleaned, label_encoder, onehot_encoder_resource,
            onehot_encoder_activity, scaler)


def prepare_input_for_repair(group: pd.DataFrame, onehot_encoder_activity: OneHotEncoder,
                             onehot_encoder_resource: OneHotEncoder, scaler: MinMaxScaler) -> np.ndarray:
    """
    Prepare input features for repairing missing case IDs.

    :param group: Group of events.
    :param onehot_encoder_activity: One-hot encoder for 'activity'.
    :param onehot_encoder_resource: One-hot encoder for 'resource'.
    :param scaler: MinMaxScaler for 'time_since_start'.
    :return: Input features for the LSTM model.
    """
    # One-hot encode 'activity' and 'resource'
    activity_encoded = onehot_encoder_activity.transform(group[['activity']])
    resource_encoded = onehot_encoder_resource.transform(group[['resource']])

    # Scale time feature
    time_scaled = scaler.transform(group[['time_since_start']])

    # Combine features into a single input array
    X_input = np.hstack([activity_encoded, resource_encoded, time_scaled])

    return X_input


def repair_missing_case_ids(model: Sequential, df_original: pd.DataFrame, onehot_encoder_activity: OneHotEncoder,
                            onehot_encoder_resource: OneHotEncoder, scaler: MinMaxScaler,
                            window_size: int = 5) -> pd.DataFrame:
    """
    Repair missing case IDs in the original DataFrame using an LSTM model.

    :param model: Trained LSTM model.
    :param df_original: Original DataFrame with missing case IDs.
    :param onehot_encoder_activity: One-hot encoder for 'activity'.
    :param onehot_encoder_resource: One-hot encoder for 'resource'.
    :param scaler: MinMaxScaler for 'time_since_start'.
    :param window_size: Size of the sliding window. Default is 5.
    :return: Repaired DataFrame with predicted case IDs.
    """
    # Create a DataFrame to store the repaired case IDs
    df_repaired = df_original.copy()

    # Identify the rows with missing case IDs
    missing_case_ids_df = df_repaired[df_repaired['case_id'].isna()]

    # Prepare to collect predictions
    predictions = []

    # Process each event with a missing case ID
    for _, event in tqdm(missing_case_ids_df.iterrows(), total=missing_case_ids_df.shape[0]):
        # Extract the current index to maintain order
        idx = event.name

        # Prepare input features for the current event and its surrounding events
        start_idx = max(0, idx - window_size + 1)
        end_idx = idx + 1  # include current event
        group_window = df_repaired.iloc[start_idx:end_idx]  # window of events

        # Prepare input for prediction
        X_input = prepare_input_for_repair(group_window, onehot_encoder_activity, onehot_encoder_resource, scaler)

        # Create sliding windows over the input data
        if len(X_input) >= window_size:  # Ensure there are enough events to form a complete window
            sequences = [X_input[i:i + window_size] for i in range(len(X_input) - window_size + 1)]
            X_padded = pad_sequences(sequences, padding='post', dtype='float32', maxlen=window_size)

            # Predict using the trained model
            y_pred = model.predict(X_padded, verbose=0)

            # Get the predicted case ID from the last prediction
            predicted_case_id = np.argmax(y_pred[-1])  # Use the last window prediction

            # Decode the predicted case ID to its original format
            if label_encoder:
                predicted_case_id = label_encoder.inverse_transform([predicted_case_id])[0]

            # Store the predicted case ID
            predictions.append((idx, predicted_case_id))

    # Update the original DataFrame with predicted case IDs
    for idx, case_id in predictions:
        df_repaired.at[idx, 'case_id'] = case_id

    return df_repaired


def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                     model_save_path: str = "model_checkpoints", batch_size: int = 32, epochs: int = 20) -> tuple[
    dict, str]:
    """
    Train an LSTM model on the training data.

    :param X_train: Training data.
    :param y_train: Training labels.
    :param X_val: Validation data.
    :param y_val: Validation labels.
    :param model_save_path: Path to save the model checkpoints.
    :param batch_size: Batch size for training. Default is 32.
    :param epochs: Number of epochs for training. Default is 20.
    :return: Training history and path to the best model.
    """
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Initialize model
    num_classes = y_train.shape[1]
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape, num_classes)

    # Model checkpoint
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_save_path, "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=15, verbose=1, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )

    return history, os.path.join(model_save_path, "best_model.keras")


if __name__ == "__main__":
    output_path = 'output'
    for data_path in ['renting', 'review', 'hospital']:
        correct_data_path = os.path.join('Correct Logs', data_path + '.csv')
        for log_name in os.listdir(data_path):
            if os.path.exists(os.path.join(output_path, log_name)):
                print("Skipping", log_name)
                continue
            if log_name == '.ipynb_checkpoints':
                continue

            print("Currently at", log_name)
            # Prepare data
            (X_train, X_test, y_train, y_test, df_cleaned, label_encoder, onehot_encoder_resource,
             onehot_encoder_activity, scaler) = prepare_data_with_sliding_window(data_path, log_name)

            # Train the model
            model_save_dir = "model_checkpoints"
            batch_size = 32
            epochs = 100

            history, best_model_path = train_lstm_model(X_train, y_train, X_test, y_test,
                                                        model_save_path=model_save_dir, batch_size=batch_size,
                                                        epochs=epochs)

            # Load your original data 
            df_original = pd.read_csv(os.path.join(data_path, log_name))

            # Clean the dataframe but keep erroneous entries
            resource_col_og = [col for col in df_original.columns if 'resource' in col.lower()][0]
            df_original = df_original[
                ['Sorted Index', 'case:concept:name', 'concept:name', 'time:timestamp', resource_col_og]]
            df_original.columns = ['event_id', 'case_id', 'activity', 'timestamp', 'resource']
            df_original['timestamp'] = pd.to_datetime(df_original['timestamp'])
            df_original['time_since_start'] = (
                    df_original['timestamp'] - df_original['timestamp'].min()).dt.total_seconds()

            # Load the ground truth data
            df_correct = pd.read_csv(correct_data_path)
            resource_col_corr = [col for col in df_correct.columns if 'resource' in col.lower()][0]
            df_correct = df_correct[
                ['Sorted Index', 'case:concept:name', 'concept:name', 'time:timestamp', resource_col_corr]]
            df_correct.columns = ['event_id', 'case_id', 'activity', 'timestamp', 'resource']

            # Load the trained model
            model = load_model(best_model_path)

            # Repair the dataframe
            df_repaired = repair_missing_case_ids(model, df_original, onehot_encoder_activity, onehot_encoder_resource,
                                                  scaler, window_size=5)

            # Add the ground truth case ID to the repaired dataframe
            df_repaired = df_repaired.merge(df_correct[['event_id', 'case_id']], on='event_id', how='left',
                                            suffixes=('', '_ground_truth'))
            df_repaired.rename(columns={'case_id_ground_truth': 'Ground Truth Case ID'}, inplace=True)

            df_repaired['Determined Case ID'] = df_repaired['case_id']
            df_repaired['Determination Probability'] = ''
            df_repaired['Determination Follow-up Probability'] = ''
            df_repaired['Original Case ID'] = df_original['case_id']
            df_repaired['Activity'] = df_repaired['activity']
            df_repaired['Timestamp'] = df_repaired['timestamp']
            df_repaired['Resource'] = df_repaired['resource']
            df_repaired['Sorted Index'] = df_repaired['event_id']

            df_out = df_repaired[
                ['Determined Case ID', 'Determination Probability', 'Determination Follow-up Probability',
                 'Original Case ID', 'Activity', 'Timestamp', 'Resource', 'Sorted Index', 'Ground Truth Case ID']]

            df_out.to_csv(os.path.join(output_path, log_name), index=False)
