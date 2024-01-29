from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import os

log_path = None  # Variable to store the result of get_file_path


def get_config():
    global log_path  # Use the global variable
    if log_path is None:
        log_path = get_file_path()  # Call get_file_path only if log_path is not already set
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10 ** -4,
        "seq_len": 12,
        "d_model": 512,
        "log_path": log_path,  # Use the stored value
        "tf_input": "Activity",
        "tf_output": "Case ID",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])


def get_file_path():
    if "DISPLAY" in os.environ:
        # GUI components can be used
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        file_path = filedialog.askopenfilename(title="Select the file that contains the event log")

        # Check if the user selected a file or canceled the dialog
        if file_path:
            return file_path
        else:
            raise ValueError("Error: No file selected.")
    else:
        # No display available, use alternative method (e.g., manual input)
        file_path = input("Enter the path to the file that contains the event log: ")

        # Check if the user entered a file path
        if file_path:
            file_path = file_path.strip('"')
            return file_path
        else:
            raise ValueError("Error: No file selected.")
