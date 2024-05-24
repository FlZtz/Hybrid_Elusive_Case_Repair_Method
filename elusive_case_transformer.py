# elusive_case_transformer.py - A dummy script to demonstrate the interaction between the user and the system.
from typing import List

import pandas as pd

added_expert_knowledge: bool
expert_attribute: str
expert_knowledge: bool
input_attributes: List[str]
new_expert_attribute: str
predetermined: bool
rule_checking: bool
threshold_value: float


def add_expert_input() -> None:
    """
    Add expert input.
    """
    global added_expert_knowledge, expert_knowledge

    print("Configuration of log repair\n"
          "Do you want to predict the Case ID values using the event log that was used for training? (yes/no): " +
          "\033[1m" + "no\n" + "\033[0m" +
          "Please ensure the new event log and its name match the process used during training.\n"
          "Enter the path to the file that contains the event log: " +
          "\033[1m" + "logs/elusive/renting_log_low.xes\n" + "\033[0m" +
          "XES file successfully read.\n" +
          "\033[3m" + "Data Loading" + "\033[0m")

    if expert_knowledge:
        print("Please note that incorporating declarative rule checking may result in assumption-based modifications.\n"
              "Do you want to proceed with ex ante rule checking in this iteration? (yes/no): " +
              "\033[1m" + "no" + "\033[0m")

    print("Please enter the minimum probability (in %) with which the Case ID must be determined in order for it to be "
          "accepted: " +
          "\033[1m" + "25" + "\033[0m")

    if expert_knowledge:
        print("Please note that incorporating declarative rule checking may result in assumption-based modifications.\n"
              "Do you want to proceed with ex post rule checking in this iteration? (yes/no): " +
              "\033[1m" + "no" + "\033[0m")

    print("For 10.00% of the events, the Case ID has originally not been recorded.")

    show_first_output()

    print("Do you want to use the repaired log as the baseline for an additional repair? (yes/no): " +
          "\033[1m" + "yes\n" + "\033[0m" +
          "Do you want to keep the probability threshold for the next repair? (yes/no): " +
          "\033[1m" + "no\n" + "\033[0m")

    choice = input("Do you want to add one or more expert attributes? (yes/no): ").strip().lower()

    if not choice or choice not in ['yes', 'no']:
        raise ValueError("Invalid choice.")

    added_expert_knowledge = choice == 'yes'

    print("\033[3m" + f"Expert attributes: {'Yes' if added_expert_knowledge else 'No'}" + "\033[0m")


def expert_input() -> None:
    """
    Provide expert input.
    """
    global expert_knowledge

    choice = input("Do you want to add one or more expert attributes? (yes/no): ").strip().lower()

    if not choice or choice not in ['yes', 'no']:
        raise ValueError("Invalid choice.")

    expert_knowledge = choice == 'yes'

    print("\033[3m" + f"Expert attributes: {'Yes' if expert_knowledge else 'No'}" + "\033[0m")


def get_attribute(attribute: str, first: bool = True) -> None:
    """
    Get the expert attribute values.

    :param attribute: User input for the expert attribute.
    :param first: Flag indicating whether it is the first function call. Default is True.
    """
    global expert_attribute, new_expert_attribute

    if not attribute or attribute not in ['start activity', 'end activity']:
        raise ValueError("Invalid input.")

    if attribute == 'start activity':
        storage = 'Start Activity'
    else:
        storage = 'End Activity'

    print("\033[3m" + f"Expert attribute: {storage}" + "\033[0m")

    if storage == 'Start Activity':
        print("Please enter the value(s) that represent(s) the attribute 'Start Activity' (separated by commas) –\n"
              "Suggestions (proportion of cases with corresponding Activity as the Start Activity): "
              "Apply for Viewing Appointment (100.00%):\n" +
              "\033[1m" + "Apply for Viewing Appointment" + "\033[0m")
        print("Does 'Apply for Viewing Appointment' always or sometimes represent the attribute 'Start Activity'?\n"
              "Enter 'always' or 'sometimes': " +
              "\033[1m" + "always" + "\033[0m")
    else:
        print("Please enter the value(s) that represent(s) the attribute 'End Activity' (separated by commas) –\n"
              "Suggestions (proportion of cases with corresponding Activity as the End Activity): "
              "Reject Prospective Tenant (65.90%), Tenant Cancels Appartment (32.73%), Evict Tenant (1.37%):\n" +
              "\033[1m" + "Reject Prospective Tenant, Tenant Cancels Appartment, Evict Tenant" + "\033[0m")
        print("Does 'Reject Prospective Tenant' always or sometimes represent the attribute 'End Activity'?\n"
              "Enter 'always' or 'sometimes': " +
              "\033[1m" + "sometimes" + "\033[0m")
        print("Does 'Tenant Cancels Appartment' always or sometimes represent the attribute 'End Activity'?\n"
              "Enter 'always' or 'sometimes': " +
              "\033[1m" + "sometimes" + "\033[0m")
        print("Does 'Evict Tenant' always or sometimes represent the attribute 'End Activity'?\n"
              "Enter 'always' or 'sometimes': " +
              "\033[1m" + "sometimes" + "\033[0m")

    if first:
        expert_attribute = storage
    else:
        new_expert_attribute = storage


def get_model() -> int:
    """
    Get the model.

    :return: Model number.
    """
    global expert_attribute, expert_knowledge, input_attributes

    if 'Resource' not in input_attributes:
        if expert_knowledge:
            return 1 if expert_attribute == 'Start Activity' else 2
        else:
            return 3
    else:
        if expert_knowledge:
            return 4 if expert_attribute == 'Start Activity' else 5
        else:
            return 6


def get_outcome() -> int:
    """
    Get the outcome.

    :return: Outcome number.
    """
    global added_expert_knowledge, new_expert_attribute, rule_checking, threshold_value

    model = get_model()

    if model == 1:
        if added_expert_knowledge:
            if rule_checking:
                return 1 if threshold_value == 0 else 2
            else:
                return 3 if threshold_value == 0 else 4
        else:
            if rule_checking:
                return 5 if threshold_value == 0 else 6
            else:
                return 7 if threshold_value == 0 else 8
    elif model == 2:
        if added_expert_knowledge:
            if rule_checking:
                return 9 if threshold_value == 0 else 10
            else:
                return 11 if threshold_value == 0 else 12
        else:
            if rule_checking:
                return 13 if threshold_value == 0 else 14
            else:
                return 15 if threshold_value == 0 else 16
    elif model == 3:
        if added_expert_knowledge:
            if new_expert_attribute == 'Start Activity':
                return 17 if threshold_value == 0 else 18
            else:
                return 19 if threshold_value == 0 else 20
        else:
            return 21 if threshold_value == 0 else 22
    elif model == 4:
        if added_expert_knowledge:
            if rule_checking:
                return 23 if threshold_value == 0 else 24
            else:
                return 25 if threshold_value == 0 else 26
        else:
            if rule_checking:
                return 27 if threshold_value == 0 else 28
            else:
                return 29 if threshold_value == 0 else 30
    elif model == 5:
        if added_expert_knowledge:
            if rule_checking:
                return 31 if threshold_value == 0 else 32
            else:
                return 33 if threshold_value == 0 else 34
        else:
            if rule_checking:
                return 35 if threshold_value == 0 else 36
            else:
                return 37 if threshold_value == 0 else 38
    else:
        if added_expert_knowledge:
            if new_expert_attribute == 'Start Activity':
                return 39 if threshold_value == 0 else 40
            else:
                return 41 if threshold_value == 0 else 42
        else:
            return 43 if threshold_value == 0 else 44


def provide_added_expert_input() -> None:
    """
    Provide additional expert input.
    """
    global added_expert_knowledge, expert_attribute, expert_knowledge, new_expert_attribute, predetermined

    new_expert_attribute = ''
    predetermined = False

    if not added_expert_knowledge:
        print("\033[3m" + "No additional expert attributes provided." + "\033[0m")
        return

    if expert_knowledge:
        if expert_attribute == 'Start Activity':
            print("Following expert attributes are available: Start Activity, End Activity and Directly Following.\n"
                  "Please enter the attribute(s) for which you have expert knowledge (separated by commas): " +
                  "\033[1m" + "End Activity" + "\033[0m")
            new_expert_attribute = 'End Activity'
        else:
            print("Following expert attributes are available: Start Activity, End Activity and Directly Following.\n"
                  "Please enter the attribute(s) for which you have expert knowledge (separated by commas): " +
                  "\033[1m" + "Start Activity" + "\033[0m")
            new_expert_attribute = 'Start Activity'

        predetermined = True
        return

    attribute = input("Following expert attributes are available: Start Activity, End Activity and Directly "
                      "Following.\n"
                      "Please enter the attribute(s) for which you have expert knowledge "
                      "(separated by commas): ").strip().lower()

    get_attribute(attribute, False)


def provide_expert_input() -> None:
    """
    Provide expert input.
    """
    global expert_attribute, expert_knowledge

    expert_attribute = ''

    if not expert_knowledge:
        print("\033[3m" + "No expert attributes provided." + "\033[0m")
        return

    attribute = input("Following expert attributes are available: Start Activity, End Activity and Directly "
                      "Following.\n"
                      "Please enter the attribute(s) for which you have expert knowledge "
                      "(separated by commas): ").strip().lower()

    get_attribute(attribute)


def rule_check() -> None:
    """
    Perform the rule check.
    """
    global added_expert_knowledge, expert_knowledge, predetermined, rule_checking

    rule_checking = False

    if not expert_knowledge and not added_expert_knowledge:
        print("\033[3m" + "No expert attributes provided." + "\033[0m")
        return

    if added_expert_knowledge and not predetermined:
        print("Please note that incorporating declarative rule checking may result in assumption-based modifications.\n"
              "Do you want to proceed with ex post rule checking in this iteration? (yes/no): " +
              "\033[1m" + "yes" + "\033[0m")
        rule_checking = True
        return

    choice = input("Please note that incorporating declarative rule checking may result in assumption-based "
                   "modifications.\n"
                   "Do you want to proceed with ex ante rule checking in this iteration? (yes/no): ").strip().lower()

    if not choice or choice not in ['yes', 'no']:
        raise ValueError("Invalid choice.")

    rule_checking = choice == 'yes'

    print("\033[3m" + f"Rule checking: {'Yes' if rule_checking else 'No'}" + "\033[0m")


def show_first_output() -> None:
    """
    Show the first output.
    """
    print("\nRepair Iteration 1:\n")

    model = get_model()

    if model == 1:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.87% of the events, the Case ID has not yet been determined.")
    elif model == 2:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.96% of the events, the Case ID has not yet been determined.")
    elif model == 3:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.86% of the events, the Case ID has not yet been determined.")
    elif model == 4:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.54% of the events, the Case ID has not yet been determined.")
    elif model == 5:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.87% of the events, the Case ID has not yet been determined.")
    else:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.56% of the events, the Case ID has not yet been determined.")

    print('-' * 80)


def show_second_output() -> None:
    """
    Show the second output.
    """
    print("\nRepair Iteration 2:\n")

    outcome = get_outcome()

    if outcome == 1:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 1.14% of the events, the Case ID has not yet been determined.")
    elif outcome == 2:
        df = pd.DataFrame({
            'Determined Case ID': [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            'Iteration Probability': [pd.NA] * 10,
            'Iteration Follow-up Probability': [pd.NA] * 10,
            'Previous Case ID': [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            'Determination Probability': [pd.NA] * 9 + [float('nan')],
            'Determination Follow-up Probability': [pd.NA] * 9 + [float('nan')],
            'Original Case ID': [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            'Activity': ['Apply for Viewing Appointment'] * 10,
            'Timestamp': [pd.to_datetime('2015-01-05')] * 10,
            'Sorted Index': list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.84% of the events, the Case ID has not yet been determined.")
    elif outcome == 3:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 6015],
            "Iteration Probability": [pd.NA] * 9 + ["86.27%"],
            "Iteration Follow-up Probability": [pd.NA] * 9 + ["0.35%"],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For all events, the Case ID has been determined.")
        return
    elif outcome == 4:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.85% of the events, the Case ID has not yet been determined.")
    elif outcome == 5:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 0.81% of the events, the Case ID has not yet been determined.")
    elif outcome == 6:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.80% of the events, the Case ID has not yet been determined.")
    elif outcome == 7:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 2112],
            "Iteration Probability": [pd.NA] * 9 + ["87.71%"],
            "Iteration Follow-up Probability": [pd.NA] * 9 + ["0.18%"],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For all events, the Case ID has been determined.")
        return
    elif outcome == 8:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.87% of the events, the Case ID has not yet been determined.")
    elif outcome == 9:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 1.35% of the events, the Case ID has not yet been determined.")
    elif outcome == 10:
        df = pd.DataFrame({
            'Determined Case ID': [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            'Iteration Probability': [pd.NA] * 10,
            'Iteration Follow-up Probability': [pd.NA] * 10,
            'Previous Case ID': [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            'Determination Probability': [pd.NA] * 9 + [float('nan')],
            'Determination Follow-up Probability': [pd.NA] * 9 + [float('nan')],
            'Original Case ID': [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            'Activity': ['Apply for Viewing Appointment'] * 10,
            'Timestamp': [pd.to_datetime('2015-01-05')] * 10,
            'Sorted Index': list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.94% of the events, the Case ID has not yet been determined.")
    elif outcome == 11:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 3767],
            "Iteration Probability": [pd.NA] * 9 + ["94.50%"],
            "Iteration Follow-up Probability": [pd.NA] * 9 + ["0.10%"],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For all events, the Case ID has been determined.")
        return
    elif outcome == 12:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.96% of the events, the Case ID has not yet been determined.")
    elif outcome == 13:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 9875],
            "Iteration Probability": [pd.NA] * 9 + ["98.60%"],
            "Iteration Follow-up Probability": [pd.NA] * 9 + ["0.04%"],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 1.24% of the events, the Case ID has not yet been determined.")
    elif outcome == 14:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873] + [float('nan')],
            "Iteration Probability": [pd.NA] * 9 + [float('nan')],
            "Iteration Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873] + [float('nan')],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873] + [float('nan')],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.94% of the events, the Case ID has not yet been determined.")
    elif outcome == 15:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 6789],
            "Iteration Probability": [pd.NA] * 9 + ["96.54%"],
            "Iteration Follow-up Probability": [pd.NA] * 9 + ["0.03%"],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For all events, the Case ID has been determined.")
        return
    elif outcome == 16:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.96% of the events, the Case ID has not yet been determined.")
    elif outcome == 17:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 0.84% of the events, the Case ID has not yet been determined.")
    elif outcome == 18:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.82% of the events, the Case ID has not yet been determined.")
    elif outcome == 19:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 1.19% of the events, the Case ID has not yet been determined.")
    elif outcome == 20:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873] + [float('nan')],
            "Iteration Probability": [pd.NA] * 9 + [float('nan')],
            "Iteration Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873] + [float('nan')],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873] + [float('nan')],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.86% of the events, the Case ID has not yet been determined.")
    elif outcome == 21:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 945],
            "Iteration Probability": [pd.NA] * 9 + ["83.91%"],
            "Iteration Follow-up Probability": [pd.NA] * 9 + ["0.23%"],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For all events, the Case ID has been determined.")
        return
    elif outcome == 22:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.86% of the events, the Case ID has not yet been determined.")
    elif outcome == 23:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 1.28% of the events, the Case ID has not yet been determined.")
    elif outcome == 24:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.52% of the events, the Case ID has not yet been determined.")
    elif outcome == 25:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 608],
            "Iteration Probability": [pd.NA] * 9 + ["91.79%"],
            "Iteration Follow-up Probability": [pd.NA] * 9 + ["0.16%"],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For all events, the Case ID has been determined.")
        return
    elif outcome == 26:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.53% of the events, the Case ID has not yet been determined.")
    elif outcome == 27:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 0.94% of the events, the Case ID has not yet been determined.")
    elif outcome == 28:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.50% of the events, the Case ID has not yet been determined.")
    elif outcome == 29:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 6628],
            "Iteration Probability": [pd.NA] * 9 + ["93.36%"],
            "Iteration Follow-up Probability": [pd.NA] * 9 + ["0.06%"],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For all events, the Case ID has been determined.")
        return
    elif outcome == 30:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.53% of the events, the Case ID has not yet been determined.")
    elif outcome == 31:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 1.30% of the events, the Case ID has not yet been determined.")
    elif outcome == 32:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.83% of the events, the Case ID has not yet been determined.")
    elif outcome == 33:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 8652],
            "Iteration Probability": [pd.NA] * 9 + ["95.70%"],
            "Iteration Follow-up Probability": [pd.NA] * 9 + ["0.07%"],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For all events, the Case ID has been determined.")
        return
    elif outcome == 34:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.86% of the events, the Case ID has not yet been determined.")
    elif outcome == 35:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 1.19% of the events, the Case ID has not yet been determined.")
    elif outcome == 36:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873] + [float('nan')],
            "Iteration Probability": [pd.NA] * 9 + [float('nan')],
            "Iteration Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873] + [float('nan')],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873] + [float('nan')],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.82% of the events, the Case ID has not yet been determined.")
    elif outcome == 37:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 5459],
            "Iteration Probability": [pd.NA] * 9 + ["93.77%"],
            "Iteration Follow-up Probability": [pd.NA] * 9 + ["0.16%"],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For all events, the Case ID has been determined.")
        return
    elif outcome == 38:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.83% of the events, the Case ID has not yet been determined.")
    elif outcome == 39:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 0.78% of the events, the Case ID has not yet been determined.")
    elif outcome == 40:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 190],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.48% of the events, the Case ID has not yet been determined.")
    elif outcome == 41:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 9603],
            "Iteration Probability": [pd.NA] * 9 + ["93.77%"],
            "Iteration Follow-up Probability": [pd.NA] * 9 + ["0.08%"],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 0.93% of the events, the Case ID has not yet been determined.")
    elif outcome == 42:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873] + [float('nan')],
            "Iteration Probability": [pd.NA] * 9 + [float('nan')],
            "Iteration Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873] + [float('nan')],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873] + [float('nan')],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.53% of the events, the Case ID has not yet been determined.")
    elif outcome == 43:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, 2597],
            "Iteration Probability": [pd.NA] * 9 + ["75.55%"],
            "Iteration Follow-up Probability": [pd.NA] * 9 + ["0.18%"],
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 9 + [float('nan')],
            "Determination Follow-up Probability": [pd.NA] * 9 + [float('nan')],
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For all events, the Case ID has been determined.")
        return
    else:
        df = pd.DataFrame({
            "Determined Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Iteration Probability": [pd.NA] * 10,
            "Iteration Follow-up Probability": [pd.NA] * 10,
            "Previous Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Determination Probability": [pd.NA] * 10,
            "Determination Follow-up Probability": [pd.NA] * 10,
            "Original Case ID": [5144, 8244, 4155, 869, 52, 3467, 1249, 5514, 1873, pd.NA],
            "Activity": ["Apply for Viewing Appointment"] * 10,
            "Timestamp": [pd.to_datetime('2015-01-05')] * 10,
            "Resource": ["Real Estate Agent 2"] + ["Hotline"] * 2 + ["Real Estate Agent 2"] * 3 +
                        ["Real Estate Agent 1"] + ["Hotline"] + ["Real Estate Agent 5"] + ["Real Estate Agent 1"],
            "Sorted Index": list(range(10))
        })
        print(df)
        print("\n... (+ 96430 more rows)\n")
        print("For 9.49% of the events, the Case ID has not yet been determined.")

    print('-' * 80)
    print("Do you want to use the repaired log as the baseline for an additional repair? (yes/no): " +
          "\033[1m" + "no\n" + "\033[0m")


def threshold() -> None:
    """
    Set the threshold value.
    """
    global added_expert_knowledge, expert_knowledge, rule_checking, threshold_value

    value = input("Please enter the minimum probability (in %) with which the Case ID must be determined in order for "
                  "it to be accepted: ").strip()

    if not value.isdigit():
        raise ValueError("Invalid input.")

    value = float(value)

    if value != 0 and value != 50:
        raise ValueError("Invalid input.")

    threshold_value = value

    print("\033[3m" + f"Threshold value: {threshold_value}" + "\033[0m")

    if expert_knowledge or added_expert_knowledge:
        suffix = 'yes' if rule_checking else 'no'

        print("Please note that incorporating declarative rule checking may result in assumption-based modifications.\n"
              "Do you want to proceed with ex post rule checking in this iteration? (yes/no): " +
              "\033[1m" + suffix + "\033[0m")

    show_second_output()


def transformer_input() -> None:
    """
    Get the input attributes for the transformer.
    """
    global input_attributes

    print("Do you want to use a specific response configuration file for model training? (yes/no): " +
          "\033[1m" + "no\n\n" + "\033[0m" +
          "Configuration of model training\n"
          "Enter the path to the file that contains the event log: " +
          "\033[1m" + "logs/renting_log_low.xes\n" + "\033[0m" +
          "XES file successfully read.")

    attributes = input("Please enter the input attribute(s) for the transformer (separated by commas): "
                       "Activity, Timestamp, ").strip().lower()

    if attributes and attributes != 'resource':
        raise ValueError("Invalid input.")

    if not attributes:
        input_attributes = ['Activity', 'Timestamp']
    else:
        input_attributes = ['Activity', 'Timestamp', 'Resource']

    print("\033[3m" + f"Input attributes for the transformer: {', '.join(input_attributes)}" + "\033[0m")
