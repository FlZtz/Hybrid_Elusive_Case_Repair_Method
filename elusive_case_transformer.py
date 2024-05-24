# elusive_case_transformer.py - A dummy script to demonstrate the interaction between the user and the system.
from typing import List

from IPython.display import display
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

    print("\n" + "\033[1m" + "For 90.00% of the events, the Case ID was originally recorded. This means that for "
                             "10.00% of the events, or 9,644 events, no Case ID was originally recorded." + "\033[0m")

    show_first_output()

    print("Do you want to use the repaired log as the baseline for an additional repair? (yes/no): " +
          "\033[1m" + "yes\n" + "\033[0m" +
          "Do you want to keep the probability threshold for the next repair? (yes/no): " +
          "\033[1m" + "no\n" + "\033[0m")

    choice = input("Do you want to add one or more expert attributes? (yes/no): ").strip().lower()

    if not choice or choice not in ['yes', 'no']:
        raise ValueError("Invalid choice.")

    added_expert_knowledge = choice == 'yes'

    provide_added_expert_input()


def expert_input() -> None:
    """
    Provide expert input.
    """
    global expert_knowledge

    choice = input("Do you want to add one or more expert attributes? (yes/no): ").strip().lower()

    if not choice or choice not in ['yes', 'no']:
        raise ValueError("Invalid choice.")

    expert_knowledge = choice == 'yes'

    provide_expert_input()


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
            attribute = 'end activity'
        else:
            print("Following expert attributes are available: Start Activity, End Activity and Directly Following.\n"
                  "Please enter the attribute(s) for which you have expert knowledge (separated by commas): " +
                  "\033[1m" + "Start Activity" + "\033[0m")
            attribute = 'start activity'

        predetermined = True
        get_attribute(attribute, False)
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


def show_first_output() -> None:
    """
    Show the first output.
    """
    print("\n" + "\033[1m" + "Repair Iteration 1:" + "\033[0m" + "\n")

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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.13% of the events, the Case ID has been determined, representing 125 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.04% of the events, the Case ID has been determined, representing 38 more events than "
                          "initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.14% of the events, the Case ID has been determined, representing 135 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.46% of the events, the Case ID has been determined, representing 443 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.13% of the events, the Case ID has been determined, representing 125 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.44% of the events, the Case ID has been determined, representing 424 more events "
                          "than initially recorded.")

    print('-' * 80)


def show_second_output() -> None:
    """
    Show the second output.
    """
    print("\n" + "\033[1m" + "Repair Iteration 2:" + "\033[0m" + "\n")

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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 98.86% of the events, the Case ID has been determined, representing 8,554 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.16% of the events, the Case ID has been determined, representing 154 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 100.00% of the events, the Case ID has been determined, representing 9,644 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.15% of the events, the Case ID has been determined, representing 144 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 99.19% of the events, the Case ID has been determined, representing 8,862 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.20% of the events, the Case ID has been determined, representing 192 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 100.00% of the events, the Case ID has been determined, representing 9,644 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.13% of the events, the Case ID has been determined, representing 125 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 98.65% of the events, the Case ID has been determined, representing 8,342 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.06% of the events, the Case ID has been determined, representing 57 more events than "
                          "initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 100.00% of the events, the Case ID has been determined, representing 9,644 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.04% of the events, the Case ID has been determined, representing 38 more events than "
                          "initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 98.76% of the events, the Case ID has been determined, representing 8,448 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.06% of the events, the Case ID has been determined, representing 57 more events than "
                          "initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 100.00% of the events, the Case ID has been determined, representing 9,644 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.04% of the events, the Case ID has been determined, representing 38 more events than "
                          "initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 99.16% of the events, the Case ID has been determined, representing 8,833 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.18% of the events, the Case ID has been determined, representing 173 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 98.81% of the events, the Case ID has been determined, representing 8,496 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.14% of the events, the Case ID has been determined, representing 135 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 100.00% of the events, the Case ID has been determined, representing 9,644 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.14% of the events, the Case ID has been determined, representing 135 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 98.72% of the events, the Case ID has been determined, representing 8,409 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.48% of the events, the Case ID has been determined, representing 462 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 100.00% of the events, the Case ID has been determined, representing 9,644 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.47% of the events, the Case ID has been determined, representing 453 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 99.06% of the events, the Case ID has been determined, representing 8,737 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.50% of the events, the Case ID has been determined, representing 482 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 100.00% of the events, the Case ID has been determined, representing 9,644 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.47% of the events, the Case ID has been determined, representing 453 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 98.70% of the events, the Case ID has been determined, representing 8,390 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.17% of the events, the Case ID has been determined, representing 163 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 100.00% of the events, the Case ID has been determined, representing 9,644 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.14% of the events, the Case ID has been determined, representing 135 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 98.81% of the events, the Case ID has been determined, representing 8,496 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.18% of the events, the Case ID has been determined, representing 173 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 100.00% of the events, the Case ID has been determined, representing 9,644 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.17% of the events, the Case ID has been determined, representing 163 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 99.22% of the events, the Case ID has been determined, representing 8,891 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.52% of the events, the Case ID has been determined, representing 501 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 99.07% of the events, the Case ID has been determined, representing 8,747 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.47% of the events, the Case ID has been determined, representing 453 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 100.00% of the events, the Case ID has been determined, representing 9,644 more events "
                          "than initially recorded." + "\033[0m")
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
        display(df)
        print("\n... (+ 96430 more rows)\n")
        print("\033[1m" + "For 90.51% of the events, the Case ID has been determined, representing 491 more events "
                          "than initially recorded." + "\033[0m")

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

    attributes = input("Please enter the input attribute(s) for the transformer (separated by commas): "
                       "Activity, Timestamp, ").strip().lower()

    if attributes and attributes != 'resource':
        raise ValueError("Invalid input.")

    if not attributes:
        input_attributes = ['Activity', 'Timestamp']
    else:
        input_attributes = ['Activity', 'Timestamp', 'Resource']
