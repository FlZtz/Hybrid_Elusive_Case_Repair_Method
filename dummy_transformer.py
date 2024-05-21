# dummy_transformer.py - A dummy script to demonstrate the interaction between the user and the system.
from typing import List

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
    model = get_model()

    if model == 1:
        print("\033[3m" + "Output 1" + "\033[0m")
    elif model == 2:
        print("\033[3m" + "Output 2" + "\033[0m")
    elif model == 3:
        print("\033[3m" + "Output 3" + "\033[0m")
    elif model == 4:
        print("\033[3m" + "Output 4" + "\033[0m")
    elif model == 5:
        print("\033[3m" + "Output 5" + "\033[0m")
    else:
        print("\033[3m" + "Output 6" + "\033[0m")

    print('-' * 80)


def show_second_output() -> None:
    """
    Show the second output.
    """
    outcome = get_outcome()

    if outcome == 1:
        print("\033[3m" + "Output 1" + "\033[0m")
    elif outcome == 2:
        print("\033[3m" + "Output 2" + "\033[0m")
    elif outcome == 3:
        print("\033[3m" + "Output 3" + "\033[0m")
    elif outcome == 4:
        print("\033[3m" + "Output 4" + "\033[0m")
    elif outcome == 5:
        print("\033[3m" + "Output 5" + "\033[0m")
    elif outcome == 6:
        print("\033[3m" + "Output 6" + "\033[0m")
    elif outcome == 7:
        print("\033[3m" + "Output 7" + "\033[0m")
    elif outcome == 8:
        print("\033[3m" + "Output 8" + "\033[0m")
    elif outcome == 9:
        print("\033[3m" + "Output 9" + "\033[0m")
    elif outcome == 10:
        print("\033[3m" + "Output 10" + "\033[0m")
    elif outcome == 11:
        print("\033[3m" + "Output 11" + "\033[0m")
    elif outcome == 12:
        print("\033[3m" + "Output 12" + "\033[0m")
    elif outcome == 13:
        print("\033[3m" + "Output 13" + "\033[0m")
    elif outcome == 14:
        print("\033[3m" + "Output 14" + "\033[0m")
    elif outcome == 15:
        print("\033[3m" + "Output 15" + "\033[0m")
    elif outcome == 16:
        print("\033[3m" + "Output 16" + "\033[0m")
    elif outcome == 17:
        print("\033[3m" + "Output 17" + "\033[0m")
    elif outcome == 18:
        print("\033[3m" + "Output 18" + "\033[0m")
    elif outcome == 19:
        print("\033[3m" + "Output 19" + "\033[0m")
    elif outcome == 20:
        print("\033[3m" + "Output 20" + "\033[0m")
    elif outcome == 21:
        print("\033[3m" + "Output 21" + "\033[0m")
    elif outcome == 22:
        print("\033[3m" + "Output 22" + "\033[0m")
    elif outcome == 23:
        print("\033[3m" + "Output 23" + "\033[0m")
    elif outcome == 24:
        print("\033[3m" + "Output 24" + "\033[0m")
    elif outcome == 25:
        print("\033[3m" + "Output 25" + "\033[0m")
    elif outcome == 26:
        print("\033[3m" + "Output 26" + "\033[0m")
    elif outcome == 27:
        print("\033[3m" + "Output 27" + "\033[0m")
    elif outcome == 28:
        print("\033[3m" + "Output 28" + "\033[0m")
    elif outcome == 29:
        print("\033[3m" + "Output 29" + "\033[0m")
    elif outcome == 30:
        print("\033[3m" + "Output 30" + "\033[0m")
    elif outcome == 31:
        print("\033[3m" + "Output 31" + "\033[0m")
    elif outcome == 32:
        print("\033[3m" + "Output 32" + "\033[0m")
    elif outcome == 33:
        print("\033[3m" + "Output 33" + "\033[0m")
    elif outcome == 34:
        print("\033[3m" + "Output 34" + "\033[0m")
    elif outcome == 35:
        print("\033[3m" + "Output 35" + "\033[0m")
    elif outcome == 36:
        print("\033[3m" + "Output 36" + "\033[0m")
    elif outcome == 37:
        print("\033[3m" + "Output 37" + "\033[0m")
    elif outcome == 38:
        print("\033[3m" + "Output 38" + "\033[0m")
    elif outcome == 39:
        print("\033[3m" + "Output 39" + "\033[0m")
    elif outcome == 40:
        print("\033[3m" + "Output 40" + "\033[0m")
    elif outcome == 41:
        print("\033[3m" + "Output 41" + "\033[0m")
    elif outcome == 42:
        print("\033[3m" + "Output 42" + "\033[0m")
    elif outcome == 43:
        print("\033[3m" + "Output 43" + "\033[0m")
    else:
        print("\033[3m" + "Output 44" + "\033[0m")

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
