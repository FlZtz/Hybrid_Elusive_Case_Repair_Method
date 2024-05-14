# dummy_transformer.py - A dummy script to demonstrate the interaction between the user and the system.
from typing import List

added_expert_knowledge: bool
expert_attribute: str = ''
expert_knowledge: bool
first_rule_checking: bool
input_attributes: List[str]
new_expert_attribute: str = ''
remaining_attributes: List[str]
second_rule_checking: bool
second_threshold_value: float
third_rule_checking: bool


def add_expert_input() -> None:
    """
    Add expert input.
    """
    global added_expert_knowledge, expert_attribute, remaining_attributes

    choice = input("Do you want to add one or more expert attributes? (yes/no): ").strip().lower()

    if not choice or choice not in ['yes', 'no']:
        raise ValueError("Invalid choice.")

    added_expert_knowledge = choice == 'yes'

    print("\033[3m" + f"Expert attributes: {'Yes' if added_expert_knowledge else 'No'}" + "\033[0m")

    if added_expert_knowledge:
        remaining_attributes = [attr for attr in ['Start Activity', 'End Activity', 'Directly Following'] if
                                attr != expert_attribute]

        attributes = ', '.join(remaining_attributes[:-1]) + ' and ' + remaining_attributes[-1]

        attribute = input("Following expert attributes are available: " + attributes + ".\n" +
                          "Please enter the attribute(s) for which you have expert knowledge "
                          "(separated by commas): ").strip().lower()

        remaining_attributes = [attr.lower() for attr in remaining_attributes]

        get_attribute(attribute, False)


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


def first_rule_check() -> None:
    """
    Perform the first rule check.
    """
    global expert_knowledge, first_rule_checking

    if not expert_knowledge:
        print("\033[3m" + "No expert attributes provided." + "\033[0m")
    else:
        print("Configuration of log repair\n"
              "Do you want to predict the Case ID values using the event log that was used for training? (yes/no): " +
              "\033[1m" + "no\n" + "\033[0m" +
              "Please ensure the new event log and its name match the process used during training.\n"
              "Enter the path to the file that contains the event log: " + "\033[1m" +
              "logs/elusive/Hospital Billing - Event Log.xes\n" + "\033[0m" +
              "XES file successfully read.\n" +
              "\033[3m" + "Data Loading" + "\033[0m")

        first_rule_checking = rule_check()

    print("Please enter the minimum probability (in %) with which the Case ID must be determined in order for it to be "
          "accepted: " + "\033[1m" + "25" + "\033[0m")

    if expert_knowledge:
        suffix = 'yes' if first_rule_checking else 'no'

        print("Please note that incorporating declarative rule checking may result in assumption-based modifications.\n"
              "Do you want to proceed with ex post rule checking in this iteration? (yes/no): " +
              "\033[1m" + suffix + "\033[0m")

    print("For 10.00% of the events, the Case ID has originally not been recorded.")

    show_first_output()


def get_attribute(attribute: str, first: bool = True) -> None:
    """
    Get the expert attribute values.

    :param attribute: User input for the expert attribute.
    :param first: Flag indicating whether it is the first function call. Default is True.
    """
    global expert_attribute, new_expert_attribute, remaining_attributes

    if first:
        if not attribute or attribute not in ['start activity', 'end activity']:
            raise ValueError("Invalid input.")
    else:
        if not attribute or attribute not in remaining_attributes:
            raise ValueError("Invalid input.")

    if attribute == 'start activity':
        storage = 'Start Activity'
    else:
        storage = 'End Activity'

    print("\033[3m" + f"Expert attribute: {storage}" + "\033[0m")

    if storage == 'Start Activity':
        print("Please enter the value(s) that represent(s) the attribute 'Start Activity' (separated by commas) –\n"
              "Suggestions (proportion of cases with corresponding Activity as the Start Activity): NEW (100.00%):\n" +
              "\033[1m" + "NEW" + "\033[0m")
        print("Does 'NEW' always or sometimes represent the attribute 'Start Activity'?\n"
              "Enter 'always' or 'sometimes': " + "\033[1m" + "always" + "\033[0m")
    else:
        print("Please enter the value(s) that represent(s) the attribute 'End Activity' (separated by commas) –\n"
              "Suggestions (proportion of cases with corresponding Activity as the End Activity): "
              "BILLED (63.50%), NEW (22.41%), DELETE (8.21%):\n" +
              "\033[1m" + "BILLED, NEW, DELETE" + "\033[0m")
        print("Does 'BILLED' always or sometimes represent the attribute 'End Activity'?\n"
              "Enter 'always' or 'sometimes': " + "\033[1m" + "sometimes" + "\033[0m")
        print("Does 'NEW' always or sometimes represent the attribute 'End Activity'?\n"
              "Enter 'always' or 'sometimes': " + "\033[1m" + "sometimes" + "\033[0m")
        print("Does 'DELETE' always or sometimes represent the attribute 'End Activity'?\n"
              "Enter 'always' or 'sometimes': " + "\033[1m" + "sometimes" + "\033[0m")

    if first:
        expert_attribute = storage
    else:
        new_expert_attribute = storage


def provide_expert_input() -> None:
    """
    Provide expert input.
    """
    global expert_knowledge

    if not expert_knowledge:
        print("\033[3m" + "No expert attributes provided." + "\033[0m")
        return

    attribute = input("Following expert attributes are available: Start Activity, End Activity and Directly "
                      "Following.\n"
                      "Please enter the attribute(s) for which you have expert knowledge "
                      "(separated by commas): ").strip().lower()

    get_attribute(attribute)


def rule_check() -> bool:
    """
    Prompt whether to perform rule checking.

    :return: User choice for rule checking.
    """
    choice = input("Please note that incorporating declarative rule checking may result in assumption-based "
                   "modifications.\n"
                   "Do you want to proceed with ex ante rule checking in this iteration? (yes/no): ").strip().lower()

    if not choice or choice not in ['yes', 'no']:
        raise ValueError("Invalid choice.")

    decision = choice == 'yes'

    print("\033[3m" + f"Rule checking: {'Yes' if decision else 'No'}" + "\033[0m")

    return decision


def second_rule_check() -> None:
    """
    Perform the second rule check.
    """
    global expert_knowledge, second_rule_checking

    if not expert_knowledge:
        print("\033[3m" + "No expert attributes provided." + "\033[0m")
    else:
        second_rule_checking = rule_check()


def second_threshold() -> None:
    """
    Set the second threshold value.
    """
    global expert_knowledge, second_rule_checking, second_threshold_value

    value = input("Please enter the minimum probability (in %) with which the Case ID must be determined in order for "
                  "it to be accepted: ").strip()

    if not value.isdigit():
        raise ValueError("Invalid input.")

    value = float(value)

    if value != 0 and value != 50:
        raise ValueError("Invalid input.")

    second_threshold_value = value

    print("\033[3m" + f"Threshold value: {second_threshold_value}" + "\033[0m")

    if expert_knowledge:
        suffix = 'yes' if second_rule_checking else 'no'

        print("Please note that incorporating declarative rule checking may result in assumption-based modifications.\n"
              "Do you want to proceed with ex post rule checking in this iteration? (yes/no): " +
              "\033[1m" + suffix + "\033[0m")

    show_second_output()


def show_first_output() -> None:
    """
    Show the first output.
    """
    print("\033[3m" + "Output" + "\033[0m")
    print("Do you want to use the repaired log as the baseline for an additional repair? (yes/no): " +
          "\033[1m" + "yes\n" + "\033[0m" +
          "Do you want to keep the probability threshold for the next repair? (yes/no): " +
          "\033[1m" + "no\n" + "\033[0m" +
          "Do you want to add one or more expert attributes? (yes/no): " +
          "\033[1m" + "no" + "\033[0m")


def show_second_output() -> None:
    """
    Show the second output.
    """
    print("\033[3m" + "Output" + "\033[0m")
    print("Do you want to use the repaired log as the baseline for an additional repair? (yes/no): " +
          "\033[1m" + "yes\n" + "\033[0m" +
          "Do you want to keep the probability threshold for the next repair? (yes/no): " +
          "\033[1m" + "yes" + "\033[0m")


def show_third_output() -> None:
    """
    Show the third output.
    """
    print("\033[3m" + "Output" + "\033[0m")
    print("Do you want to use the repaired log as the baseline for an additional repair? (yes/no): " +
          "\033[1m" + "no" + "\033[0m")


def third_rule_check() -> None:
    """
    Perform the third rule check.
    """
    global added_expert_knowledge, expert_knowledge, third_rule_checking

    if not expert_knowledge and not added_expert_knowledge:
        print("\033[3m" + "No expert attributes provided." + "\033[0m")
    else:
        third_rule_checking = rule_check()
        suffix = 'yes' if third_rule_checking else 'no'
        print("Please note that incorporating declarative rule checking may result in assumption-based modifications.\n"
              "Do you want to proceed with ex post rule checking in this iteration? (yes/no): " +
              "\033[1m" + suffix + "\033[0m")

    show_third_output()


def transformer_input() -> None:
    """
    Get the input attributes for the transformer.
    """
    global input_attributes

    print("Do you want to use a specific response configuration file for model training? (yes/no): " + "\033[1m" +
          "no\n\n" + "\033[0m" +
          "Configuration of model training\n"
          "Enter the path to the file that contains the event log: " + "\033[1m" +
          "logs/Hospital Billing - Event Log.xes\n" + "\033[0m" +
          "XES file successfully read.")

    attributes = input("Please enter the input attribute(s) for the transformer (separated by commas): " + "\033[1m" +
                       "Activity, Timestamp, " + "\033[0m").strip().lower()

    if attributes and attributes != 'resource':
        raise ValueError("Invalid input.")

    if not attributes:
        input_attributes = ['Activity', 'Timestamp']
    else:
        input_attributes = ['Activity', 'Timestamp', 'Resource']

    print("\033[3m" + f"Input attributes for the transformer: {', '.join(input_attributes)}" + "\033[0m")
