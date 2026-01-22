import os
import sys
from ruamel.yaml import YAML

def update_nested_key(data, key_path, new_value):
    keys = key_path.split('.')
    current = data
    for key in keys[:-1]:
        if key in current:
            current = current[key]
        else:
            return False
    if keys[-1] in current:
        current[keys[-1]] = new_value
        return True
    else:
        return False

def update_yaml_files(yaml_file, key_path, value):
    yaml = YAML()
    yaml.preserve_quotes = True  # Optional: preserves quotes if needed

    file_path = os.path.join(os.getcwd(), yaml_file)

    if not os.path.isfile(file_path):
        print(f"Error: File '{yaml_file}' not found.")
        return

    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.load(file)
        except Exception as e:
            print(f"Error reading '{yaml_file}': {e}")
            return

    if update_nested_key(yaml_data, key_path, value):
        with open(file_path, 'w') as file:
            try:
                yaml.dump(yaml_data, file)
                print(f"Updated '{yaml_file}' successfully.")
            except Exception as e:
                print(f"Error writing to '{yaml_file}': {e}")
    else:
        print(f"Warning: Key path '{key_path}' not found in '{yaml_file}'")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python update_yaml.py <yaml_file> <key_path> <type> <value>")
        sys.exit(1)

    yaml_file = sys.argv[1]
    key_path_to_set = sys.argv[2]
    type_of_value = sys.argv[3]

    if type_of_value == 'str':
        value_to_set = str(sys.argv[4])
    elif type_of_value == 'int':
        value_to_set = int(sys.argv[4])
    elif type_of_value == 'float':
        value_to_set = float(sys.argv[4])
    else:
        print(f"Unsupported type: {type_of_value}")
        sys.exit(1)

    update_yaml_files(yaml_file, key_path_to_set, value_to_set)
