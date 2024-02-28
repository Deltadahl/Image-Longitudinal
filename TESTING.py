import json

def print_json_keys(file_path):
    # Open the JSON file
    with open(file_path, 'r') as file:
        # Load the data from the file
        data = json.load(file)

        # Check if the data is a dictionary and print its keys
        if isinstance(data, dict):
            print("Keys in the JSON file:")
            for key in data.keys():
                print(key)
        else:
            print("The JSON file does not contain a dictionary.")

# Usage
print_json_keys("data/synthetic/json/noise_0.0_eta_approx_and_lv_data_100k.json")

