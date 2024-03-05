import json
import glob
import os

# The path to the state.json file
json_file_path = 'data-encodec/state.json'

# The directory containing the arrow files
data_encodec_path = 'data-encodec/*.arrow'

def reset_data_files(json_file_path):
    # Open the JSON file for reading
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Reset the "_data_files" to an empty list
    data['_data_files'] = []
    
    # Open the JSON file for writing
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

def clear_arrow_files(data_encodec_path):
    # Find all arrow files in the directory
    arrow_files = glob.glob(data_encodec_path)
    
    # Iterate over the list of filepaths & remove each file
    for file_path in arrow_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error while deleting file {file_path}: {e}")

if __name__ == "__main__":
    reset_data_files(json_file_path)
    clear_arrow_files(data_encodec_path)
    print("The '_data_files' have been reset to empty and all arrow files have been deleted.")
