import pickle

# Function to load data from pickle file
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print("The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == '__main__':
    # Path to the pickle file
    file_path = '/work/b0990106x/TextRL/replay_buffer/replay_buffer_update_0.pkl'

    # Load data from pickle file
    data = load_pickle(file_path)
    print("length of pickle data: ", len(data))
    print("type of pickle data: ", type(data))
    print("pickle data: ", data)    

