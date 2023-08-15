import pickle
from IPython import embed

def main():
    data_path = 'int2_dataset/interaction_scenario/complete_scenario/8/051412344805-051412355205.pickle'
    with open(data_path, 'rb+') as f:
        data = pickle.load(f)
    embed()

if __name__ == "__main__":
    main()