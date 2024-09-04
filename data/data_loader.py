from beir.datasets.data_loader import GenericDataLoader

def load_data(data_path, split):
    return GenericDataLoader(data_folder=data_path).load(split=split)
