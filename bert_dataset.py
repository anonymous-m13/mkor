from data_loader.data_loader import get_tokenizer
from data_loader.wikicorpus import get_wikicorpus_dataset


# num_workers = multiprocessing.cpu_count()
num_workers = 1
root = "./data"

dataset = "bert"



for size in ["base"]:
    for case in ["cased"]:
        model_name = f"bert_{size}_{case}"
        tokenizer = get_tokenizer(model_name)
        get_wikicorpus_dataset(root, model_name, "wikicorpus_phase2", tokenizer, num_workers, False)
