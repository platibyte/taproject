from datasets import load_dataset

def get():
    ds = load_dataset("MuskumPillerum/General-Knowledge")
    return ds