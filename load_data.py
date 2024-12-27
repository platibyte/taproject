from datasets import load_dataset

def get():
    ds = load_dataset("MuskumPillerum/General-Knowledge")
    return ds

if __name__ == '__main__':
    data = get()
    data.save_to_disk('General-Knowledge')