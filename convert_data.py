import datasets
import pandas as pd

data = datasets.load_from_disk('General-Knowledge/train')

df = pd.DataFrame(data)

df.to_parquet('dataset.parquet')