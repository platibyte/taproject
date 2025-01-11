import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from functools import partial
import pandas as pd

# Daten laden
df = pd.read_parquet('dataset_complete.parquet')
input_texts= df['Question'].tolist() + df['Answer'].tolist()
GPT_input_texts = df['Question'].tolist() + df['Response'].tolist()

# Average-Pooling-Funktion von der Huggingface-Quelle
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Tokenizer und model von der Huggingface-Quelle
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
model = AutoModel.from_pretrained('intfloat/e5-small-v2')

# Vorbereiten des tokenizers für pooling
tokenizer_kwargs = {
    'max_length': 80,
    'padding': 'max_length',
    'truncation': True,
    'return_tensors': 'pt'
}
tokenizer_with_args = partial(tokenizer, **tokenizer_kwargs)

# Tokenize the input texts
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ProcessPoolExecutor zum umgehen des Global Interpreter Lock
def pooling(section):
    results = []
    with ProcessPoolExecutor() as executor:
        for i, tokens in enumerate(executor.map(
            tokenizer_with_args, section
            )):
            results.append(tokens)
    # Tensoren müssen wieder kombiniert werden mit torch.cat
    combined_batch_dict = {
        key: torch.cat([torch.tensor(batch[key]) for batch in results], dim=0)
        for key in results[0]
    }

    return combined_batch_dict

batch_dict = pooling(input_texts)

# batch_dict auf gpu verschieben
for key in batch_dict:
    batch_dict[key] = batch_dict[key].to(device)

outputs = model(**batch_dict)

embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])


# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
