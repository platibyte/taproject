import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from functools import partial
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Daten laden
df = pd.read_parquet('dataset_complete.parquet')
questions = df['Question'].tolist()
answers = df['Answer'].tolist()
responses = df['Response'].tolist()  # GPT-Antworten

# Average-Pooling-Funktion von der Huggingface-Quelle
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Tokenizer und Modell laden
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
model = AutoModel.from_pretrained('intfloat/e5-small-v2')

# Modell auf die GPU verschieben, falls verfügbar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenizer-Argumente vorbereiten
tokenizer_kwargs = {
    'max_length': 80,
    'padding': 'max_length',
    'truncation': True,
    'return_tensors': 'pt'
}
tokenizer_with_args = partial(tokenizer, **tokenizer_kwargs)

# Pooling-Funktion mit Multiprocessing
def tokenize_and_embed(texts):
    with ProcessPoolExecutor() as executor:
        tokenized_batches = list(executor.map(tokenizer_with_args, texts))

    # Tensoren kombinieren
    combined_batch_dict = {
        key: torch.cat([batch[key] for batch in tokenized_batches], dim=0)
        for key in tokenized_batches[0]
    }
    
    # Tokenisierte Texte auf die GPU verschieben
    for key in combined_batch_dict:
        combined_batch_dict[key] = combined_batch_dict[key].to(device)

    # Embeddings berechnen
    with torch.no_grad():
        outputs = model(**combined_batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, combined_batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalisierung

    return embeddings

# Embeddings für Fragen, Antworten und GPT-Antworten berechnen
question_embeddings = tokenize_and_embed(questions)
answer_embeddings = tokenize_and_embed(answers)
gpt_embeddings = tokenize_and_embed(responses)

# Ranking und Evaluation
def evaluate_ranking(question_embeddings, answer_embeddings, correct_indices):
    scores = torch.matmul(question_embeddings, answer_embeddings.T)  # Cosinus-Ähnlichkeiten

    ranks = []
    for i, correct_idx in enumerate(correct_indices):
        ranked_indices = torch.argsort(scores[i], descending=True)  # Antworten sortieren
        rank = (ranked_indices == correct_idx).nonzero(as_tuple=True)[0].item() + 1  # Rang ermitteln
        ranks.append(rank)

    return ranks

# Korrekte Antwort-Indizes
correct_indices = list(range(len(answers)))  # Annahme: Antworten sind in der richtigen Reihenfolge

# Berechnung der Ränge
ranks = evaluate_ranking(question_embeddings, answer_embeddings, correct_indices)

# Evaluation: Mean Rank, MRR und Precision@k
mean_rank = sum(ranks) / len(ranks)
mrr = sum(1.0 / rank for rank in ranks) / len(ranks)
precision_at_1 = sum(1 for rank in ranks if rank == 1) / len(ranks)

print(f"Mean Rank: {mean_rank}")
print(f"MRR: {mrr}")
print(f"Precision@1: {precision_at_1}")

# GPT-Antworten vergleichen (optional)
gpt_scores = torch.matmul(gpt_embeddings, answer_embeddings.T)
gpt_ranks = evaluate_ranking(gpt_embeddings, answer_embeddings, correct_indices)
mean_rank_gpt = sum(gpt_ranks) / len(gpt_ranks)
mrr_gpt = sum(1.0 / rank for rank in gpt_ranks) / len(gpt_ranks)
precision_at_1_gpt = sum(1 for rank in gpt_ranks if rank == 1) / len(gpt_ranks)

print(f"GPT Mean Rank: {mean_rank_gpt}")
print(f"GPT MRR: {mrr_gpt}")
print(f"GPT Precision@1: {precision_at_1_gpt}")
