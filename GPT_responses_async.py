import openai
import asyncio
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def call_openai_with_retries(prompt: str, task_id: int, retries: int = 5, delay: float = 1.0):
    """Sendet eine Anfrage an die OpenAI API mit Retries bei Fehlern."""
    for attempt in range(retries):
        try:
            logging.info(f"Task {task_id}: Sending prompt (Attempt {attempt + 1}).")
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
            )
            logging.info(f"Task {task_id}: Prompt completed.")
            return response.choices[0].message.content
        except openai.error.RateLimitError as e:
            logging.warning(f"Task {task_id}: Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponentielles Backoff
        except Exception as e:
            logging.error(f"Task {task_id}: Error - {e}")
            return f"Fehler: {e}"
    return f"Fehler: Too many retries for task {task_id}"

async def main(prompts: list):
    """Erstellt eine Liste von Tasks und führt sie parallel aus."""
    total_tasks = len(prompts)
    completed_tasks = 0

    async def wrapped_task(task_id, prompt):
        nonlocal completed_tasks
        async with asyncio.Semaphore(5):
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, call_openai_with_retries, prompt, task_id)
        completed_tasks += 1
        logging.info(f"Progress: {completed_tasks}/{total_tasks} tasks completed.")
        return result

    tasks = [wrapped_task(i, prompt) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return results

if __name__ == '__main__':
    try:
        # Lade die Fragen
        df = pd.read_parquet('dataset.parquet')

        if 'Question' not in df.columns:
            raise ValueError("Die Spalte 'Question' fehlt im Dataset.")
        
        # df = df[:10]  # Begrenze die Verarbeitung auf 10 Fragen
        questions = df['Question'].to_list()
        logging.info(f"Starte {len(questions)} Prompts.")

        # Parallele Abfrage von GPT
        responses = asyncio.run(main(questions))

        # Speichern als txt und parquet
        with open('responses.txt', 'w') as file:
            [file.writelines(f'{i}§ {line}\n') for i, line in enumerate(responses)]

        df['Response'] = responses
        df.to_parquet('GPT_responses.parquet')
        logging.info("Antworten wurden erfolgreich gespeichert.")
    except Exception as e:
        logging.error(f"Fehler: {e}")
