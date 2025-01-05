import openai
import asyncio
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
import time
from csv import writer, QUOTE_ALL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def call_openai_with_retries(prompt: str, task_id: int, retries: int = 5, delay: float = 1.0):
    """Sendet eine Anfrage an die OpenAI API mit Retries bei Fehlern."""
    for attempt in range(retries):
        try:
            logging.info(f"Task {task_id}: Sending prompt (Attempt {attempt + 1}).")
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
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
    # Initialisiere Zähler
    total_tasks = len(prompts)
    completed_tasks = 0

    async def wrapped_task(task_id, prompt):
        """Startet einen Prompt asynchron und mit Threading."""
        # Variable aus übergeordneter Ebene übernehmen und bearbeiten
        nonlocal completed_tasks

        # Begrenzen auf 5 gleichzeitige Abfragen
        async with asyncio.Semaphore(2):
            # Asynchrones ausführen von Funktionen
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, call_openai_with_retries, prompt, task_id)
                save_result(task_id, result)

        completed_tasks += 1
        
        logging.info(f"Progress: {completed_tasks}/{total_tasks} tasks completed.")
        return result

    tasks = [wrapped_task(i, prompt) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return results


def load_list() -> list:
    # Lade die Fragen
    df = pd.read_parquet('dataset.parquet')

    #df = df[:10]  # Begrenze die Verarbeitung auf 10 Fragen
    questions = df['Question'].to_list()
    logging.info(f"Starte {len(questions)} Prompts.")
    
    return questions, df


def save_result(i, result):
    with open('GPTresponses.csv', 'a') as file:
        writer_object = writer(file, quoting=QUOTE_ALL, delimiter=";")
        writer_object.writerow([i, result])
        file.close()


if __name__ == '__main__':
    try:
        # Parallele Abfrage von GPT
        prompts, df = load_list()
        responses = asyncio.run(main(prompts))

        df['Response'] = responses
        df.to_parquet('dataset_completed.parquet')

        logging.info("Antworten wurden erfolgreich gespeichert.")

    except Exception as e:
        logging.error(f"Fehler: {e}")
