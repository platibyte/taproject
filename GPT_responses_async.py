import openai
import asyncio
import pandas as pd
import logging


async def call_openai(id: int , prompt: str):
    """Sendet eine Anfrage an die OpenAI API und gibt die Antwort zurück."""
    try:
        logging.info(f'Sending prompt {id}: {prompt}')
        response = await openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return id, response.choices[0].message.content
    except Exception as e:
        logging.error(f'Error for prompt {id}: {e}')
        return id, f"Fehler: {e}"

async def main(prompts: list) -> list:
    """Erstellt eine Liste von Tasks und führt sie parallel aus."""
    total_tasks = len(prompts)
    completed_task = 0

    # wrap tasks for progress tracking
    async def wrapped_task(task_id, prompt):
        nonlocal completed_task
        id, result = await call_openai(task_id, prompt)
        completed_task += 1
        print(f'Progress: {completed_task}/{total_tasks} prompts completed')
        return result

    tasks = [wrapped_task(i, prompt) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == '__main__':
    # Lade die Fragen
    df = pd.read_parquet('dataset.parquet')
    df = df[:10]
    questions = df['Question'].to_list()
    print(f'Starte {len(questions)} prompts.')
    # Parallele Abfrage von GPT
    responses = asyncio.run(main(questions))

    # Speichern als txt und parquet
    with open('responses.txt', 'w') as file:
        [file.writelines(f'{i}§ {line}\n') for i,line in enumerate(responses)]
    
    try:
        df['Response'] = responses
    except Exception as e:
        print(f'Fehler beim Zusammenführen der Antworten mit dem DataFrame: {e}')
    
    df.to_parquet('GPT_responses.parquet')