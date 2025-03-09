import requests
import pandas as pd

import asyncio
from ollama import AsyncClient
import os

df = pd.read_csv('labeled_data.csv')

df = df.sample(n=45, random_state=42).copy()  # random_state para reproducibilidad

print(df.head())



models = [ #buscar como se llaman cada uno de los modelos.
            "deepseek-r1:1.5b",
            "deepseek-r1:7b",
            "deepseek-r1:8b"
        ]

import re
import json

#pasamos prompt y modelo
async def enviar_prompt(prompt, auxModel):
    client = AsyncClient()
    message = {'role': 'user', 'content': prompt}
    response = await client.chat(model=auxModel, messages=[message])
    return response['message']['content']

def deleteThink(texto):
    """
    Elimina las etiquetas <think> y el contenido dentro de ellas de la cadena de texto.
    
    Args:
        texto (str): Texto de entrada.
    
    Returns:
        str: Texto sin las etiquetas <think> y su contenido.
    """
    patron = r"<think>.*?</think>"
    texto_limpio = re.sub(patron, "", texto, flags=re.DOTALL)
    return texto_limpio.strip()

async def main():

    if os.path.exists("modelEvaluation.csv"):
        os.remove("modelEvaluation.csv")
    
    for i in range(0, len(df), 30):

        batch_tweets = df['tweet'].iloc[i:i+30]

        model_predictions = {
                            "deepseek-r1:1.5b" : [],
                        "deepseek-r1:7b" : [],
                        "deepseek-r1:8b" : []

        }

        promises = []
        for tweet in batch_tweets:

        
                        #     "llama3.3",
                        # "llama3.2",
                        # "llama3.1",
                        # "phi4",
                        # "phi3",
                        # "qwen2",
                        # "qwen2.5"
        
            # message = f"""
            # Clasify these messages in the following categories:
            # 0 - hate speech 1 - offensive language 2 - neither
            # IMPORTANT!!!: Put only the number to express the category.
        
            # {tweet}
            # """

            message = f"""
            Classify the following message into one of these categories:
            0 - Hate Speech
            1 - Offensive Language
            2 - Neither
            
            IMPORTANT:
            - Respond **only** with a single number (0, 1, or 2).
            - Do **not** include any explanations or additional text.
            - If the message is unclear, choose the closest matching category.
        
            Tweet: {tweet}
            """
        
            for model in models:
                # Procesar las líneas JSON y extraer texto
                promises.append((model, enviar_prompt(message, model)))

        results = await asyncio.gather(*(promise[1] for promise in promises))
         
        
        for idx, (model, _) in enumerate(promises):
            if model_predictions[model] is None:
                model_predictions[model] = []
            model_predictions[model].append(deleteThink(results[idx]))

        dfModelPredictions = pd.DataFrame(model_predictions)

        finalDataFrame = pd.concat([df.iloc[i:i+30].reset_index(drop=True), dfModelPredictions.reset_index(drop=True)], axis=1) # concat 0 se unen ampliando las filas, 1 no

        print(finalDataFrame)

        # Verificar si el archivo ya existe para manejar el encabezado
        file_exists = os.path.isfile("modelEvaluation.csv")

        finalDataFrame.to_csv("modelEvaluation.csv", mode='a', header=not file_exists, index=False, encoding='utf-8')
    


    

# await main()

if __name__ == '__main__':
    asyncio.run(main())


