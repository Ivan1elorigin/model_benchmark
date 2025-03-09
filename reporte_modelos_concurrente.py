import requests
import pandas as pd

import asyncio
from ollama import AsyncClient

df = pd.read_csv('labeled_data.csv')

#df = df.sample(n=10, random_state=42).copy()  # random_state para reproducibilidad

print(df.head())

model_predictions = {
                    "deepseek-r1:1.5b" : [],
                "deepseek-r1:7b" : [],
                "deepseek-r1:8b" : []

}

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
    
    for i in range(0, len(df), 30):

        batch_tweets = df['tweet'].iloc[i:i+30]

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
        
                model_predictions[model].append(enviar_prompt(message, model))
        
        for model in models:
        # Sustituimos el array de llamadas por un array de resultados para cada array de peticiones a modelos. Ya que la operación devuelve un array respuestas
            model_predictions[model] = await asyncio.gather(*model_predictions[model])
        
        for model in models:
            model_predictions[model] = [deleteThink(text) for text in model_predictions[model]]
    


    

# await main()

if __name__ == '__main__':
    asyncio.run(main())

dfModelPredictions = pd.DataFrame(model_predictions)

finalDataFrame = pd.concat([df.reset_index(drop=True), dfModelPredictions.reset_index(drop=True)], axis=1) # concat 0 se unen ampliando las filas, 1 no

finalDataFrame.to_csv("modelEvaluation.csv", index=False)
