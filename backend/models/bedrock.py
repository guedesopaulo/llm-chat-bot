# backend/models/bedrock.py

import os
from dotenv import load_dotenv
from langchain_community.llms import Bedrock

def load_bedrock_chat():
    load_dotenv()  # Carrega as credenciais do .env

    model_id = "amazon.titan-text-express-v1"
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    # Aqui usamos o LangChain LLM Bedrock wrapper
    model = Bedrock(
        model_id=model_id,
        region_name=region,
        credentials_profile_name=None  # use None se .env está correto
    )

    # Simulando o comportamento de "processor" como no Gemma
    def processor(messages):
        """Montador de prompt estilo chat para Bedrock"""
        prompt = ""
        for m in messages:
            role = m["role"].capitalize()
            content = m["content"]
            if isinstance(content, list):  # para compatibilidade com multimodal
                text_content = [c["text"] for c in content if c["type"] == "text"]
                prompt += f"{role}: {' '.join(text_content)}\n"
            else:
                prompt += f"{role}: {content}\n"
        prompt += "Assistant: "
        return prompt

    return processor, model
