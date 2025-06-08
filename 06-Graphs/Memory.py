from mem0 import Memory
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

qdrant_host ='localhost'  
neo4j_url="bolt://localhost:7687"
neo4j_name="neo4j"
neo4j_password="reform-william-center-vibrate-press-5829"

config={
    'version': 'v1.1',
    'custom_fact_extraction_prompt': None,
    'custom_update_memory_prompt': None,
    'embedder':{
        'provider':'openai',
    'config': {
        'api_key': openai_api_key,
        "model": 'text-embedding-3-small'
    }
    },
    'llm': {
        "provider": "openai",
        'config':{
            'api_key': openai_api_key,
            'model': 'gpt-3.5-turbo',
            
        }
    },
    'vector_store': {
        'provider': 'qdrant',
        'config': {
            'host': qdrant_host,
            'port': 6333,
    }
    },
    'graph_store': {
        'provider': 'neo4j',
        'config': {
            'url': neo4j_url,
            'username': neo4j_name,
            'password': neo4j_password,
        }
    }
}

mem_client = Memory.from_config(config)

client = OpenAI(
    api_key=openai_api_key,
    
)



def chat(message):
    mem_result = mem_client.search(query=message, user_id="l123")
    
    memories = "\n".join([m["memory"] for m in mem_result.get("results")])
    
    system_prompt = f'''
        You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
        systematically analyze input content, extract structured knowledge, and maintain an
        optimized memory store. Your primary function is information distillation
        and knowledge preservation with contextual awareness.

        Tone: Professional analytical, precision-focused, with clear uncertainty signaling
        
        Memory and Score:
        {memories}
    '''
    
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": message
        }
    ]
    
    result=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    
    messages.append(
        {
            "role": "assistant",
            "content": result.choices[0].message.content
        }
    )
    
    mem_client.add(messages,user_id="l123")
    
    return result.choices[0].message.content

while True:
    message = input("You: ")
    if message.strip().lower() == "exit":
        print("Goodbye!")
        break
    chat_response = chat(message)
    print(f"🤖: {chat_response}")