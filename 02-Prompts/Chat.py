from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

result = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
) # This is Zero-shot prompting

print(result.choices[0].message.content)