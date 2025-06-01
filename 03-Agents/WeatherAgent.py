import json
from dotenv import load_dotenv
from openai import OpenAI
import requests

load_dotenv()

client = OpenAI()

def get_weather(city:str):
    print(f'🔨 Tool Called: get_weather: ', city)
    url=f'https://wttr.in/{city}?format=%C+%t'
    response = requests.get(url)
    
    if response.status_code == 200:
        weather_data = response.text.strip()
        return weather_data
    return "Unable to fetch weather data"

available_tools ={
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name and returns the current weather in that city.",
    }
}

system_prompt = f'''
You are a helpful AI assistant who is specialized in resolving user queries.
You work on start, plan action and observe mode.
For the given query and available tools, plan the step by step execution, based on the planning select the relevant tool from the available tools and based on the tool selection, you perfom an action to call the tool.
Wait for the observation from the tool and based on the observation, resolve the user query.

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query

Output Format:
{{
    step: "string", 
    content: "string" 
    function: "the name of the function if step is action",
    input: "the input parameter for the function
}}

Available Tools:
get_weather: Takes a city name and returns the current weather in that city.


Example:
User Query: "What is the weather in New York?"
Output: {{ "step": "plan", "content": "The user is interested in weather data for New York." }}
Output: {{ "step": "plan", "content": "From the available tools i should call get_weather" }}
Output: {{ "step": "action", "function": "get_weather", "input": "New York" }}
Output: {{ "step": "observe", "output":"12 Degree Celsius" }}
Output: {{ "step": "output", "content":"The weather for New york seems to be 12 degrees Celsius" }}
'''

messages=[
    { "role": "system", "content": system_prompt }
]

while True:
    query= input("> ")
    messages.append({ "role": "user", "content": query })
    while True:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_format={
            "type": "json_object",
        },
        messages=messages 
    )
    
        parsed_response = json.loads(response.choices[0].message.content)
        messages.append({ 'role': 'assistant', 'content': json.dumps(parsed_response) })
    
        if parsed_response.get("step") == "plan":
            print(f'🧠: {parsed_response.get("content")}')
            continue
    
        if parsed_response.get("step") == "action":
            tool_name = parsed_response.get("function")
            tool_input = parsed_response.get("input")
        
            if available_tools.get(tool_name,False) != False:
                output=available_tools[tool_name].get("fn")(tool_input)
                messages.append({
                    'role': 'assistant',
                    'content': json.dumps({"step": "observe", "output": output})
                })
    
        if parsed_response.get("step") == "output":
            print(f'🤖: {parsed_response.get('content')}')
            break
