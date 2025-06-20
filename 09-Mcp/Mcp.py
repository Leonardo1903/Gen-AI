from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("My MCP")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b 

@mcp.tool()
def getWeather(city: str):
    """Get the weather for a city"""
    url = f'https://wttr.in/{city}?format=%C+%t'
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.text.strip()
        return weather_data
    else:
        return "Could not retrieve weather data."