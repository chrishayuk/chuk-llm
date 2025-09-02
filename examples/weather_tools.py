#!/usr/bin/env python3
"""
Example tools file for weather functionality.
Usage: chuk-llm ask "What's the weather in Paris?" --tools weather_tools.py
"""

import random
from typing import Optional

def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get current weather information for a location"""
    # Mock weather data - replace with real API in production
    temperatures = {
        "celsius": random.randint(-10, 35),
        "fahrenheit": random.randint(14, 95)
    }
    
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "overcast"]
    
    return {
        "location": location,
        "temperature": temperatures.get(unit, temperatures["celsius"]),
        "unit": unit,
        "condition": random.choice(conditions),
        "humidity": random.randint(30, 90),
        "wind_speed": random.randint(5, 25)
    }

def get_forecast(location: str, days: int = 3) -> list:
    """Get weather forecast for multiple days"""
    forecast = []
    for i in range(min(days, 7)):  # Max 7 days
        day_weather = get_weather(location)
        day_weather["day"] = f"Day {i+1}"
        forecast.append(day_weather)
    
    return forecast

def check_air_quality(location: str) -> dict:
    """Check air quality index for a location"""
    aqi_levels = ["Good", "Moderate", "Unhealthy for sensitive groups", "Unhealthy", "Very unhealthy"]
    
    return {
        "location": location,
        "aqi": random.randint(0, 200),
        "level": random.choice(aqi_levels),
        "main_pollutant": random.choice(["PM2.5", "PM10", "NO2", "O3"])
    }

# Export functions for CLI usage
__all__ = ['get_weather', 'get_forecast', 'check_air_quality']