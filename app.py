#!/usr/bin/env python
# coding: utf-8
# Conversational Agent with Gemini API


import os
import requests
import datetime
import wikipedia
from dotenv import load_dotenv, find_dotenv
from typing import Type
from textblob import TextBlob
from PIL import Image
from io import BytesIO
import re
from collections import Counter

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(find_dotenv())



@tool
def get_current_temperature(latitude: float = None, longitude: float = None, city: str = None) -> str:
    """Get current weather information for a location.
    You can provide either coordinates (latitude, longitude) OR city name.
    
    Examples:
    - get_current_temperature(city="Istanbul")
    - get_current_temperature(city="New York")
    - get_current_temperature(latitude=40.7128, longitude=-74.0060)
    """
    
    try:
        # Eğer şehir ismi verilmişse, koordinatları bul
        if city and not (latitude and longitude):
            # Geocoding API kullanarak şehrin koordinatlarını bul
            geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
            geo_response = requests.get(geocoding_url)
            
            if geo_response.status_code == 200:
                geo_data = geo_response.json()
                if geo_data.get('results'):
                    location = geo_data['results'][0]
                    latitude = location['latitude']
                    longitude = location['longitude']
                    city_name = location['name']
                    country = location.get('country', '')
                else:
                    return f"❌ City '{city}' not found. Please check the spelling."
            else:
                return "❌ Error finding city location"
        
        if not (latitude and longitude):
            return "❌ Please provide either city name or coordinates"
        
        # Open-Meteo API için parametreler
        BASE_URL = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'current': 'temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,wind_direction_10m',
            'hourly': 'temperature_2m',
            'forecast_days': 1,
            'timezone': 'auto'
        }
        
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            results = response.json()
            
            # Weather code descriptions
            weather_codes = {
                0: "☀️ Clear sky",
                1: "🌤️ Mainly clear",
                2: "⛅ Partly cloudy",
                3: "☁️ Overcast",
                45: "🌫️ Foggy",
                48: "🌫️ Depositing rime fog",
                51: "🌦️ Light drizzle",
                53: "🌦️ Moderate drizzle",
                61: "🌧️ Slight rain",
                63: "🌧️ Moderate rain",
                65: "🌧️ Heavy rain",
                71: "🌨️ Slight snow",
                73: "🌨️ Moderate snow",
                75: "🌨️ Heavy snow",
                77: "🌨️ Snow grains",
                80: "🌦️ Slight rain showers",
                81: "🌦️ Moderate rain showers",
                82: "🌧️ Heavy rain showers",
                95: "⛈️ Thunderstorm",
                96: "⛈️ Thunderstorm with hail"
            }
            
            current = results.get('current', {})
            current_temp = current.get('temperature_2m', 0)
            feels_like = current.get('apparent_temperature', 0)
            humidity = current.get('relative_humidity_2m', 0)
            wind_speed = current.get('wind_speed_10m', 0)
            wind_direction = current.get('wind_direction_10m', 0)
            weather_code = current.get('weather_code', 0)
            weather_desc = weather_codes.get(weather_code, "Unknown")
            
            # Rüzgar yönünü hesapla
            directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            index = round(wind_direction / 22.5) % 16
            wind_dir = directions[index]
            
            location_str = city_name if 'city_name' in locals() else f"({latitude}, {longitude})"
            if 'country' in locals() and country:
                location_str += f", {country}"
            
            result = f"""
🌍 Weather for {location_str}:

{weather_desc}
🌡️ Temperature: {current_temp}°C
🤔 Feels like: {feels_like}°C
💧 Humidity: {humidity}%
💨 Wind: {wind_speed} km/h from {wind_dir}

📍 Coordinates: {latitude:.4f}, {longitude:.4f}
🕒 Time: {current.get('time', 'Unknown')}
"""
            return result
            
        else:
            return f"❌ Error fetching weather data: Status {response.status_code}"
            
    except Exception as e:
        return f"❌ Error getting weather: {str(e)}"

        

@tool
def analyze_sentiment(text: str) -> str:
    """REQUIRED tool to analyze sentiment. Use this whenever sentiment analysis is requested.
    DO NOT guess sentiment scores - this tool provides accurate analysis.
    Analyzes polarity (-1 to 1) and subjectivity (0 to 1) of text.
    """
    from textblob import TextBlob
    
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.5:
            sentiment = "Very Positive 😊"
        elif polarity > 0.1:
            sentiment = "Positive 🙂"
        elif polarity > -0.1:
            sentiment = "Neutral 😐"
        elif polarity > -0.5:
            sentiment = "Negative 😞"
        else:
            sentiment = "Very Negative 😢"
        
        if subjectivity > 0.7:
            objectivity = "Highly subjective (personal opinion)"
        elif subjectivity > 0.4:
            objectivity = "Somewhat subjective"
        else:
            objectivity = "Mostly objective (factual)"
        
        result = f"""
📊 Sentiment Analysis Results:
📝 Text: "{text[:100]}{'...' if len(text) > 100 else ''}"

🎭 Overall Sentiment: {sentiment}
📈 Polarity Score: {polarity:.3f} (range: -1 to 1)
💭 Subjectivity Score: {subjectivity:.3f} (range: 0 to 1)
📖 Type: {objectivity}

💡 Interpretation:
- Polarity shows how positive/negative the text is
- Subjectivity shows how opinionated vs factual the text is
"""
        return result
        
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

        

@tool
def describe_image_from_url(image_url: str) -> str:
    """ALWAYS use this tool when asked about images. NEVER guess what's in an image.
    This tool analyzes and describes the contents of an image from a URL.
    Uses computer vision AI to detect objects, scenes, faces, text, and colors."""
    
    import requests
    import base64
    from io import BytesIO
    from PIL import Image
    
    try:
        # Görüntüyü indir
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            return f"Error: Could not download image from URL. Status code: {response.status_code}"
        
        # PIL ile görüntüyü aç
        img = Image.open(BytesIO(response.content))
        
        # Temel görüntü bilgileri
        width, height = img.size
        format = img.format
        mode = img.mode
        
        # API key'leri al
        api_key = os.environ.get('IMAGGA_API_KEY', '')
        api_secret = os.environ.get('IMAGGA_API_SECRET', '')
        
        if api_key and api_secret:
            # Imagga API'ye gönder - DÜZELTİLMİŞ URL
            api_url = f'https://api.imagga.com/v2/tags?image_url={image_url}'
            
            response = requests.get(
                api_url,
                auth=(api_key, api_secret)
            )
            
            if response.status_code == 200:
                data = response.json()
                tags = data.get('result', {}).get('tags', [])[:10]
                
                # Sonuçları formatla
                tag_list = []
                for tag in tags:
                    confidence = tag['confidence']
                    tag_name = tag['tag']['en']
                    if confidence > 30:  # Eşiği düşürdüm
                        tag_list.append(f"- {tag_name} ({confidence:.1f}% confidence)")
                
                result = f"""
🖼️ Image Analysis Results:
📍 URL: {image_url[:50]}{'...' if len(image_url) > 50 else ''}
📐 Dimensions: {width} x {height} pixels
📄 Format: {format if format else 'Unknown'}
🎨 Color Mode: {mode}

🔍 Detected Objects/Concepts (AI Vision):
{chr(10).join(tag_list) if tag_list else 'No objects detected'}

💡 This is what I see in the image using AI vision analysis.
"""
            else:
                # API hatası
                result = f"""
🖼️ Basic Image Information:
📍 URL: {image_url[:50]}{'...' if len(image_url) > 50 else ''}
📐 Dimensions: {width} x {height} pixels
📄 Format: {format if format else 'Unknown'}
🎨 Color Mode: {mode}

❌ API Error: {response.status_code} - {response.text[:100]}
"""
        else:
            # API key yoksa
            result = f"""
🖼️ Basic Image Information:
📍 URL: {image_url[:50]}{'...' if len(image_url) > 50 else ''}
📐 Dimensions: {width} x {height} pixels
📄 Format: {format if format else 'Unknown'}
🎨 Color Mode: {mode}

ℹ️ API keys not configured.
"""
        
        return result
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"




@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[:3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary[:200]}...")
        except (
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)



@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """ALWAYS use this tool for ANY currency conversion or exchange rate question.
    Convert between currencies using real-time exchange rates.
    
    MUST USE FOR:
    - Currency conversions (e.g., "convert X to Y")
    - Exchange rate questions (e.g., "what's the rate between X and Y")
    - Money calculations between different currencies
    - Questions about how much something costs in another currency
    
    Example: convert_currency(100, "USD", "EUR")
    Supports: USD, EUR, GBP, TRY, JPY, CNY, INR, AUD, CAD, CHF, etc."""
    
    try:
        # ExchangeRate-API kullanıyoruz (ücretsiz plan mevcut)
        api_key = os.environ.get('EXCHANGERATE_API_KEY', '')
        
        if api_key:
            # API key varsa premium endpoint kullan
            url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency.upper()}/{to_currency.upper()}/{amount}"
        else:
            # API key yoksa ücretsiz endpoint kullan (limit var)
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if api_key:
                # Premium API response
                if data.get('result') == 'success':
                    converted = data.get('conversion_result', 0)
                    rate = data.get('conversion_rate', 0)
                    
                    result = f"""
💱 Currency Conversion Result:
📊 {amount:,.2f} {from_currency.upper()} = {converted:,.2f} {to_currency.upper()}
📈 Exchange Rate: 1 {from_currency.upper()} = {rate:.4f} {to_currency.upper()}
🕒 Last Updated: {data.get('time_last_update_utc', 'Unknown')}
"""
                else:
                    return f"Error: {data.get('error-type', 'Unknown error')}"
            else:
                # Ücretsiz API
                rates = data.get('rates', {})
                if to_currency.upper() in rates:
                    rate = rates[to_currency.upper()]
                    converted = amount * rate
                    
                    result = f"""
💱 Currency Conversion Result:
📊 {amount:,.2f} {from_currency.upper()} = {converted:,.2f} {to_currency.upper()}
📈 Exchange Rate: 1 {from_currency.upper()} = {rate:.4f} {to_currency.upper()}
⚠️ Note: Using free tier with limited requests
"""
                else:
                    return f"Error: Currency {to_currency} not found"
            
            return result
        else:
            return f"Error fetching exchange rates: Status {response.status_code}"
            
    except Exception as e:
        return f"Error converting currency: {str(e)}"
        


@tool
def translate_text(text: str, target_language: str) -> str:
    """[MANDATORY TOOL] Translate ANY text between languages. 
    YOU MUST USE THIS TOOL for all translation requests!
    
    TRIGGER WORDS: çevir, translate, tercüme, türkçeye, to english, to turkish
    
    Args:
        text: Text to translate (the actual text, not the whole request)
        target_language: 'tr' for Turkish, 'en' for English, etc.
    
    Example calls:
    - User: "şunu türkçeye çevir: hello" → translate_text("hello", "tr")
    - User: "translate merhaba to english" → translate_text("merhaba", "en")
    """
    
    try:
        import urllib.parse
        
        # Dil kodları mapping
        language_map = {
            'turkish': 'tr', 'türkçe': 'tr',
            'english': 'en', 'ingilizce': 'en', 
            'spanish': 'es', 'ispanyolca': 'es',
            'french': 'fr', 'fransızca': 'fr',
            'german': 'de', 'almanca': 'de',
            'italian': 'it', 'italyanca': 'it',
            'portuguese': 'pt', 'portekizce': 'pt',
            'russian': 'ru', 'rusça': 'ru',
            'japanese': 'ja', 'japonca': 'ja',
            'korean': 'ko', 'korece': 'ko',
            'chinese': 'zh', 'çince': 'zh',
            'arabic': 'ar', 'arapça': 'ar'
        }
        
        # Dil kodunu al
        target_code = language_map.get(target_language.lower(), target_language.lower())
        
        # Basit dil algılama (Türkçe karakterler var mı?)
        source_code = 'tr' if any(c in text for c in 'çğıöşüÇĞİÖŞÜ') else 'en'
        
        # Daha detaylı dil algılama
        common_words = {
            'tr': ['ve', 'bir', 'bu', 'için', 'ama', 'olan', 'ile', 'merhaba', 'benim'],
            'en': ['the', 'and', 'is', 'to', 'of', 'a', 'in', 'that', 'hello'],
            'es': ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'hola'],
            'fr': ['le', 'de', 'un', 'et', 'être', 'que', 'pour', 'dans', 'bonjour'],
            'de': ['der', 'die', 'und', 'in', 'ein', 'zu', 'das', 'ist', 'hallo']
        }
        
        text_lower = text.lower()
        for lang, words in common_words.items():
            if sum(1 for word in words if word in text_lower) >= 2:
                source_code = lang
                break
        
        # URL encode the text
        encoded_text = urllib.parse.quote(text)
        
        # MyMemory API URL - belirli dil çifti ile
        url = f"https://api.mymemory.translated.net/get?q={encoded_text}&langpair={source_code}|{target_code}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['responseStatus'] == 200:
                translated = data['responseData']['translatedText']
                
                # Eğer limit aşımı varsa
                if 'MYMEMORY WARNING' in translated:
                    return "⚠️ Daily limit exceeded. Please try again tomorrow."
                
                result = f"""
🌐 Translation Result:
📝 Original: "{text}"
🔍 Detected Language: {source_code}
🎯 Target Language: {target_code}
✨ Translation: "{translated}"

📊 API: MyMemory (Free tier - 5000 chars/day)
"""
                return result
            else:
                return f"❌ Translation failed: {data.get('responseDetails', 'Unknown error')}"
        else:
            return f"❌ API Error: Status {response.status_code}"
            
    except Exception as e:
        return f"❌ Error translating text: {str(e)}"




@tool
def get_nasa_apod() -> str:
    """Get NASA's Astronomy Picture of the Day"""
    
    try:
        api_key = os.environ.get('NASA_API_KEY', 'DEMO_KEY')
        url = f"https://api.nasa.gov/planetary/apod?api_key={api_key}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            title = data.get('title', 'No title')
            date = data.get('date', 'Unknown date')
            explanation = data.get('explanation', 'No description')[:300]
            image_url = data.get('url', '')
            hd_url = data.get('hdurl', '')
            media_type = data.get('media_type', '')
            
            result = f"""NASA Astronomy Picture of the Day
Date: {date}
Title: {title}

Description:
{explanation}...

Media Type: {media_type}
Image URL: {image_url}
HD URL: {hd_url}"""
            
            return result
        else:
            return f"Error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_mars_photos(sol: int = 1000) -> str:
    """Get photos from Mars rovers. Sol = Mars day (1000 = day 1000 on Mars)"""
    
    try:
        api_key = os.environ.get('NASA_API_KEY', 'DEMO_KEY')
        url = f"https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos?sol={sol}&api_key={api_key}&page=1"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            photos = data.get('photos', [])[:5]
            
            if not photos:
                return f"No photos found for sol {sol}"
            
            result = f"Mars Rover Photos - Sol {sol}\n"
            result += f"Total photos: {len(data.get('photos', []))}\n\n"
            
            for i, photo in enumerate(photos, 1):
                camera = photo.get('camera', {}).get('full_name', 'Unknown')
                earth_date = photo.get('earth_date', 'Unknown')
                img_src = photo.get('img_src', '')
                
                result += f"{i}. Camera: {camera}\n"
                result += f"   Date: {earth_date}\n"
                result += f"   URL: {img_src}\n\n"
            
            return result
        else:
            return f"Error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"


# Define tools
tools = [

    get_current_temperature, 
    search_wikipedia, 
    analyze_sentiment, 
    describe_image_from_url, 
    convert_currency, 
    translate_text,
    get_mars_photos,
    get_nasa_apod
]


# Initialize Gemini model - using Gemini 2.0 Flash
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.environ['GOOGLE_API_KEY'],
    temperature=0,  
    top_p=0.8,      
    convert_system_message_to_human=True
)



# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to various tools.
    
    TOOL USAGE RULES:
    1. WEATHER: Use get_current_temperature for weather info
    2. WIKIPEDIA: Use search_wikipedia for general knowledge
    3. SENTIMENT: Use analyze_sentiment for emotion analysis
    4. IMAGES: Use describe_image_from_url for image analysis
    5. CURRENCY: Use convert_currency for exchange rates
    6. TRANSLATION: Use translate_text for language translation
    7. NEWS: Use get_latest_news for current news and events
    
    Be helpful and use the appropriate tools!"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Initialize chat history
chat_history = InMemoryChatMessageHistory()

# Create the agent
agent = create_tool_calling_agent(model, tools, prompt)

# Create agent executor without deprecated memory
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,  # This will show tool usage in terminal
)

# Wrapper for maintaining chat history
def run_with_history(input_text):
    messages = chat_history.messages
    result = agent_executor.invoke({
        "input": input_text,
        "chat_history": messages
    })
    # Add messages to history
    chat_history.add_user_message(input_text)
    chat_history.add_ai_message(result['output'])
    return result

def main():
    print("=== Conversational Agent with Gemini 2.0 Flash ===")
    print("Available tools: weather (needs coordinates), wikipedia search")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
            
        if not user_input:
            continue
        
        try:
            print("\n🤖 Assistant is thinking...")
            print("-" * 50)
            
            # Invoke the agent with history
            result = run_with_history(user_input)
            
            print("-" * 50)
            print(f"\n🤖 Assistant: {result['output']}")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
    
if __name__ == "__main__":
    
    # Start interactive session
    main()