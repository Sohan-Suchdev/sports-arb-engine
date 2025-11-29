import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class OddsAPIManager:
    def __init__(self):
        self.api_key = os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("ERROR: API Key not found! Check your .env file.")
        
        # Base URL for The Odds API
        self.base_url = "https://api.the-odds-api.com/v4/sports"

    def fetch_live_odds(self, sport='soccer_epl', region='uk'):
        url = f"{self.base_url}/{sport}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': region,
            'markets': 'h2h', # Head to Head (Home/Draw/Away)
            'oddsFormat': 'decimal'
        }

        print(f"Connecting to API for {sport}...")
        
        try:
            response = requests.get(url, params=params)
            
            # Check for specific API errors
            if response.status_code == 401:
                print("Authorization Error: Your API Key is invalid.")
                return None
            
            if response.status_code == 429:
                print("Quota Exceeded: You have used all your free requests.")
                return None

            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                return None
            
            data = response.json()
            print(f"Success! Received {len(data)} live events.")
            return data

        except Exception as e:
            print(f"Connection Failed: {str(e)}")
            return None
