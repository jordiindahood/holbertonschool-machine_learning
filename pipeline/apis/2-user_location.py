#!/usr/bin/env python3
""" script 2 """

import sys
import requests
from datetime import datetime


def get_user_location(url):
    """
    Fetches GitHub user location from the API URL.
    Handles:
      - 404 (Not found)
      - 403 (Rate limit exceeded)
      - 200 (prints location)
    """
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        location = data.get("location")
        if location:
            print(location)
        else:
            print("Not found")
    elif response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_time = int(response.headers.get("X-Ratelimit-Reset", 0))
        now = int(datetime.utcnow().timestamp())
        minutes = (reset_time - now) // 60
        print(f"Reset in {minutes} min")
    else:
        print("Error:", response.status_code)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./2-user_location.py <github_api_user_url>")
        sys.exit(1)
    url = sys.argv[1]
    get_user_location(url)
