#!/usr/bin/env python3
""" script 1 """

import requests


def sentientPlanets():
    """
    Returns a list of names of home planets of all sentient species.
    Sentient species are identified if 'sentient' is in
    classification or designation.
    """

    url = "https://swapi.dev/api/species/"
    planets = set()  # Use a set to avoid duplicates

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break
        data = response.json()
        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()
            if "sentient" in classification or "sentient" in designation:
                homeworld_url = species.get("homeworld")
                if homeworld_url:
                    hw_response = requests.get(homeworld_url)
                    if hw_response.status_code == 200:
                        hw_data = hw_response.json()
                        planets.add(hw_data.get("name", "unknown"))
                    else:
                        planets.add("unknown")
                else:
                    planets.add("unknown")
        url = data.get("next")  # pagination

    return list(planets)
