#!/usr/bin/env python3
""" script 0 """

import requests


def availableShips(passengerCount):
    """
    Uses SWAPI API and handles pagination.
    """

    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break
        data = response.json()
        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0").replace(",", "")
            if passengers.lower() == "n/a":
                continue
            try:
                passengers_num = int(passengers)
                if passengers_num >= passengerCount:
                    ships.append(ship["name"])
            except ValueError:
                # Some entries have ranges like "30-165", take the max
                if "-" in passengers:
                    max_passengers = int(passengers.split("-")[-1])
                    if max_passengers >= passengerCount:
                        ships.append(ship["name"])
                # skip non-numeric otherwise
                continue
        # Move to next page
        url = data.get("next")

    return ships
