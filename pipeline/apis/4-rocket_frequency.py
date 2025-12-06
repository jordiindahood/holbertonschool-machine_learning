#!/usr/bin/env python3
""" script 4 """

import requests
from collections import defaultdict


def rocket_frequency():
    """
    Fetch all SpaceX launches and display the number of launches per rocket.
    Format: <rocket name>: <count>
    Ordered by count descending, then alphabetically.
    """

    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets/"

    # Fetch all launches
    response = requests.get(launches_url)
    if response.status_code != 200:
        return
    launches = response.json()

    # Count launches per rocket id
    rocket_counts = defaultdict(int)
    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            rocket_counts[rocket_id] += 1

    # Map rocket IDs to names
    rocket_names = {}
    for rocket_id in rocket_counts:
        r_resp = requests.get(rockets_url + rocket_id)
        if r_resp.status_code == 200:
            rocket_names[rocket_id] = r_resp.json().get("name", "unknown")
        else:
            rocket_names[rocket_id] = "unknown"

    # Prepare list of tuples (name, count)
    rockets_list = [
        (rocket_names[rid], count) for rid, count in rocket_counts.items()
    ]

    # Sort by count descending, then name ascending
    rockets_list.sort(key=lambda x: (-x[1], x[0]))

    # Print result
    for name, count in rockets_list:
        print(f"{name}: {count}")


if __name__ == "__main__":
    rocket_frequency()
