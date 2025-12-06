#!/usr/bin/env python3
""" script 3 """
import requests


def get_first_launch():
    """
    Fetches all SpaceX launches and prints the first launch in the format:
    <launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
    """
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets/"
    launchpads_url = "https://api.spacexdata.com/v4/launchpads/"

    # Fetch all launches
    r = requests.get(launches_url)
    if r.status_code != 200:
        return
    launches = r.json()

    # Sort by date_unix
    launches.sort(key=lambda x: x.get("date_unix", 0))

    first_launch = launches[0]

    # Get launch details
    launch_name = first_launch.get("name", "unknown")
    launch_date = first_launch.get("date_local", "unknown")

    # Get rocket name
    rocket_id = first_launch.get("rocket")
    rocket_data = requests.get(rockets_url + rocket_id).json()
    rocket_name = rocket_data.get("name", "unknown")

    # Get launchpad info
    launchpad_id = first_launch.get("launchpad")
    launchpad_data = requests.get(launchpads_url + launchpad_id).json()
    lp_name = launchpad_data.get("name", "unknown")
    lp_locality = launchpad_data.get("locality", "unknown")

    print(
        f"{launch_name} ({launch_date}) {rocket_name} - {lp_name} ({lp_locality})"
    )


if __name__ == "__main__":
    get_first_launch()
