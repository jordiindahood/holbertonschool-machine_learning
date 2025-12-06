#!/usr/bin/env python3
""" script 3 """
import requests


def get_first_launch():
    """
    Retrieves the first SpaceX launch and prints:
    <launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
    """

    # Get all launches
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets/"
    launchpads_url = "https://api.spacexdata.com/v4/launchpads/"

    response = requests.get(launches_url)
    if response.status_code != 200:
        return

    launches = response.json()

    # Sort by date_unix
    launches.sort(key=lambda x: x.get("date_unix", 0))

    first_launch = launches[0]

    # Get rocket info
    rocket_id = first_launch.get("rocket")
    rocket_name = (
        requests.get(rockets_url + rocket_id).json().get("name", "unknown")
    )

    # Get launchpad info
    launchpad_id = first_launch.get("launchpad")
    lp_data = requests.get(launchpads_url + launchpad_id).json()
    lp_name = lp_data.get("name", "unknown")
    lp_locality = lp_data.get("locality", "unknown")

    # Get launch name and date
    launch_name = first_launch.get("name", "unknown")
    launch_date = first_launch.get("date_local", "unknown")

    # Print formatted string
    print(
        f"{launch_name} ({launch_date}) {rocket_name} - {lp_name} ({lp_locality})"
    )


if __name__ == "__main__":
    get_first_launch()
