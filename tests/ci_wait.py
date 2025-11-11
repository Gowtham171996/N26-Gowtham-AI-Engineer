import sys
import time

import requests

# The critical endpoint we need to be available to start testing
API_URL = "http://localhost:8000/status"
# Maximum time (seconds) we allow for the service to start
MAX_WAIT_TIME = 90
# Initial delay before the first check (allows containers a moment to boot)
INITIAL_DELAY = 5
# Time to wait between retries (increases slightly on each failure)
RETRY_INTERVAL = 2


def wait_for_service():
    """
    Polls the API_URL until it returns a 200 status or the MAX_WAIT_TIME is exceeded.
    Exits with code 1 on failure.
    """
    print(f"Waiting up to {MAX_WAIT_TIME} seconds for service at {API_URL}...")

    time.sleep(INITIAL_DELAY)
    start_time = time.time()
    current_interval = RETRY_INTERVAL

    while time.time() - start_time < MAX_WAIT_TIME:
        try:
            response = requests.get(API_URL, timeout=5)
            if response.status_code == 200:
                print(f"\n[SUCCESS] Service is up! Status: {response.status_code}")
                # Quantitative confirmation of wait time
                wait_duration = int(time.time() - start_time)
                print(f"Total wait time: {wait_duration} seconds.")
                return 0
        except requests.exceptions.ConnectionError:
            pass  # Expected when the service is still booting

        # Print progress and wait for the calculated interval
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(current_interval)

        # Simple increase of interval to prevent hammering the service
        current_interval = min(
            current_interval * 1.1, 10
        )  # Max 10 seconds between checks

    # If the loop exits due to timeout
    print(
        f"\n[FAILURE] Service failed to start within the {MAX_WAIT_TIME} second timeout."
    )
    return 1


if __name__ == "__main__":
    sys.exit(wait_for_service())
