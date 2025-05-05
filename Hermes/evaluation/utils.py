import time


def retry(max_attempts, delay):
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempts + 1} failed.\n"
                          f"{e}\n"
                          f"Retrying in {delay} seconds.")
                    attempts += 1
                    time.sleep(delay)
            print("[ERROR] Max retry attempts exceeded.")

        return wrapper

    return decorator
