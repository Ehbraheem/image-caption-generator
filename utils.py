from shutil import disk_usage


def __free_disk_space__():
    total, used, free = disk_usage(__file__)
    print(total, used, free)

    return free / 1024.0**3 # Convert to GB
