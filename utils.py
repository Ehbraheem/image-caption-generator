from shutil import disk_usage
from os.path import join
from glob import glob
from functools import reduce
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image


def __free_disk_space__():
    total, used, free = disk_usage(__file__)
    print(total, used, free)

    return free / 1024.0**3 # Convert to GB

def parse_dir(image_dir: str, image_exts=["jpg", "jpeg", "png"]):
    files = reduce(lambda accum, ext: accum + glob(join(image_dir, f'*.{ext}')), image_exts, [])
    print(f'Files found \n-------------- \n{files}')
    files = set(files)
    image_array = map(to_nd_array, files) if len(files) < 100 else parallel_execution(to_nd_array, files)

    return image_array

def to_nd_array(img_path):
    image = Image.open(img_path)
    return img_path, np.array(image)


def parallel_execution(task, args):
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(task, arg) for arg in args]
        submitted_jobs = as_completed(futures)
        results = [job.result() for job in submitted_jobs]

    return results

def write_caption_line(file):
    return lambda val: file.write(f'{val[0]}: {val[1]}\n')
