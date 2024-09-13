from shutil import disk_usage
from os.path import join
from glob import glob
from functools import reduce
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup


def __free_disk_space__():
    total, used, free = disk_usage(__file__)
    print(total, used, free)

    return free / 1024.0**3 # Convert to GB

def parse_dir(image_dir: str, image_exts=["jpg", "jpeg", "png"]):
    files = reduce(lambda accum, ext: accum + glob(join(image_dir, f'*.{ext}')), image_exts, [])
    print(f'Files found \n-------------- \n{files}')
    files = set(files)
    array_lambda = lambda data: (data, to_nd_array(data))
    image_array = map(array_lambda, files) if len(files) < 100 else parallel_execution(array_lambda, files)

    return image_array

def parse_url(url: str):
    page = open_url(url)
    soup = BeautifulSoup(page, 'html.parser')
    img_tags = soup.find_all('img')

    img_links = map(lambda img: img.get('src'), img_tags)
    img_links = filter(lambda img: not ('svg' in img or '1x1' in img), img_links)
    img_links = reduce(
        lambda accum, img: accum + [f'https:{img}'] if img.startswith('//') else accum,
        img_links,
        []
    )

    images = parallel_execution(download_image, img_links)
    array_lambda = lambda data: (data[0], to_nd_array(data[1]))
    images_array = map(array_lambda, images) if len(images) < 100 else parallel_execution(array_lambda, images)

    # Skip very small images
    images_array = filter(lambda data: data[1].shape[0] * data[1].shape[1] > 400, images_array) 

    return images_array


def to_nd_array(img):
    image = Image.open(img)
    return np.array(image)

def download_image(url: str):
    print(f'Downloading {url}')
    content = open_url(url, response_type='content')
    data = BytesIO(content)
    print(f"Done downloading {url}")

    return url, data

def parallel_execution(task, args):
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(task, arg) for arg in args]
        submitted_jobs = as_completed(futures)
        results = [job.result() for job in submitted_jobs]

    return results

def write_caption_line(file):
    return lambda val: file.write(f'{val[0]}: {val[1]}\n')

def open_url(url: str, response_type='text'):
    response = requests.get(url)

    return getattr(response, response_type)
