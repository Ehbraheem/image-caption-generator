import time
from utils import parse_url, parallel_execution, write_caption_line
from model import caption_image


def caption(url: str, output_file='caption.txt'):
    with open(output_file, 'w') as file:
        try:

            start = time.perf_counter()
            print(f'Captioning of all images in {url}\n----------------')
            image_array = parse_url(url)
            writer = write_caption_line(file)

            # TODO: Investigate why ThreadPoolExecute is slow
            # captions = parallel_execution(lambda val: (val[0], caption_image(val[1])), image_array)
            # TODO: THis is also slow. Investivate with links that has 100s of images to confirm effectiveness.
            # captions = map(lambda val: (val[0], caption_image(val[1])), image_array)
            # For loop for the win
            captions = []
            for link, img in image_array:
                caption = caption_image(img)
                captions.append((link, caption))
            
            parallel_execution(writer, captions)
            print(f'Completed the task of captioning images in {url}')
            print(f'Total time: {time.perf_counter() - start}')
        except Exception as e:
            print(f'Exception occurred while trying to caption images in {url}')
            print(e)
            raise e
        



if __name__ == '__main__':
    url = 'https://en.wikipedia.org/wiki/IBM'

    caption(url)