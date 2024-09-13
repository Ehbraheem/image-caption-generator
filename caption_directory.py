import time
from utils import parse_dir, parallel_execution, write_caption_line
from model import caption_image


def caption(directory: str, output_file='caption.txt'):
    with open(output_file, 'w') as file:
        try:

            start = time.perf_counter()
            print(f'Captioning of all images in {directory}\n----------------')
            image_array = parse_dir(directory)
            writer = write_caption_line(file)

            captions = parallel_execution(lambda val: (val[0], caption_image(val[1])), image_array)
            
            parallel_execution(writer, captions)
            print(f'Completed the task of captioning images in {directory}')
            print(f'Total time: {time.perf_counter() - start}')
        except Exception as e:
            print(f'Exception occurred while trying to caption images in {directory}')
            print(e)
            raise e
        



if __name__ == '__main__':
    img_path = './data/test'

    caption(img_path, 'text-caption.txt')