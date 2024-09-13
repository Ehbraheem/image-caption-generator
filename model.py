import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
import numpy as np
from utils import __free_disk_space__

def setup():
    # Blip2 requires 10GB of free space
    if __free_disk_space__() >= 10.5:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    else:
        processor = AutoProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

    return processor, model


processor, model = setup()

def caption_image(input_image: np.ndarray):
    raw_image = Image.fromarray(input_image).convert('RGB')

    text = 'the image of'
    inputs = processor(images=raw_image, text=text, return_tensors='pt')

    outputs = model.generate(**inputs, max_length=50, max_new_tokens=50)

    # Decode the generated tokens to text
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption



if __name__ == '__main__':
    __free_disk_space__()
    img_path = './data/train/Road_signs_off_Alausa,Lagos.jpg'
    image = Image.open(img_path)
    image_array = np.array(image)

    caption = caption_image(image_array)
    print(caption)
