import gradio as gr
from model import caption_image

interface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs='textbox',
    title='Image Captioning',
    description="Web application for generating captions for images using pretrained BLIP models"
)

interface.launch(share=True)
