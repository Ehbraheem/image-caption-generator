import gradio as gr

demo = gr.Interface(
    fn=lambda name: f'Hello {name}!',
    inputs='textbox',
    outputs='textbox'
)

demo.launch(server_name="0.0.0.0", server_port= 7860)
