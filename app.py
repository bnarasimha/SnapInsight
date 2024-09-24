import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import uuid
import json

# Initialize BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Create directory for saving images
os.makedirs("saved_images", exist_ok=True)

# Path for the single JSON file
DATA_FILE = "image_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

def process_image(image, gallery_state):
    if image is None:
        return gallery_state, gallery_state
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Generate caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Save image
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join("saved_images", filename)
    image.save(image_path)
    
    # Update data
    data = load_data()
    data.insert(0, {"image": filename, "caption": caption})
    save_data(data)
    
    # Update gallery state
    gallery_state = [(os.path.join("saved_images", item["image"]), item["caption"]) for item in data]
    
    return gallery_state, gallery_state

def load_initial_gallery():
    data = load_data()
    return [(os.path.join("saved_images", item["image"]), item["caption"]) for item in data]

css = """
h1 {
    text-align: center;
    display:block;
}
"""

# Create Gradio interface
with gr.Blocks(css=css) as iface:
    gr.Markdown("# Image Captioning Feed")
    gr.Markdown("Upload an image to generate a caption and add it to the feed.")
    
    with gr.Row():
        image_input = gr.Image(sources=["upload", "webcam"], type="pil", height=500, width=500)
        gallery_output = gr.Gallery(label="Image Feed", columns=[3], rows=[1], object_fit="contain", height="auto", show_download_button="true")
    
    gallery_state = gr.State(load_initial_gallery())
    
    image_input.change(
        process_image,
        inputs=[image_input, gallery_state],
        outputs=[gallery_state, gallery_output]
    )
    
    gallery_state.change(
        lambda x: x,
        inputs=[gallery_state],
        outputs=[gallery_output]
    )

# Launch the app
iface.launch()
