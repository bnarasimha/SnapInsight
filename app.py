import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import uuid
import json
from user_agents import parse

# Initialize BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Create directory for saving images
os.makedirs("saved_images", exist_ok=True)

# Path for the single JSON file
DATA_FILE = "image_data.json"

def load_data():
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
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
    try:
        with open("data.json", "r") as f:
            data = json.load(f)
            return [(os.path.join("saved_images", item["image"]), item["caption"]) for item in data]
    except (FileNotFoundError, json.JSONDecodeError):
        return [] 

def get_device_type(request):
    user_agent = request.headers.get("User-Agent")
    print(user_agent)
    if "Mobile" in user_agent or "Android" in user_agent or "iPhone" in user_agent:
        return "mobile"
    else:
        return "laptop"
    
css = """
h1 {
    text-align: center;
    display:block;
}
"""

# Create Gradio interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Image Captioning Feed")
    gr.Markdown("Upload an image to generate a caption and add it to the feed.")
    
    def set_sources(request: gr.Request):
        device_type = get_device_type(request)
        new_sources = ["upload"] if device_type == "mobile" else ["upload", "webcam"]
        return new_sources, gr.update(sources=new_sources)
    
    sources = gr.State(value=["upload", "webcam"]) 
    print(sources.value)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(sources=sources.value, type="pil", height=500, width=750)
            capture_btn = gr.Button(value="Submit", min_width=700)
        with gr.Column(scale=1):
            gallery_output = gr.Gallery(label="Image & Captions Feed", columns=[3], rows=[1], object_fit="contain", height="auto", show_download_button=True)
    
    gallery_state = gr.State(load_initial_gallery())
    
    capture_btn.click(
        process_image,
        inputs=[image_input, gallery_state],
        outputs=[gallery_state, gallery_output]
    )
    
    gallery_state.change(
        lambda x: x,
        inputs=[gallery_state],
        outputs=[gallery_output]
    )
    demo.load(fn=set_sources, inputs=[], outputs=[sources, image_input])

demo.launch(server_name="0.0.0.0", server_port=7860)
