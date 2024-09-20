import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

feed = []

def process_image(image):
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Generate caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Add to feed
    feed.insert(0, (image, caption))
    
    # Return the updated feed
    return [(img, cap) for img, cap in feed]

# Create Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(sources=["upload", "webcam"], type="pil"),
    outputs=gr.Gallery(label="Image Feed", columns=[2], height="auto"),
    title="Image Captioning Feed",
    description="Upload an image or take a picture to generate a caption and add it to the feed."
)

# Launch the app
demo.launch(server_name="0.0.0.0", server_port=7860)
