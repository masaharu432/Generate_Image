import requests
import torch
from PIL import Image
from IPython.display import Image as display
from IPython.display import display
from transformers import CLIPProcessor

from aesthetics_predictor import AestheticsPredictorV1

#
# Load the aesthetics predictor
#
model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"

predictor = AestheticsPredictorV1.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

#
# Download sample image
#
image_path = '/content/drive/MyDrive/Colab Notebooks/Generate_Image/generate.png'
image = Image.open(image_path)
#
# Preprocess the image
#
inputs = processor(images=image, return_tensors="pt")

#
# Move to GPU
#
device = "cuda"
predictor = predictor.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

#
# Inference for the image
#
with torch.no_grad(): # or `torch.inference_model` in torch 1.9+
    outputs = predictor(**inputs)
prediction = outputs.logits

display(image_path)

# Print the aesthetics score

print(f"Aesthetics score: {prediction}")