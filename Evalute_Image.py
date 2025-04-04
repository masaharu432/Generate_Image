import requests
import torch
from PIL import Image, ImageDraw, ImageFont  # Pillowライブラリをインポート
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
with torch.no_grad():  # or `torch.inference_model` in torch 1.9+
    outputs = predictor(**inputs)
prediction = outputs.logits.item()  # スカラー値に変換

# スコアを画像に埋め込む
output_path = '/content/drive/MyDrive/Colab Notebooks/Generate_Image/generate_with_score.png'

# 描画用オブジェクトを作成
draw = ImageDraw.Draw(image)

# デフォルトフォントを使用
font = ImageFont.load_default()

# スコアを描画する位置とテキスト
text = f"Aesthetics Score: {prediction:.2f}"
text_position = (50, 50)  # テキストの描画位置 (x, y)
text_color = (255, 0, 0)  # テキストの色 (RGB)

# テキストを画像に描画
draw.text(text_position, text, fill=text_color, font=font)

# 画像を保存
image.save(output_path)
print(f"スコアを埋め込んだ画像を保存しました: {output_path}")

# Print the aesthetics score
print(f"Aesthetics score: {prediction}")