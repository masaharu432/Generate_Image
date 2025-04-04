import argparse
import torch
from diffusers import StableDiffusionPipeline

# 引数の設定
parser = argparse.ArgumentParser(description="Generate an image using Stable Diffusion.")
parser.add_argument("--prompt", type=str, required=True, help="The text prompt to generate the image.")
args = parser.parse_args()

# モデルとデバイスの設定
model_id = "stabilityai/stable-diffusion-2-1-base"
device = "cuda"

# プロンプト
prompt = args.prompt

# パイプラインの作成
pipe = StableDiffusionPipeline.from_pretrained(model_id, variant="fp16", torch_dtype=torch.float16)
pipe = pipe.to(device)

# パイプラインの実行
generator = torch.Generator(device).manual_seed(42)
with torch.autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]

# 生成した画像の保存
image.save("generate.png")