# coding=utf-8
import os
import numpy as np
import cv2
from PIL import Image
# Flask utils
from flask import Flask, request, jsonify, render_template

# python -m flask run --host=0.0.0.0 --port=5000
# uvicorn main:app --reload --host=0.0.0.0 --port=5000
# "http://172.23.250.43:5000/"

# pip install diffusers transformers accelerate scipy safetensors
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from torch import autocast

device = "cuda"
# model_path = "CompVis/stable-diffusion-v1-4"
model_path = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token="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"
)
pipe = pipe.to(device)


# Define a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return "Hello world"

@app.route('/test', methods=['GET'])
def test():
    prompt = "many landmark, high quality, high definition, extremely detailed"

    seed = 5232
    width = 640
    height = 480

    image = pipe(prompt = prompt,
                num_inference_steps = 10,
                width = width,
                height = height,
                generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None).images[0]
    image.save("./static/test.jpg")
    return render_template('index.html', image_file="test.jpg")


if __name__ == '__main__':
    app.run(debug=True)
