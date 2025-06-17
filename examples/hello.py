# examples/vision_quickstart.py
"""
ChukLLM Vision - Quick Start
Just 3 lines to analyze an image!
"""

from chuk_llm import ask_openai_gpt4o_sync, ask_anthropic_sonnet_sync, ask_watsonx_granite_vision_sync
from PIL import Image, ImageDraw

# Create a simple test image
img = Image.new('RGB', (200, 100), 'white')
draw = ImageDraw.Draw(img)
draw.text((50, 40), "HELLO AI!", fill='red')
img.save("hello.png")

# That's it! Just call with image path
print(ask_openai_gpt4o_sync("What text do you see?", "hello.png"))

# Try with Claude too
print(ask_anthropic_sonnet_sync("What color is the text?", "hello.png"))

# Or analyze from a URL
print(ask_openai_gpt4o_sync(
    "What's in this image?", 
    "https://picsum.photos/200/300"
))

# Or analyze from a URL
print(ask_anthropic_sonnet_sync(
    "What's in this image?", 
    "https://picsum.photos/200/300"
))

# # Or analyze from a URL
# print(ask_watsonx_granite_vision_sync(
#     "What's in this image?", 
#     "https://picsum.photos/200/300"
# ))