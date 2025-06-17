# examples/vision_simple.py
"""
Vision example - analyze images with LLMs
Generates a test image using Pillow if needed
"""

from chuk_llm import ask_openai_gpt4o, ask_openai_gpt4o_sync, ask_anthropic_sonnet, ask_gemini_flash, ask_openai_gpt3_5_sync
from PIL import Image, ImageDraw, ImageFont
import io
from pathlib import Path

def create_test_image(filename="test_image.png"):
    """Create a simple test image with Pillow"""
    # Create a new image with white background
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    # Red rectangle
    draw.rectangle([50, 50, 150, 150], fill='red', outline='black', width=2)
    
    # Blue circle
    draw.ellipse([200, 50, 300, 150], fill='blue', outline='black', width=2)
    
    # Green triangle
    draw.polygon([(100, 200), (50, 280), (150, 280)], fill='green', outline='black', width=2)
    
    # Yellow star-like shape
    draw.polygon([(250, 200), (230, 240), (270, 240)], fill='yellow', outline='black', width=2)
    
    # Add some text
    try:
        # Try to use a better font if available
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        # Fall back to default font
        font = ImageFont.load_default()
    
    draw.text((120, 20), "Shapes Test", fill='black', font=font)
    
    # Save the image
    img.save(filename)
    print(f"Created test image: {filename}")
    return img

def create_chart_image(filename="chart_image.png"):
    """Create a simple bar chart image"""
    img = Image.new('RGB', (500, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw axes
    draw.line([(50, 350), (450, 350)], fill='black', width=2)  # X-axis
    draw.line([(50, 50), (50, 350)], fill='black', width=2)   # Y-axis
    
    # Draw bars
    bar_data = [
        ("Jan", 120, 'lightblue'),
        ("Feb", 200, 'lightgreen'),
        ("Mar", 150, 'lightcoral'),
        ("Apr", 300, 'lightyellow'),
        ("May", 250, 'lightpink')
    ]
    
    bar_width = 60
    spacing = 20
    x_start = 80
    
    for i, (month, value, color) in enumerate(bar_data):
        x = x_start + i * (bar_width + spacing)
        height = int(value * 0.8)  # Scale to fit
        y_bottom = 350
        y_top = y_bottom - height
        
        # Draw bar
        draw.rectangle([x, y_top, x + bar_width, y_bottom], fill=color, outline='black')
        
        # Draw label
        draw.text((x + bar_width//2 - 10, y_bottom + 5), month, fill='black')
        
        # Draw value
        draw.text((x + bar_width//2 - 15, y_top - 20), str(value), fill='black')
    
    # Title
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((180, 20), "Monthly Sales Data", fill='black', font=font)
    
    img.save(filename)
    print(f"Created chart image: {filename}")
    return img

# Async examples
async def vision_examples():
    """Run async vision examples"""
    print("Async Vision Examples")
    print("=" * 50)
    
    # Create test images if they don't exist
    if not Path("test_image.png").exists():
        create_test_image()
    
    if not Path("chart_image.png").exists():
        create_chart_image()
    
    # Example 1: Simple image analysis
    print("\n1. Analyzing shapes image...")
    response = await ask_openai_gpt4o("What shapes and colors do you see in this image?", "test_image.png")
    print(f"GPT-4o: {response[:200]}...")
    
    # Example 2: Chart analysis
    print("\n2. Analyzing chart...")
    response = await ask_anthropic_sonnet(
        "Analyze this chart. What type of chart is it and what does it show?", 
        "chart_image.png"
    )
    print(f"Claude: {response[:200]}...")
    
    # Example 3: Using image bytes directly
    print("\n3. Using image bytes...")
    img = Image.new('RGB', (200, 200), color='purple')
    draw = ImageDraw.Draw(img)
    draw.text((50, 90), "HELLO", fill='white')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    response = await ask_gemini_flash("What text do you see in this image?", img_bytes.read())
    print(f"Gemini: {response}")

# Sync examples
def vision_sync_examples():
    """Run sync vision examples"""
    print("Sync Vision Examples")
    print("=" * 50)
    
    # Create test images if they don't exist
    if not Path("test_image.png").exists():
        create_test_image()
    
    # Simple sync usage
    print("\n1. Basic sync analysis...")
    response = ask_openai_gpt4o_sync(
        "Describe this image in detail. What shapes, colors, and text do you see?", 
        "test_image.png"
    )
    print(f"Response: {response[:300]}...")
    
    # Non-vision model (no image parameter)
    print("\n2. Non-vision model...")
    response = ask_openai_gpt3_5_sync("What are the primary colors?")
    print(f"Response: {response}")
    
    # Create and analyze an image on the fly
    print("\n3. Dynamic image creation and analysis...")
    
    # Create a simple math problem image
    img = Image.new('RGB', (300, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a math problem
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 50), "2 + 2 = ?", fill='black', font=font)
    draw.text((50, 100), "5 × 3 = ?", fill='blue', font=font)
    
    # Save and analyze
    img.save("math_problem.png")
    response = ask_openai_gpt4o_sync(
        "Solve the math problems shown in this image", 
        "math_problem.png"
    )
    print(f"Math solutions: {response}")

def compare_providers_example():
    """Compare how different providers describe the same image"""
    print("\n\nComparing Providers")
    print("=" * 50)
    
    # Create a complex test image
    img = Image.new('RGB', (400, 400), color='lightgray')
    draw = ImageDraw.Draw(img)
    
    # Draw a face
    # Head
    draw.ellipse([100, 50, 300, 250], fill='peachpuff', outline='black', width=2)
    
    # Eyes
    draw.ellipse([140, 100, 180, 140], fill='white', outline='black', width=2)
    draw.ellipse([220, 100, 260, 140], fill='white', outline='black', width=2)
    
    # Pupils
    draw.ellipse([155, 115, 165, 125], fill='black')
    draw.ellipse([235, 115, 245, 125], fill='black')
    
    # Nose
    draw.polygon([(200, 140), (190, 170), (210, 170)], outline='black', width=2)
    
    # Smile
    draw.arc([150, 160, 250, 220], start=0, end=180, fill='black', width=3)
    
    # Hair
    for i in range(100, 301, 20):
        draw.arc([i, 40, i+30, 80], start=0, end=180, fill='brown', width=5)
    
    # Add text
    draw.text((150, 300), "Happy Face", fill='black')
    
    # Save
    img.save("face_drawing.png")
    
    prompt = "Describe what you see in this drawing. Is it happy or sad?"
    
    # Test different providers
    providers_to_test = [
        ("GPT-4o", ask_openai_gpt4o_sync),
        # Uncomment if you have API keys for these:
        # ("Claude", ask_anthropic_sonnet_sync),
        # ("Gemini", ask_gemini_flash_sync),
    ]
    
    for provider_name, func in providers_to_test:
        try:
            print(f"\n{provider_name}:")
            response = func(prompt, "face_drawing.png")
            print(f"{response[:200]}...")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import asyncio
    
    # Run sync examples
    vision_sync_examples()
    
    # Run async examples
    asyncio.run(vision_examples())
    
    # Run comparison
    compare_providers_example()
    
    # Cleanup message
    print("\n\nCreated test images:")
    for img in ["test_image.png", "chart_image.png", "math_problem.png", "face_drawing.png"]:
        if Path(img).exists():
            print(f"  - {img}")