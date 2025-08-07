#!/usr/bin/env python3
"""
Azure OpenAI Discovery Example with dotenv support
"""

import asyncio
import logging
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    
    # Look for .env file in current directory or parent directories
    env_file = Path(".env")
    if not env_file.exists():
        env_file = Path("../.env")
    if not env_file.exists():
        env_file = Path("../../.env")
    
    if env_file.exists():
        print(f"📁 Loading environment from: {env_file.absolute()}")
        load_dotenv(env_file)
    else:
        print("📁 No .env file found, using system environment variables")
        
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables")
    print("   Install with: pip install python-dotenv")

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

async def test_azure_discovery_with_env():
    """Test Azure OpenAI discovery with environment variables from .env"""
    
    print("\n🔍 Azure OpenAI Discovery Test with Environment Variables")
    print("=" * 60)
    
    # Check what we have loaded
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    print("📋 Environment Variables Status:")
    print(f"  AZURE_OPENAI_ENDPOINT: {'✅ Set' if azure_endpoint else '❌ Not set'}")
    if azure_endpoint:
        # Show first and last 20 chars for security
        masked_endpoint = f"{azure_endpoint[:20]}...{azure_endpoint[-20:]}" if len(azure_endpoint) > 40 else azure_endpoint
        print(f"    Value: {masked_endpoint}")
    
    print(f"  AZURE_OPENAI_API_KEY: {'✅ Set' if azure_api_key else '❌ Not set'}")
    if azure_api_key:
        print(f"    Length: {len(azure_api_key)} characters")
    
    if not azure_endpoint:
        print("\n❌ Missing AZURE_OPENAI_ENDPOINT")
        print("💡 Add to your .env file:")
        print("   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com")
        return
    
    if not azure_api_key:
        print("\n❌ Missing AZURE_OPENAI_API_KEY") 
        print("💡 Add to your .env file:")
        print("   AZURE_OPENAI_API_KEY=your-api-key-here")
        return
    
    # Test the discoverer
    try:
        print("\n🚀 Testing Azure OpenAI Discovery...")
        
        from chuk_llm.llm.discovery import DiscovererFactory
        
        discoverer = DiscovererFactory.create_discoverer(
            "azure_openai",
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version="2024-02-01"
        )
        
        print(f"✅ Discoverer created: {type(discoverer).__name__}")
        
        print("\n🔍 Running discovery...")
        models = await discoverer.discover_models()
        
        print(f"✅ Discovery completed: {len(models)} models found")
        
        # Categorize results
        deployed_models = [m for m in models if m.get("deployment_status") == "deployed"]
        available_models = [m for m in models if "available" in m.get("deployment_status", "")]
        fallback_models = [m for m in models if m.get("source", "").endswith("fallback")]
        
        if deployed_models:
            print(f"\n🚀 Active Deployments ({len(deployed_models)}):")
            for model in deployed_models[:5]:
                name = model.get("name", "Unknown")
                underlying = model.get("underlying_model", "")
                status = model.get("status", "unknown")
                print(f"   • {name} -> {underlying} [{status}]")
        
        if available_models:
            print(f"\n📚 Available Models ({len(available_models)}):")
            for model in available_models[:5]:
                name = model.get("name", "Unknown") 
                performance = model.get("performance_tier", "standard")
                print(f"   • {name} [{performance}]")
        
        if fallback_models:
            print(f"\n🔄 Fallback Models Used ({len(fallback_models)}):")
            print("   (API discovery failed, using known Azure models)")
            for model in fallback_models[:3]:
                name = model.get("name", "Unknown")
                family = model.get("model_family", "unknown")
                print(f"   • {name} [{family}]")
        
        # Show summary
        print(f"\n💡 Summary:")
        if deployed_models:
            print(f"   🚀 {len(deployed_models)} active deployments found")
        if available_models:
            print(f"   📚 {len(available_models)} models available for deployment")
        if fallback_models:
            print(f"   🔄 {len(fallback_models)} fallback models loaded")
        
        print(f"\n✅ Azure OpenAI discovery test completed successfully!")
        
    except Exception as e:
        print(f"❌ Discovery test failed: {e}")
        
        if "404" in str(e):
            print("\n🔧 Troubleshooting 404 errors:")
            print("   • Check if your Azure OpenAI endpoint is correct")
            print("   • Ensure your resource supports the /openai/deployments endpoint")
            print("   • Try the fallback models which should still work")
        elif "401" in str(e) or "403" in str(e):
            print("\n🔧 Troubleshooting authentication errors:")
            print("   • Verify your AZURE_OPENAI_API_KEY is correct")
            print("   • Check if the API key has the required permissions")
            print("   • Try regenerating the API key in Azure portal")

def create_sample_env_file():
    """Create a sample .env file"""
    
    env_file = Path(".env")
    if env_file.exists():
        print(f"📁 .env file already exists at: {env_file.absolute()}")
        return
    
    sample_content = """# Azure OpenAI Configuration
# Get these values from your Azure OpenAI resource in the Azure portal

# Your Azure OpenAI endpoint (without /openai suffix)
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com

# Your Azure OpenAI API key
AZURE_OPENAI_API_KEY=your-api-key-here

# Optional: Azure AD token (alternative to API key)
# AZURE_AD_TOKEN=your-azure-ad-token

# Optional: API version (defaults to 2024-02-01)
# AZURE_OPENAI_API_VERSION=2024-02-01
"""
    
    print(f"📝 Creating sample .env file at: {env_file.absolute()}")
    with open(env_file, 'w') as f:
        f.write(sample_content)
    
    print("✅ Sample .env file created!")
    print("\n📋 Next steps:")
    print("1. Edit the .env file with your actual Azure OpenAI credentials")
    print("2. Get your endpoint and API key from the Azure portal")
    print("3. Run this script again to test the discovery")

async def main():
    """Main function"""
    
    print("🔷 Azure OpenAI Discovery with dotenv")
    print("=" * 40)
    
    # Check if we have environment variables
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    if not azure_endpoint or not azure_api_key:
        print("⚙️  Environment variables not found")
        
        env_file = Path(".env")
        if not env_file.exists():
            print("📝 Creating sample .env file...")
            create_sample_env_file()
        else:
            print(f"📁 Found .env file: {env_file.absolute()}")
            print("💡 Please verify your Azure OpenAI credentials in the .env file")
        
        return
    
    # Run the discovery test
    await test_azure_discovery_with_env()

if __name__ == "__main__":
    asyncio.run(main())