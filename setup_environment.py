#!/usr/bin/env python3
"""
Environment Setup Script for GPT-Researcher
Helps configure API keys and test the setup for optimal researcher performance.
"""

import os
import sys
from pathlib import Path

def create_env_template():
    """Create a .env template file with required API keys."""
    env_template = """# GPT-Researcher API Configuration
# Required API Keys for optimized researcher functionality

# OpenAI API Key (Required for LLM operations)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Tavily API Key (Required for web search and retrieval)
# Get from: https://tavily.com/ 
TAVILY_API_KEY=your_tavily_api_key_here

# Optional API Keys for enhanced functionality

# LangChain API Key (Optional, for LangSmith tracing)
# Get from: https://smith.langchain.com/
LANGCHAIN_API_KEY=your_langchain_api_key_here

# Additional Search Engines (Optional alternatives to Tavily)
BING_API_KEY=your_bing_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
SERPAPI_API_KEY=your_serpapi_api_key_here

# Configuration Settings
RETRIEVER=tavily
LLM_PROVIDER=openai
FAST_LLM=openai:gpt-4o-mini
SMART_LLM=openai:gpt-4o
TEMPERATURE=0.4
MAX_ITERATIONS=3
VERBOSE=true

# Performance Settings for Optimized Researcher
MAX_SEARCH_RESULTS_PER_QUERY=5
TOTAL_WORDS=1200
TOKEN_LIMIT=2000
"""
    
    env_file = Path('.env')
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_template)
        print(f"✅ Created .env template at {env_file.absolute()}")
        print("📝 Please edit .env file and add your actual API keys")
    else:
        print(f"⚠️ .env file already exists at {env_file.absolute()}")
    
    return env_file

def load_environment():
    """Load environment variables from .env file."""
    env_file = Path('.env')
    if not env_file.exists():
        print(f"❌ .env file not found. Please run create_env_template() first.")
        return False
    
    # Load environment variables
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if not value.startswith('your_') and value:  # Skip template placeholders
                    os.environ[key] = value
    
    return True

def test_api_keys():
    """Test if the required API keys are properly configured."""
    print("\n🔍 Testing API Key Configuration...")
    
    required_keys = {
        'OPENAI_API_KEY': 'OpenAI API (Required for LLM)',
        'TAVILY_API_KEY': 'Tavily API (Required for web search)'
    }
    
    optional_keys = {
        'LANGCHAIN_API_KEY': 'LangChain API (Optional)',
        'BING_API_KEY': 'Bing Search API (Optional)',
        'GOOGLE_API_KEY': 'Google Search API (Optional)'
    }
    
    all_good = True
    
    # Test required keys
    print("\n📋 Required API Keys:")
    for key, description in required_keys.items():
        value = os.getenv(key)
        if value and not value.startswith('your_') and len(value) > 10:
            print(f"✅ {key}: Configured ({description})")
        else:
            print(f"❌ {key}: Missing or invalid ({description})")
            all_good = False
    
    # Test optional keys
    print("\n📋 Optional API Keys:")
    for key, description in optional_keys.items():
        value = os.getenv(key)
        if value and not value.startswith('your_') and len(value) > 10:
            print(f"✅ {key}: Configured ({description})")
        else:
            print(f"⚪ {key}: Not configured ({description})")
    
    return all_good

def test_openai_connection():
    """Test OpenAI API connection."""
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        print("\n🤖 Testing OpenAI API connection...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'API connection successful'"}],
            max_tokens=10
        )
        
        if response.choices[0].message.content:
            print("✅ OpenAI API: Connection successful!")
            return True
        else:
            print("❌ OpenAI API: Unexpected response")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI API: Connection failed - {str(e)}")
        return False

def test_tavily_connection():
    """Test Tavily API connection."""
    try:
        import requests
        
        print("\n🔍 Testing Tavily API connection...")
        api_key = os.getenv('TAVILY_API_KEY')
        
        response = requests.post(
            "https://api.tavily.com/search",
            headers={'Content-Type': 'application/json'},
            json={
                "api_key": api_key,
                "query": "test query",
                "max_results": 1
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Tavily API: Connection successful!")
            return True
        else:
            print(f"❌ Tavily API: Connection failed - Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Tavily API: Connection failed - {str(e)}")
        return False

def main():
    """Main setup routine."""
    print("🚀 GPT-Researcher Environment Setup")
    print("=" * 50)
    
    # Step 1: Create .env template
    print("\n📝 Step 1: Setting up environment template...")
    env_file = create_env_template()
    
    # Step 2: Load environment
    print("\n🔧 Step 2: Loading environment variables...")
    if not load_environment():
        print("❌ Failed to load environment. Please check your .env file.")
        return False
    
    # Step 3: Test API keys
    print("\n🔍 Step 3: Testing API configuration...")
    keys_ok = test_api_keys()
    
    if not keys_ok:
        print("\n⚠️ Please update your .env file with valid API keys and run this script again.")
        print(f"📁 Edit: {env_file.absolute()}")
        return False
    
    # Step 4: Test connections
    print("\n🌐 Step 4: Testing API connections...")
    openai_ok = test_openai_connection()
    tavily_ok = test_tavily_connection()
    
    if openai_ok and tavily_ok:
        print("\n🎉 SUCCESS! All API connections are working.")
        print("✅ Your environment is ready for GPT-Researcher!")
        return True
    else:
        print("\n❌ Some API connections failed. Please check your keys.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Ready to run Procore disruption analysis!")
        sys.exit(0)
    else:
        print("\n🔧 Please fix the issues above and try again.")
        sys.exit(1) 