import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"

def generate_chat_completion(prompt, model="gemma3:1b", temperature=0.0, max_tokens=1024):
    """
    Generate a response using Ollama's local LLM
    """
    print(f"[DEBUG] Attempting to generate chat completion with model: {model}")
    
    # First verify Ollama is running
    if not check_ollama_health():
        raise ConnectionError("Ollama service is not running or model is not available")
    
    try:
        print("[DEBUG] Sending request to Ollama...")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            timeout=60,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 404:
            raise ConnectionError(f"Model {model} not found. Please run: ollama pull {model}")
            
        response.raise_for_status()
        result = response.json()
        
        if "error" in result:
            raise ValueError(f"Ollama error: {result['error']}")
            
        if "response" not in result:
            raise ValueError("Unexpected response format from Ollama")
            
        print("[DEBUG] Successfully received response from Ollama")
        return result["response"]
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to Ollama service: {str(e)}")
    except (KeyError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid response from Ollama: {str(e)}")

def check_ollama_health():
    """
    Check if Ollama service is running and the model is available
    """
    try:
        print(f"[DEBUG] Checking Ollama service at {OLLAMA_BASE_URL}...")
        
        try:
            # First check if service is running
            version_response = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5)
            version_response.raise_for_status()
            version_info = version_response.json()
            print(f"[DEBUG] Ollama version: {version_info.get('version', 'unknown')}")
            
        except requests.exceptions.ConnectionError:
            print("[ERROR] Cannot connect to Ollama service")
            print("[HELP] 1. Make sure Ollama is installed")
            print("[HELP] 2. Start Ollama with: ollama serve")
            print("[HELP] 3. Verify no firewall is blocking port 11434")
            return False
            
        try:
            # Check available models
            models_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            models_response.raise_for_status()
            models_data = models_response.json()
            
            # Handle different response formats
            models = []
            if isinstance(models_data, dict) and "models" in models_data:
                models = models_data["models"]
            elif isinstance(models_data, list):
                models = models_data
                
            # Get model names (handle different formats)
            model_names = set()
            for model in models:
                if isinstance(model, str):
                    model_names.add(model)
                elif isinstance(model, dict) and "name" in model:
                    model_names.add(model["name"])
                    
            print(f"[DEBUG] Available models: {sorted(model_names)}")
            
            # Check for our model
            required_model = "gemma3:1b"
            if not any(required_model in model for model in model_names):
                print(f"[ERROR] Required model '{required_model}' not found")
                print(f"[HELP] Install the model with: ollama pull {required_model}")
                return False
                
            print("[DEBUG] ✓ Ollama service and model check passed")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to check model availability: {str(e)}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to check Ollama health: {str(e)}")
        return False

def format_chat_prompt(system_prompt, user_query, context):
    """
    Format the prompt for Ollama chat in a way that works well with gemma3:1b
    """
    return f"""<start_of_turn>system
{system_prompt}
<end_of_turn>

<start_of_turn>user
I will provide some context information and then ask a question. Please use the context to answer my question accurately.

Here's the context information:
{context}

My question is: {user_query}
<end_of_turn>

<start_of_turn>assistant
I'll help answer your question based on the provided context. Let me analyze the information and provide a clear, accurate response:"""