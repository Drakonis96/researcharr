import requests
import json
from typing import List, Dict, Any, Optional

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Researcharr"
        }
        
    def fetch_models(self) -> Dict[str, Any]:
        """Fetch available models from OpenRouter"""
        response = requests.get(
            f"{self.base_url}/models",
            headers=self.headers,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        providers = {}
        for model in data.get("data", []):
            provider_id = model["id"].split("/")[0]
            if provider_id not in providers:
                providers[provider_id] = {
                    "id": provider_id,
                    "name": provider_id,
                    "models": []
                }
            providers[provider_id]["models"].append({
                "id": model["id"],
                "name": model.get("name", model["id"]),
                "contextLength": model.get("context_length", 0),
                "promptPrice": float(model.get("pricing", {}).get("prompt", "0")),
                "completionPrice": float(model.get("pricing", {}).get("completion", "0")),
                "isEmbedding": "embedding" in model["id"].lower()
            })
            
        return {"providers": providers, "allModels": data.get("data", [])}
        
    def create_embedding(self, model: str, text: str) -> List[float]:
        """Create embedding for a text"""
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json={"model": model, "input": text},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
        
    def chat_completion(self, model: str, messages: List[Dict[str, str]], 
                       max_tokens: int = 4096, temperature: float = 0.7) -> Dict[str, Any]:
        """Send chat completion request"""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
            "model": data.get("model", model)
        }
        
    def stream_chat_completion(self, model: str, messages: List[Dict[str, str]],
                               max_tokens: int = 4096, temperature: float = 0.7):
        """Stream chat completion"""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            },
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except:
                        pass
