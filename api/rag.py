# Vercel Serverless Function for RAG (Retrieval-Augmented Generation)
# Uses Pinecone for vector search + OpenAI embeddings

from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request

# Configuration - Multiple Pinecone indexes for different knowledge bases
# Each index has its own host. Format: PINECONE_HOST_{INDEX_NAME_UPPERCASE}
# e.g., PINECONE_HOST_NUTRITION_RAG, PINECONE_HOST_RAG_ROMEO_AND_JULIET
DEFAULT_INDEX = "nutrition-rag"


def get_pinecone_host(index_name: str) -> str:
    """Get the Pinecone host for a given index name."""
    # Try index-specific environment variable first
    env_key = f"PINECONE_HOST_{index_name.upper().replace('-', '_')}"
    host = os.environ.get(env_key)
    if host:
        return host
    # Fallback to default PINECONE_HOST for backwards compatibility
    return os.environ.get("PINECONE_HOST", "")


def get_embedding(text: str, api_key: str) -> list:
    """Get embedding from OpenAI API."""
    request_body = {
        "model": "text-embedding-3-small",
        "input": text
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps(request_body).encode('utf-8'),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )

    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        return data["data"][0]["embedding"]


def search_pinecone(query_vector: list, pinecone_key: str, index_name: str = DEFAULT_INDEX, n_results: int = 5) -> list:
    """Search Pinecone for similar vectors."""
    pinecone_host = get_pinecone_host(index_name)
    if not pinecone_host:
        raise ValueError(f"Pinecone host not configured for index: {index_name}")

    request_body = {
        "vector": query_vector,
        "topK": n_results,
        "includeMetadata": True
    }

    req = urllib.request.Request(
        f"https://{pinecone_host}/query",
        data=json.dumps(request_body).encode('utf-8'),
        headers={
            "Content-Type": "application/json",
            "Api-Key": pinecone_key
        }
    )

    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        return data.get("matches", [])


def search_docs(query: str, openai_key: str, pinecone_key: str, index_name: str = DEFAULT_INDEX, n_results: int = 5) -> list:
    """Search a knowledge base using vector similarity."""
    # Get query embedding from OpenAI
    query_vector = get_embedding(query, openai_key)

    # Search Pinecone
    matches = search_pinecone(query_vector, pinecone_key, index_name, n_results)

    # Convert to our format
    results = []
    for match in matches:
        metadata = match.get("metadata", {})
        results.append({
            "text": metadata.get("text", ""),
            "title": metadata.get("title", "Unknown Title"),
            "url": metadata.get("url", ""),
            "score": match.get("score", 0)
        })

    return results


def format_context(results: list) -> str:
    """Format search results into context string."""
    if not results:
        return "No relevant articles found."

    context_parts = []
    for i, doc in enumerate(results, 1):
        title = doc.get("title", "Unknown Title")
        url = doc.get("url", "")
        text = doc.get("text", "")

        context_parts.append(f"{i}. Title: '{title}' (URL: {url}):\n\t- {text}")

    return "\n\n".join(context_parts)


def call_llm(messages: list, system_prompt: str, api_key: str, model: str = "gpt-4o-mini", provider: str = "openai") -> str:
    """Call LLM API (OpenAI or Anthropic)."""

    if provider == "anthropic":
        request_body = {
            "model": model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages
        }

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(request_body).encode('utf-8'),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
        )

        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data["content"][0]["text"]
    else:
        # OpenAI
        openai_messages = [{"role": "system", "content": system_prompt}] + messages

        request_body = {
            "model": model,
            "max_tokens": 4096,
            "messages": openai_messages
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(request_body).encode('utf-8'),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )

        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data["choices"][0]["message"]["content"]


# System prompt for RAG (used when retrieveOnly=False)
RAG_SYSTEM_PROMPT = """You are a helpful assistant with access to a knowledge base of documents.

When answering questions:
1. Base your answers on the provided context from retrieved documents
2. Cite the source titles/URLs when referencing specific information
3. If the context doesn't contain relevant information, say so and offer general guidance
4. Be accurate and don't make claims not supported by the provided sources
5. If multiple sources have different perspectives, acknowledge this"""


class handler(BaseHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Health check endpoint."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "ok",
            "backend": "pinecone",
            "message": "RAG API (Pinecone + OpenAI embeddings) - supports multiple knowledge bases"
        }).encode())

    def do_POST(self):
        def send_json_response(status_code, data):
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
            body = json.loads(post_data.decode('utf-8'))
        except (json.JSONDecodeError, ValueError) as e:
            send_json_response(400, {"error": f"Invalid request: {str(e)}"})
            return

        query = body.get("query", "")
        messages = body.get("messages", [])
        provider = body.get("provider", "openai")
        model = body.get("model", "gpt-4o-mini")
        api_key = body.get("apiKey")
        access_code = body.get("accessCode")
        retrieve_only = body.get("retrieveOnly", False)
        n_results = body.get("nResults", 5)
        index_name = body.get("indexName", DEFAULT_INDEX)  # Which Pinecone index to search
        rag_source = body.get("ragSource", "nutrition")  # For display purposes

        # Get API keys
        openai_key = None
        pinecone_key = os.environ.get("PINECONE_API_KEY")

        if not pinecone_key:
            send_json_response(500, {"error": "PINECONE_API_KEY not configured"})
            return

        if api_key:
            openai_key = api_key
        elif access_code:
            valid_code = os.environ.get("ACCESS_CODE")
            if access_code != valid_code:
                send_json_response(401, {"error": "Invalid access code"})
                return
            openai_key = os.environ.get("OPENAI_API_KEY")
        else:
            send_json_response(400, {"error": "No API key provided"})
            return

        try:
            # Search for relevant documents in the specified index
            results = search_docs(query, openai_key, pinecone_key, index_name, n_results)
            context = format_context(results)

            # If retrieve_only, just return the context
            if retrieve_only:
                send_json_response(200, {
                    "context": context,
                    "query": query,
                    "n_results": len(results),
                    "indexName": index_name,
                    "ragSource": rag_source
                })
                return

            # Build augmented message
            user_message = f"""[RAG Context from {rag_source} Knowledge Base]
{context}
[End RAG Context]

User Question: {query}"""

            augmented_messages = messages + [{"role": "user", "content": user_message}]

            # Call LLM
            response_text = call_llm(augmented_messages, RAG_SYSTEM_PROMPT, openai_key, model, provider)

            send_json_response(200, {
                "content": response_text,
                "context": context,
                "query": query
            })

        except Exception as e:
            send_json_response(500, {"error": str(e)})
