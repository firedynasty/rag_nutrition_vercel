# Vercel Serverless Function for RAG (Retrieval-Augmented Generation)
# Uses OpenAI embeddings (lightweight, no heavy ML dependencies)

from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request

# LanceDB import
try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

# Configuration
LANCEDB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_nutrition/databases/my_lancedb")
TABLE_NAME = "nutrition_openai"  # From rag_config.toml


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


def search_nutrition_docs(query: str, api_key: str, n_results: int = 5) -> list:
    """Search the nutrition knowledge base using vector similarity."""
    if not LANCEDB_AVAILABLE:
        return []

    # Get query embedding
    query_vector = get_embedding(query, api_key)

    # Search LanceDB
    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table(TABLE_NAME)

    results = table.search(query_vector).limit(n_results).to_list()

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


# System prompt for nutrition RAG
RAG_SYSTEM_PROMPT = """You are a helpful nutrition and health assistant with access to a knowledge base of nutrition research articles.

When answering questions:
1. Base your answers on the provided context from retrieved articles
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
            "lancedb_available": LANCEDB_AVAILABLE,
            "message": "Nutrition RAG API (OpenAI embeddings)"
        }).encode())

    def do_POST(self):
        self.send_cors_headers()

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            body = json.loads(post_data.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            return

        query = body.get("query", "")
        messages = body.get("messages", [])
        provider = body.get("provider", "openai")
        model = body.get("model", "gpt-4o-mini")
        api_key = body.get("apiKey")
        access_code = body.get("accessCode")
        retrieve_only = body.get("retrieveOnly", False)
        n_results = body.get("nResults", 5)

        # Get API key
        if not api_key:
            if access_code:
                valid_code = os.environ.get("ACCESS_CODE")
                if access_code != valid_code:
                    self.send_response(401)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Invalid access code"}).encode())
                    return
                api_key = os.environ.get("OPENAI_API_KEY")
            else:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No API key provided"}).encode())
                return

        if not LANCEDB_AVAILABLE:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "LanceDB not available"}).encode())
            return

        try:
            # Search for relevant documents
            results = search_nutrition_docs(query, api_key, n_results)
            context = format_context(results)

            # If retrieve_only, just return the context
            if retrieve_only:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "context": context,
                    "query": query,
                    "n_results": len(results)
                }).encode())
                return

            # Build augmented message
            user_message = f"""[RAG Context from Nutrition Knowledge Base]
{context}
[End RAG Context]

User Question: {query}"""

            augmented_messages = messages + [{"role": "user", "content": user_message}]

            # Call LLM
            response_text = call_llm(augmented_messages, RAG_SYSTEM_PROMPT, api_key, model, provider)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "content": response_text,
                "context": context,
                "query": query
            }).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
