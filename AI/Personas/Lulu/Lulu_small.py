import os
import faiss
import subprocess
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ---------- CONFIG ----------
MODEL_PATH = r"C:\Users\psim4\Desktop\Victus\Projects\Github projects\Agents.models\Ggufs\qwen2.5-7b-instruct-q4_k_m.gguf"
DATA_DIR = r"C:\Users\psim4\data1"
LOG_FILE = r"C:\Users\psim4\Desktop\Victus\Projects\Github projects\Agents.models\Personas\Lulu\lulu_convo.txt"

CHUNK_SIZE = 500
TOP_K = 3
MAX_HISTORY_CHARS = 3000
CTX_SIZE = 4096
THREADS = 8

ALLOWED_FILES = {
    "scrape3.py": r"C:\Users\psim4\wscripts\scrape3.py",
    "rag4.py": r"C:\Users\psim4\wscripts\rag4.py"
}

EMBED_MODEL_PATH = (
    r"C:\Users\psim4\.cache\huggingface\hub"
    r"\models--sentence-transformers--all-MiniLM-L6-v2"
    r"\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)

PROMPT_TEMPLATE = """You are having a live conversation with a human.

Role: Lulu, a helpful AI coding assistant.
Always format code inside markdown code blocks with the correct language.
Preserve indentation and formatting exactly.
Keep responses concise and to the point.

help user get android app onto samsung s25 device.

Context:
{context}

Conversation so far:
{history}

User: {query}
Lulu:"""

CHAT_HISTORY = []

# ---------- INITIALIZE ----------
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_threads=THREADS,
    n_gpu_layers=-1,
    verbose=False
)

embedder = SentenceTransformer(
    EMBED_MODEL_PATH,
    local_files_only=True,
)

# ---------- HELPERS ----------
def trim_history(history, limit):
    total = 0
    trimmed = []
    for msg in reversed(history):
        total += len(msg)
        trimmed.append(msg)
        if total > limit:
            break
    return list(reversed(trimmed))


def chunk_text(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


def load_documents():
    chunks = []
    if not os.path.exists(DATA_DIR):
        return chunks

    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(DATA_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                chunks.extend(chunk_text(f.read(), CHUNK_SIZE))
    return chunks


def embed_texts(texts):
    return embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")


def handle_command(user_input):
    text = user_input.lower().strip()

    if text.startswith("open "):
        name = text.replace("open ", "").strip()
        if name in ALLOWED_FILES:
            subprocess.Popen(["notepad", ALLOWED_FILES[name]])
            return "(command executed: opened file)"

    if text.startswith("run "):
        name = text.replace("run ", "").strip()
        if name in ALLOWED_FILES:
            subprocess.Popen(
                ["cmd", "/k", "python", ALLOWED_FILES[name]],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            return "(command executed: ran file)"

    return None


def save_to_log(user_text, assistant_text):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}]\n")
        f.write(f"User: {user_text}\n")
        f.write(f"Lulu: {assistant_text}\n")
        f.write("-" * 50 + "\n")


def build_faiss_index():
    print("Loading documents...")
    docs = load_documents()

    if not docs:
        print("[INFO] No documents found. Continuing without context.")
        return docs, None, False

    print("Embedding documents...")
    embeddings = embed_texts(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print(f"[INFO] Loaded {len(docs)} document chunks into FAISS.")
    return docs, index, True


DOCS, INDEX, USE_CONTEXT = build_faiss_index()


def get_context(query):
    if USE_CONTEXT and INDEX is not None and DOCS:
        q_emb = embed_texts([query])
        _, idx = INDEX.search(q_emb, TOP_K)
        return "\n".join(DOCS[i] for i in idx[0])
    return "(No context available)"


def generate_reply(query):
    global CHAT_HISTORY

    command_result = handle_command(query)
    if command_result is not None:
        CHAT_HISTORY += [f"User: {query}", f"Lulu: {command_result}"]
        save_to_log(query, command_result)
        return command_result

    CHAT_HISTORY = trim_history(CHAT_HISTORY, MAX_HISTORY_CHARS)
    context = get_context(query)

    prompt = PROMPT_TEMPLATE.format(
        history="\n".join(CHAT_HISTORY),
        query=query,
        context=context
    )

    output = llm(
        prompt,
        max_tokens=512,
        stop=["\nUser:"],
        temperature=0.7
    )

    reply = output["choices"][0]["text"].strip()
    CHAT_HISTORY += [f"User: {query}", f"Lulu: {reply}"]
    save_to_log(query, reply)
    return reply


# ---------- MAIN LOOP ----------
print("\nLulu is ready.")
print("Type 'exit' or 'quit' to close.")
print("Commands:")
print("  open scrape3.py")
print("  run scrape3.py")
print("  /regen")

while True:
    query = input("\nYou: ").strip()

    if not query:
        continue

    if query.lower() in ("exit", "quit"):
        break

    if query == "/regen":
        if len(CHAT_HISTORY) < 2 or not CHAT_HISTORY[-2].startswith("User: "):
            print("\n[INFO] No previous reply to regenerate.")
            continue

        last_user_query = CHAT_HISTORY[-2].replace("User: ", "", 1)

        if CHAT_HISTORY[-1].startswith("Lulu: "):
            CHAT_HISTORY.pop()

        query = last_user_query
        print("\n[INFO] Regenerating last reply...")

    reply = generate_reply(query)
    print(f"\nLulu: {reply}")