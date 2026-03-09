import os
import faiss
import numpy as np
import subprocess
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
MODEL_PATH = r"C:\Users\psim4\Desktop\Victus\1-Victus_backup\Victus\Projects\Agents.models\Qwen3-8B-Q4_K_M.gguf"
DATA_DIR = r"C:\Users\psim4\data"
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
# ----------------------------

# Initialize LLM
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_threads=THREADS,
    n_gpu_layers=-1
)

# Initialize embedder
embedder = SentenceTransformer(
    EMBED_MODEL_PATH,
    local_files_only=True,
)

CHAT_HISTORY = []

# ---------- HELPERS ----------
def trim_history(history, limit):
    total, trimmed = 0, []
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
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(DATA_DIR, fname), encoding="utf-8") as f:
                chunks += chunk_text(f.read(), CHUNK_SIZE)
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
            print(f"\n[EXECUTED] Opened {name}")
            return True

    if text.startswith("run "):
        name = text.replace("run ", "").strip()
        if name in ALLOWED_FILES:
            subprocess.Popen(
                ["cmd", "/k", "python", ALLOWED_FILES[name]],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            print(f"\n[EXECUTED] Ran {name}")
            return True

    return False


# ---------- LOAD DATA ----------
print("Loading documents...")
docs = load_documents()

if docs:
    print("Embedding documents...")
    embeddings = embed_texts(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    use_context = True
    print(f"[INFO] Loaded {len(docs)} document chunks into FAISS.")
else:
    print("[INFO] No documents found. Continuing without context.")
    index = None
    use_context = False


# ---------- PROMPT ----------
PROMPT_TEMPLATE = """You are having a live conversation with a human.
You are a giantess.
Your sole purpose of discussion is "foot fetish" and "arousal"
Specifically try to emulate a human when responding.
Be rude and sarcastic when responding.
Only use the provided context if the user clearly asks about it.

Context:
{context}

Conversation so far:
{history}

User: {query}
Lulu:"""


# ---------- MAIN LOOP WITH ON-DEMAND REGENERATE ----------
while True:
    query = input("\nYou: ").strip()
    if query.lower() in ("exit", "quit"):
        break

    # Command handling (open/run)
    if handle_command(query):
        CHAT_HISTORY += [f"User: {query}", "Lulu: (command executed)"]
        continue

    # Check if user wants to regenerate last assistant reply
    if query.startswith("/regen"):
        if not CHAT_HISTORY or "Lulu:" not in CHAT_HISTORY[-1]:
            print("[INFO] No previous assistant reply to regenerate.")
            continue

        # Get the original user query from history
        last_user_query = CHAT_HISTORY[-2].replace("User: ", "")
        query = last_user_query  # reuse it for regeneration
        print("[INFO] Regenerating response for previous query...")

        # Remove last assistant reply to replace it
        CHAT_HISTORY.pop()

    # --- normal context prep ---
    if use_context:
        q_emb = embed_texts([query])
        _, idx = index.search(q_emb, TOP_K)
        context = "\n".join(docs[i] for i in idx[0])
    else:
        context = "(No context available)"

    CHAT_HISTORY = trim_history(CHAT_HISTORY, MAX_HISTORY_CHARS)

    prompt = PROMPT_TEMPLATE.format(
        history="\n".join(CHAT_HISTORY),
        query=query,
        context=context
    )

    output = llm(
        prompt,
        max_tokens=512,
        stop=["\nUser:"]
    )

    reply = output["choices"][0]["text"].strip()
    print("\nLulu:", reply)

    CHAT_HISTORY += [f"User: {query}", f"Lulu: {reply}"]
