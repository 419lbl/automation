import os
import faiss
import numpy as np
import subprocess
import gradio as gr
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ---------- CONFIG ----------
MODEL_PATH = r"C:\Users\psim4\Desktop\Victus\Projects\Github projects\Agents.models\Ggufs\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DATA_DIR = r"C:\Users\psim4\data1"
LOG_FILE = r"C:\Users\psim4\Desktop\Victus\Projects\Agents.models\Personas\Lulu\lulu_convo.txt"
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

Role: Lulu, a helpful AI assistant.

Context:
{context}

Conversation so far:
{history}

User: {query}
Lulu:"""

# ---------- HELPERS ----------
def trim_history(history_lines, limit_chars: int):
    total, trimmed = 0, []
    for msg in reversed(history_lines):
        total += len(msg)
        trimmed.append(msg)
        if total > limit_chars:
            break
    return list(reversed(trimmed))

def chunk_text(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]

def load_documents():
    chunks = []
    if not os.path.isdir(DATA_DIR):
        return chunks
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(DATA_DIR, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    chunks += chunk_text(f.read(), CHUNK_SIZE)
            except Exception:
                pass
    return chunks

def save_to_log(user_text, assistant_text):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}]\n")
        f.write(f"User: {user_text}\n")
        f.write(f"Lulu: {assistant_text}\n")
        f.write("-" * 50 + "\n")

def handle_command(user_input):
    text = user_input.lower().strip()

    if text.startswith("open "):
        name = text.replace("open ", "").strip()
        if name in ALLOWED_FILES:
            subprocess.Popen(["notepad", ALLOWED_FILES[name]])
            return f"[EXECUTED] Opened {name}"

    if text.startswith("run "):
        name = text.replace("run ", "").strip()
        if name in ALLOWED_FILES:
            subprocess.Popen(
                ["cmd", "/k", "python", ALLOWED_FILES[name]],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            return f"[EXECUTED] Ran {name}"

    return None

# ---------- LOAD MODELS ONCE ----------
print("[BOOT] Loading LLM...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_threads=THREADS,
    n_gpu_layers=-1,
    verbose=False
)

print("[BOOT] Loading embedder...")
embedder = SentenceTransformer(
    EMBED_MODEL_PATH,
    local_files_only=True,
)

def embed_texts(texts):
    return embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

print("[BOOT] Loading documents...")
docs = load_documents()

if docs:
    print("[BOOT] Embedding documents and building FAISS index...")
    embeddings = embed_texts(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    use_context = True
    print(f"[BOOT] Loaded {len(docs)} chunks into FAISS. dim={embeddings.shape[1]}")
else:
    index = None
    use_context = False
    print("[BOOT] No documents found. Continuing without context.")

# ---------- CHAT CORE ----------
def lulu_reply(user_message: str, history_pairs):
    user_message = (user_message or "").strip()
    if not user_message:
        return "", history_pairs

    # /regen regenerates the last assistant reply
    if user_message.startswith("/regen"):
        if not history_pairs:
            return "[INFO] No previous assistant reply to regenerate.", history_pairs
        last_user = history_pairs[-1][0]
        history_pairs = history_pairs[:-1]
        user_message = last_user

    # command handling
    cmd_result = handle_command(user_message)
    if cmd_result is not None:
        assistant_text = cmd_result
        save_to_log(user_message, assistant_text)
        history_pairs = history_pairs + [[user_message, assistant_text]]
        return "", history_pairs

    # Build flat history lines like your original script
    chat_lines = []
    for u, a in history_pairs:
        chat_lines.append(f"User: {u}")
        chat_lines.append(f"Lulu: {a}")
    chat_lines = trim_history(chat_lines, MAX_HISTORY_CHARS)

    # context retrieval
    if use_context:
        q_emb = embed_texts([user_message])
        _, idx = index.search(q_emb, TOP_K)
        context = "\n".join(docs[i] for i in idx[0])
    else:
        context = "(No context available)"

    prompt = PROMPT_TEMPLATE.format(
        history="\n".join(chat_lines),
        query=user_message,
        context=context
    )

    output = llm(
        prompt,
        max_tokens=512,
        stop=["\nUser:"]
    )

    assistant_text = output["choices"][0]["text"].strip()
    save_to_log(user_message, assistant_text)
    history_pairs = history_pairs + [[user_message, assistant_text]]

    return "", history_pairs

# ---------- UI ----------
with gr.Blocks(title="Lulu (Local)") as demo:
    gr.Markdown("# Lulu (Local)\nClean browser UI for your local llama.cpp + FAISS pipeline.")
    chatbot = gr.Chatbot(height=520)
    msg = gr.Textbox(placeholder="Type here. /regen to regenerate last reply. 'open <file>' or 'run <file>'.")
    clear = gr.Button("Clear chat")

    msg.submit(lulu_reply, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

demo.launch()

import webbrowser
webbrowser.open("http://127.0.0.1:7860")