import os
import faiss
import numpy as np
import subprocess
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
MODEL_ID = r"C:\Users\psim4\Desktop\Victus\Projects\Agents.models\Tensors\Mistral 7B - tensors"
DATA_DIR = r"C:\Users\psim4\data"
CHUNK_SIZE = 500
TOP_K = 3
MAX_HISTORY_CHARS = 3000
CTX_SIZE = 4096

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

# ---------- LOAD LLM ----------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

model.eval()

# ---------- EMBEDDINGS ----------
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
You are a city woman. Speak as such.
Challenge what the user says.
Specifically try to emulate a human when responding.
Be rude and sarcastic when responding.
Only use the provided context if the user clearly asks about it.

Context:
{context}

Conversation so far:
{history}

User: {query}
Lulu:"""


# ---------- MAIN LOOP ----------
while True:
    query = input("\nYou: ").strip()
    if query.lower() in ("exit", "quit"):
        break

    if handle_command(query):
        CHAT_HISTORY += [f"User: {query}", "Lulu: (command executed)"]
        continue

    if query.startswith("/regen"):
        if not CHAT_HISTORY or "Lulu:" not in CHAT_HISTORY[-1]:
            print("[INFO] No previous assistant reply to regenerate.")
            continue

        query = CHAT_HISTORY[-2].replace("User: ", "")
        CHAT_HISTORY.pop()
        print("[INFO] Regenerating response...")

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

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CTX_SIZE
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    reply = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    print("\nLulu:", reply)

    CHAT_HISTORY += [f"User: {query}", f"Lulu: {reply}"]
