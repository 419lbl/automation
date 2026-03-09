import torch
import requests
import re
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline

# ==========================
# CONFIGURATION
# ==========================
MODEL_PATH = r"C:\Users\psim4\Desktop\Victus\1-Victus_backup\Victus\Projects\phi-3 Agent"  # Local model path
FINVIZ_API_KEY = "1d5ae8aa-5414-4fd5-8c08-bb12373c501d"       # Your Finviz Elite API key

# Finviz base URLs
FINVIZ_STOCK_ENDPOINT = "https://elite.finviz.com/export.ashx?v=152&c=5,6,7,82,26,28,30,31,84,38,55,56,57,58,125,126,76,67,86,87,88,65,66"     # Single stock
FINVIZ_SCREENER_ENDPOINT = "https://elite.finviz.com/export.ashx?v=152&c=5,6,7,82,26,28,30,31,84,38,55,56,57,58,125,126,76,67,86,87,88,65,66"  # Screener

# ==========================
# STREAMLIT PAGE CONFIG
# ==========================
st.set_page_config(page_title="Lulu — Local AI + Finviz", page_icon="💹")

st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #111;
    color: #33ccff;
    border-right: 2px solid #ff7b00;
}
.chat-container {
    padding: 10px;
    border-radius: 12px;
}
.user-bubble {
    background-color: #222;
    padding: 8px 12px;
    border-radius: 12px;
    margin-bottom: 6px;
}
.lulu-bubble {
    background-color: #1a1a1a;
    padding: 8px 12px;
    border-left: 4px solid #ff7b00;
    border-radius: 12px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD LOCAL MODEL
# ==========================
@st.cache_resource
def load_model():
    print("Loading local model... please wait")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.9,
        do_sample=True
    )

    llm = HuggingFacePipeline(pipeline=text_gen)
    return model, tokenizer, text_gen, llm


model, tokenizer, text_gen, llm = load_model()

# ==========================
# FINVIZ TOOL FUNCTION
# ==========================
def get_stock_data(symbol: str, view: int = 111):
    """
    Clean Markdown-formatted Finviz output.
    """
    try:
        url = f"https://elite.finviz.com/export.ashx?v={view}&t={symbol.upper()}&auth={FINVIZ_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        lines = response.content.decode("utf-8").strip().splitlines()
        if len(lines) < 2:
            return f"⚠️ No data found for {symbol.upper()}."

        headers = [h.strip() for h in lines[0].split(",")]
        values = [v.strip() for v in lines[1].split(",")]

        # Build Markdown table
        table = "| Field | Value |\n|:------|:------|\n"
        for h, v in zip(headers, values):
            table += f"| **{h}** | {v} |\n"

        return f"### 📈 {symbol.upper()} — Finviz Summary\n\n" + table

    except Exception as e:
        return f"⚠️ Error fetching data: {e}"



# ==========================
# DECIDE ACTION
# ==========================
def decide_action(user_input: str):
    """Detect if user wants a stock quote or screener list."""
    text = user_input.lower()

    # Screener mode
    if any(word in text for word in ["top", "list", "filter", "high dividend", "cheap", "under", "above", "high yield"]):
        filters = "fa_div_high"  # Example screener filter (can customize)
        return "fetch_screener", None, filters

    # Stock mode
    if any(word in text for word in ["stock", "ticker", "quote", "price", "market"]):
        tickers = re.findall(r"\b[A-Za-z]{1,5}\b", user_input)
        blacklist = {"stock", "share", "price", "market", "for", "about", "get", "the"}
        tickers = [t.upper() for t in tickers if t.lower() not in blacklist]
        if tickers:
            return "fetch_stock", tickers[-1], None

    return "chat", None, None

# ==========================
# STREAMLIT APP UI
# ==========================
st.title("💬 Lulu — Local AI Assistant + Finviz Elite Integration")
st.write("Chat naturally — Lulu will call Finviz automatically when needed.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.experimental_rerun()
    st.info("Enjoy.\nModel: Phi-3.")

# Display conversation
for user, assistant in st.session_state.history:
    st.markdown(f"<div class='user-bubble'><b>You:</b> {user}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='lulu-bubble'><b>Lulu:</b> {assistant}</div>", unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    st.markdown(f"<div class='user-bubble'><b>You:</b> {user_input}</div>", unsafe_allow_html=True)

    action, symbol, filters = decide_action(user_input)

    if action == "fetch_stock" and symbol:
        with st.spinner(f"Fetching stock data for {symbol}..."):
            finviz_data = get_stock_data(symbol=symbol)
        st.markdown(f"<div class='lulu-bubble'>{finviz_data}</div>", unsafe_allow_html=True)
        st.session_state.history.append((user_input, finviz_data))

    elif action == "fetch_screener" and filters:
        with st.spinner("Fetching top screener results..."):
            finviz_data = get_stock_data(filters=filters)
        st.markdown(f"<div class='lulu-bubble'>{finviz_data}</div>", unsafe_allow_html=True)
        st.session_state.history.append((user_input, finviz_data))

    else:
        with st.spinner("Lulu is thinking..."):
            messages = [{"role": "system", "content": (
                "You are Lulu, a friendly, grounded AI assistant. "
                "Speak naturally. If the user asks about stocks, refer to the fetched Finviz data."
            )}]
            messages.append({"role": "user", "content": user_input})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            output = text_gen(prompt)
            response = output[0]["generated_text"][len(prompt):].strip()

        st.markdown(f"<div class='lulu-bubble'><b>Lulu:</b> {response}</div>", unsafe_allow_html=True)
        st.session_state.history.append((user_input, response))
