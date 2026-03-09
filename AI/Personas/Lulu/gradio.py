from llama_cpp import Llama
import gradio as gr

MODEL_PATH = r"C:\Users\psim4\Desktop\Victus\Projects\Github projects\Agents.models\Ggufs\qwen2.5-coder-7b-instruct-q4_k_m.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=20,
    verbose=False,
)

SYSTEM_PROMPT = """You are a helpful coding assistant.
Always put code inside fenced markdown code blocks with the correct language.
Preserve indentation exactly.
Format code cleanly and do not flatten whitespace.
"""

def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for item in history:
            role = item.get("role")
            content = item.get("content", "")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    reply = ""

    stream = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        token = delta.get("content", "")
        if token:
            reply += token
            yield reply

css = """
pre, code {
    white-space: pre-wrap !important;
    tab-size: 4;
    font-family: Consolas, 'Courier New', monospace;
}
"""

demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Local GGUF Chat",
    textbox=gr.Textbox(
        lines=8,
        placeholder="Ask for code here...",
        show_label=False
    ),
    css=css,
)

demo.queue()
demo.launch(inbrowser=True)