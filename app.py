import gradio as gr
from transformers import pipeline

# Load model
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

# Summarization Function
def generate_summary(article, max_len, min_len):

    if not article.strip():
        return "Please enter an article."

    if min_len >= max_len:
        return "Min Length should be less than Max Length."

    summary = summarizer(
        article,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )

    return summary[0]["summary_text"]

# Custom CSS for Modern UI
custom_css = """
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.gradio-container {
    font-family: 'Poppins', sans-serif;
    max-width: 1100px !important;
    margin: auto;
    padding-top: 20px;
}

.main-container {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 25px;
    padding: 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}

textarea {
    border-radius: 18px !important;
    border: none !important;
    padding: 15px !important;
    font-size: 16px !important;
    background: rgba(255,255,255,0.95) !important;
}

button {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 15px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    padding: 14px !important;
    transition: 0.3s ease;
}

button:hover {
    transform: scale(1.03);
    opacity: 0.9;
}

.slider {
    padding-top: 10px;
}

.footer {
    text-align: center;
    margin-top: 20px;
    color: #94a3b8;
    font-size: 14px;
}
"""

# Gradio UI
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div class="main-container">
        <div class="title">
            AI Article Summarizer
        </div>

        <div class="subtitle">
            Summarize long articles instantly using Hugging Face Transformers
        </div>
    """)

    with gr.Row():

        with gr.Column(scale=1):

            article_input = gr.Textbox(
                lines=18,
                placeholder="Paste your article here...",
                label="Input Article"
            )

            with gr.Row():

                max_len = gr.Slider(
                    minimum=50,
                    maximum=300,
                    value=120,
                    step=10,
                    label="Max Length"
                )

                min_len = gr.Slider(
                    minimum=20,
                    maximum=150,
                    value=50,
                    step=10,
                    label="Min Length"
                )

            summarize_btn = gr.Button("Generate Summary")

        with gr.Column(scale=1):

            summary_output = gr.Textbox(
                lines=18,
                label="Generated Summary"
            )

    summarize_btn.click(
        fn=generate_summary,
        inputs=[article_input, max_len, min_len],
        outputs=summary_output
    )

    gr.HTML("""
        <div class="footer">
            Built with Hugging Face Transformers • Gradio • Python
        </div>
    </div>
    """)

# Launch App
demo.launch()   