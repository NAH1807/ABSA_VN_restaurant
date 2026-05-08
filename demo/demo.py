import os
import sys
import gradio as gr
import pandas as pd

# Add src directory to path for imports
# Since demo.py is in demo/ subdirectory, go up one level to reach src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference.absa_pipeline import (
    ABSAPipeline
)


# =========================
# LOAD PIPELINE
# =========================

print("Loading ABSA Pipeline...")

pipeline = ABSAPipeline()

print("Pipeline loaded successfully!")


# =========================
# PREDICT FUNCTION
# =========================

def analyze_review(text):

    # Empty input
    if text.strip() == "":

        return pd.DataFrame([
            {
                "Aspect": "",
                "Aspect Term": "",
                "Sentiment": ""
            }
        ])

    # Predict
    results = pipeline.predict(text)

    # No aspect found
    if len(results) == 0:

        return pd.DataFrame([
            {
                "Aspect": "None",
                "Aspect Term": "No aspect detected",
                "Sentiment": "None"
            }
        ])

    # Format results
    formatted = []

    for item in results:

        formatted.append({

            "Aspect":
                item["aspect"],

            "Aspect Term":
                item["term"],

            "Sentiment":
                item["sentiment"]
        })

    return pd.DataFrame(formatted)


# =========================
# EXAMPLES
# =========================

EXAMPLES = [

    [
        "Đồ ăn ngon nhưng phục vụ chậm"
    ],

    [
        "Không gian đẹp nhưng giá hơi đắt"
    ],

    [
        "Nhân viên thân thiện và đồ uống rất ngon"
    ],

    [
        "Quán đông và hơi ồn"
    ],

    [
        "Giá hợp lý, đồ ăn ổn"
    ]
]


# =========================
# CUSTOM CSS
# =========================

CUSTOM_CSS = """
footer {
    visibility: hidden;
}

.gradio-container {
    font-family: 'Arial';
}

h1 {
    text-align: center;
}
"""


# =========================
# BUILD UI
# =========================

with gr.Blocks(
    css=CUSTOM_CSS,
    theme=gr.themes.Soft()
) as demo:

    # Title
    gr.Markdown(
        """
        # Vietnamese ABSA System
        ### PhoBERT + BiLSTM + CRF
        
        Aspect-Based Sentiment Analysis for Vietnamese Restaurant Reviews
        """
    )

    # Input section
    with gr.Row():

        review_input = gr.Textbox(
            label="Input Review",
            placeholder="Nhập đánh giá nhà hàng...",
            lines=5
        )

    # Buttons
    with gr.Row():

        analyze_button = gr.Button(
            "Analyze",
            variant="primary"
        )

        clear_button = gr.Button(
            "Clear"
        )

    # Output
    result_table = gr.Dataframe(
        headers=[
            "Aspect",
            "Aspect Term",
            "Sentiment"
        ],
        label="ABSA Results",
        interactive=False
    )

    # Examples
    gr.Examples(
        examples=EXAMPLES,
        inputs=review_input
    )

    # Button actions
    analyze_button.click(
        fn=analyze_review,
        inputs=review_input,
        outputs=result_table
    )

    clear_button.click(
        fn=lambda: ("", None),
        inputs=[],
        outputs=[
            review_input,
            result_table
        ]
    )

    # Footer
    gr.Markdown(
        """
        ---
        Developed using:
        
        - PhoBERT
        - BiLSTM
        - CRF
        - VLSP2018-ABSA Dataset
        """
    )


# =========================
# LAUNCH
# =========================

if __name__ == "__main__":

    demo.launch(
        share=True
    )