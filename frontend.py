import time

import gradio as gr
import httpx


def analyze(input_text, progress=gr.Progress()):
    progress(0, desc="Analyzing...")
    input_text = input_text.strip()
    score = 0

    if not input_text:
        return [
            (input_text, "error"),
        ], score

    with httpx.Client(base_url="http://localhost:8080") as client:
        resp = client.post("/analyze/send", json={"text": input_text})
        resp_data = resp.json()
        task_id = resp_data["task_id"]
        status = resp_data["task_status"]

        while status not in ("failure", "success"):
            resp = client.get(f"/analyze/status/{task_id}")
            resp.raise_for_status()
            resp_data = resp.json()
            status = resp_data["task_status"]
            progress.update()
            time.sleep(0.5)

        res_label = resp_data.get("task_result", "error")

        if isinstance(res_label, dict):
            score = res_label["score"]
            res_label = res_label["label"]

        return [
            (input_text, res_label),
        ], score


def main():
    with gr.Blocks() as demo:
        with gr.Column():
            input_text = gr.Textbox(show_copy_button=True, max_lines=1, label="Input text")

            with gr.Row():
                out_text = gr.HighlightedText(
                    label="Result",
                    combine_adjacent=True,
                    show_legend=True,
                    color_map={
                        "negative": "red",
                        "positive": "green",
                        "neutral": "yellow",
                        "error": "gray",
                    },
                )
                score = gr.Number(label="Score", precision=3)
        bt = gr.Button("Analyze")
        bt.click(analyze, inputs=input_text, outputs=[out_text, score])

    demo.queue(max_size=3)
    demo.launch(max_threads=5)


if __name__ == "__main__":
    main()
