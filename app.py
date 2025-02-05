import gradio as gr
import asyncio
from research import research_retrieval

def run_research(user_query: str, iteration_limit: int = 10) -> str:
    try:
        report = asyncio.run(research_retrieval(user_query, iteration_limit))
        return report
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# OpenDeepResearcher")
    gr.Markdown("Enter your research query and optionally adjust the iteration limit (default is 10).")

    with gr.Row():
        query_input = gr.Textbox(
            lines=3,
            placeholder="Enter your research query/topic here...",
            label="Research Query"
        )
        iter_input = gr.Number(value=10, label="Iteration Limit", precision=0)

    output_text = gr.Textbox(lines=20, label="Final Report")
    run_button = gr.Button("Run Research")
    run_button.click(fn=run_research, inputs=[query_input, iter_input], outputs=output_text)

if __name__ == "__main__":
    demo.launch()
