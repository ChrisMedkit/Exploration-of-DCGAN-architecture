"""Front end (the proposal's "type text, get an image" demo): Gradio app.

    python app.py      # opens http://127.0.0.1:7860

TODO (Plan step 12), needs `gradio` (see OUTLINE.md, Environment):
  import gradio as gr
  G = generate.load_generator(<best checkpoint from reports/results.csv>)
  clip_model, _ = text_encoder.load_clip()
  def run(prompt, n, seed):
      imgs = generate.generate(prompt, G, clip_model, n, seed)
      return torchvision.utils.make_grid(data.to_image_range(imgs), nrow=4).permute(1, 2, 0).numpy()
  gr.Interface(run,
               [gr.Textbox(label="Describe a flower"), gr.Slider(1, 16, 8, step=1, label="Images"),
                gr.Number(0, label="Seed")],
               gr.Image(label="Generated (64x64)"),
               examples=[["this flower has large pink petals with a yellow center", 8, 0]]).launch()
Keep the demo honest: show 64x64 output and note in the UI that the model only knows flowers.
"""


def main() -> None:
    raise NotImplementedError("Milestone 4 (Plan step 12)")


if __name__ == "__main__":
    main()
