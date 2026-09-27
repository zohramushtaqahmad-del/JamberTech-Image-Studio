import torch
import spaces
import gradio as gr
from diffusers import DiffusionPipeline

# Load the pipeline once at startup
print("Loading Z-Image-Turbo pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")
print("Pipeline loaded!")

@spaces.GPU
def generate_image(prompt, height, width, num_inference_steps, seed, randomize_seed, progress=gr.Progress(track_tqdm=True)):
    """Generate an image from the given prompt."""
    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator("cuda").manual_seed(int(seed))
    image = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    
    return image, seed

# Example prompts
examples = [
    ["Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan. Neon lightning-bolt lamp, bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda."],
    ["A majestic dragon soaring through clouds at sunset, scales shimmering with iridescent colors, detailed fantasy art style"],
    ["Cozy coffee shop interior, warm lighting, rain on windows, plants on shelves, vintage aesthetic, photorealistic"],
    ["Astronaut riding a horse on Mars, cinematic lighting, sci-fi concept art, highly detailed"],
    ["Portrait of a wise old wizard with a long white beard, holding a glowing crystal staff, magical forest background"],
    ["Futuristic cyberpunk city skyline at night, neon lights reflecting on wet streets, ultra-detailed 8K"],
]

# JamberTech custom theme - dark tech aesthetic
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ===== JAMBERTECH BRANDING ===== */
:root {
    --jt-primary: #00d4ff;
    --jt-secondary: #7b2fff;
    --jt-accent: #ff6b35;
    --jt-dark: #0a0a14;
    --jt-card: #10101e;
    --jt-border: rgba(0, 212, 255, 0.2);
    --jt-glow: 0 0 20px rgba(0, 212, 255, 0.3);
}

body, .gradio-container {
    background: var(--jt-dark) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Header banner */
.jt-header {
    background: linear-gradient(135deg, #0a0a14 0%, #10102a 50%, #0a0a14 100%);
    border-bottom: 1px solid var(--jt-border);
    padding: 1.5rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.jt-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.05), transparent);
    animation: scanline 3s linear infinite;
}

@keyframes scanline {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.jt-logo {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #00d4ff 0%, #7b2fff 50%, #ff6b35 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}

.jt-tagline {
    color: rgba(0, 212, 255, 0.7) !important;
    font-size: 0.85rem !important;
    letter-spacing: 4px !important;
    text-transform: uppercase !important;
    margin-top: 0.2rem !important;
    font-weight: 300 !important;
}

.jt-subtitle {
    color: rgba(255,255,255,0.6) !important;
    font-size: 1rem !important;
    margin-top: 0.5rem !important;
}

/* Glowing badge */
.jt-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,47,255,0.1));
    border: 1px solid var(--jt-primary);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.75rem;
    color: var(--jt-primary);
    letter-spacing: 2px;
    text-transform: uppercase;
    box-shadow: var(--jt-glow);
    margin-top: 0.5rem;
}

/* Gradio blocks override */
.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    background: var(--jt-dark) !important;
}

/* Input panels */
.gr-block, .gr-form, .gr-box, .block {
    background: var(--jt-card) !important;
    border: 1px solid var(--jt-border) !important;
    border-radius: 12px !important;
}

label, .gr-label {
    color: rgba(255,255,255,0.85) !important;
    font-family: 'Inter', sans-serif !important;
}

textarea, input[type="text"], input[type="number"] {
    background: #0d0d1e !important;
    border: 1px solid var(--jt-border) !important;
    color: #fff !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}

textarea:focus, input:focus {
    border-color: var(--jt-primary) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.15) !important;
    outline: none !important;
}

/* Primary button */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #00d4ff, #7b2fff) !important;
    border: none !important;
    color: #fff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.25) !important;
}

button.primary:hover, .gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 212, 255, 0.45) !important;
}

/* Sliders */
input[type="range"] {
    accent-color: var(--jt-primary) !important;
}

/* Output image panel */
.gr-image, .image-container {
    background: #0d0d1e !important;
    border: 1px solid var(--jt-border) !important;
    border-radius: 12px !important;
    box-shadow: var(--jt-glow) !important;
}

/* Footer */
.jt-footer {
    text-align: center;
    padding: 1.5rem;
    border-top: 1px solid var(--jt-border);
    margin-top: 1rem;
    color: rgba(255,255,255,0.4);
    font-size: 0.82rem;
    letter-spacing: 1px;
}

.jt-footer a {
    color: var(--jt-primary) !important;
    text-decoration: none !important;
    transition: color 0.2s;
}

.jt-footer a:hover {
    color: var(--jt-secondary) !important;
}

/* Accordion */
.gr-accordion {
    border: 1px solid var(--jt-border) !important;
    background: #0d0d1e !important;
    border-radius: 8px !important;
}

/* Number input */
.gr-number {
    background: #0d0d1e !important;
    border: 1px solid var(--jt-border) !important;
    color: #fff !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--jt-dark); }
::-webkit-scrollbar-thumb { background: var(--jt-primary); border-radius: 3px; }

/* Pulse animation for generate button */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 4px 20px rgba(0, 212, 255, 0.25); }
    50% { box-shadow: 0 4px 40px rgba(0, 212, 255, 0.6); }
}

button.primary {
    animation: pulse-glow 2.5s ease-in-out infinite !important;
}

button.primary:hover {
    animation: none !important;
}
"""

# Build the Gradio interface
with gr.Blocks(css=custom_css, fill_height=True) as demo:
    
    # ===== JamberTech Header =====
    gr.HTML("""
    <div class="jt-header">
        <div class="jt-logo">⚡ JamberTech</div>
        <div class="jt-tagline">Official AI Image Studio</div>
        <div class="jt-subtitle">Ultra-fast AI image generation powered by Z-Image-Turbo</div>
        <div class="jt-badge">🚀 Generates in ~8 Steps</div>
    </div>
    """)
    
    with gr.Row(equal_height=False):
        # Left column — Controls
        with gr.Column(scale=1, min_width=340):
            prompt = gr.Textbox(
                label="✨ Image Prompt",
                placeholder="Describe the image you want to create...\ne.g. 'A cyberpunk cityscape at night, neon lights, ultra-detailed 8K'",
                lines=5,
                max_lines=10,
                autofocus=True,
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    height = gr.Slider(
                        minimum=512, maximum=2048, value=1024, step=64,
                        label="Height (px)", info="Image height in pixels"
                    )
                    width = gr.Slider(
                        minimum=512, maximum=2048, value=1024, step=64,
                        label="Width (px)", info="Image width in pixels"
                    )
                
                num_inference_steps = gr.Slider(
                    minimum=1, maximum=20, value=9, step=1,
                    label="Inference Steps",
                    info="9 steps = 8 DiT forward passes (recommended)"
                )
                
                with gr.Row():
                    randomize_seed = gr.Checkbox(label="🎲 Random Seed", value=True)
                    seed = gr.Number(label="Seed", value=42, precision=0, visible=False)
                
                def toggle_seed(randomize):
                    return gr.Number(visible=not randomize)
                
                randomize_seed.change(toggle_seed, inputs=[randomize_seed], outputs=[seed])
            
            generate_btn = gr.Button(
                "⚡ GENERATE IMAGE",
                variant="primary",
                size="lg",
                scale=1
            )
            
            gr.Examples(
                examples=examples,
                inputs=[prompt],
                label="💡 Try these prompts",
                examples_per_page=6,
            )
        
        # Right column — Output
        with gr.Column(scale=1, min_width=340):
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                format="png",
                show_label=False,
                height=620,
                buttons=["download", "share"],
            )
            used_seed = gr.Number(
                label="🎲 Seed Used",
                interactive=False,
                container=True,
            )
    
    # ===== JamberTech Footer =====
    gr.HTML("""
    <div class="jt-footer">
        ⚡ <strong style="color:rgba(255,255,255,0.7)">JamberTechOfficial</strong> AI Image Studio &nbsp;|&nbsp;
        Model: <a href="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo" target="_blank">Tongyi-MAI/Z-Image-Turbo</a> (Apache 2.0) &nbsp;|&nbsp;
        Powered by <a href="https://huggingface.co" target="_blank">🤗 HuggingFace ZeroGPU</a> &nbsp;|&nbsp;
        Built by <strong style="color:rgba(255,255,255,0.7)">JamberTech</strong>
    </div>
    """)
    
    # Connect buttons
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, num_inference_steps, seed, randomize_seed],
        outputs=[output_image, used_seed],
    )
    
    prompt.submit(
        fn=generate_image,
        inputs=[prompt, height, width, num_inference_steps, seed, randomize_seed],
        outputs=[output_image, used_seed],
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True)
