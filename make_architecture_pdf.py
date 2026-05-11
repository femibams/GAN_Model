"""Generate a student-friendly PDF explaining the architecture in this repo.

Run:  python make_architecture_pdf.py
Output: architecture_explained.pdf
"""
from fpdf import FPDF


PRIMARY = (30, 60, 110)      # navy
ACCENT  = (200, 90, 40)      # warm orange
INK     = (35, 35, 40)
MUTED   = (110, 110, 120)
BOX_BG  = (242, 246, 252)
RULE    = (210, 215, 225)


class Doc(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*MUTED)
        self.cell(0, 8, "GAN Architecture, Explained Simply", align="L")
        self.set_x(-30)
        self.cell(0, 8, f"{self.page_no() - 1}", align="R")
        self.set_draw_color(*RULE)
        self.line(15, 18, 195, 18)
        self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, "Built from models.py, train.py, config.py", align="C")

    # ---- Content helpers --------------------------------------------------
    def H1(self, text):
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*PRIMARY)
        self.multi_cell(0, 10, text)
        self.ln(2)

    def H2(self, text):
        self.ln(3)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*PRIMARY)
        self.multi_cell(0, 8, text)
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.6)
        x = self.get_x()
        y = self.get_y()
        self.line(x + 15, y, x + 35, y)
        self.set_line_width(0.2)
        self.ln(3)

    def H3(self, text):
        self.ln(1)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*ACCENT)
        self.multi_cell(0, 6, text)
        self.ln(0.5)

    def P(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*INK)
        self.multi_cell(0, 5.6, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*INK)
        x = self.get_x()
        y = self.get_y()
        self.set_xy(x + 4, y)
        self.cell(4, 5.6, "-")
        self.set_xy(x + 8, y)
        self.multi_cell(0, 5.6, text)

    def callout(self, title, body):
        self.ln(1)
        x0 = self.get_x()
        y0 = self.get_y()
        # Reserve box width
        w = 180
        # Estimate height by rendering into a temp position
        self.set_font("Helvetica", "B", 11)
        title_h = 6
        self.set_font("Helvetica", "", 10.5)
        # Crude height estimate
        approx_lines = max(1, int(len(body) / 95) + body.count("\n"))
        body_h = approx_lines * 5.2 + 4
        h = title_h + body_h + 4

        self.set_fill_color(*BOX_BG)
        self.set_draw_color(*RULE)
        self.rect(x0, y0, w, h, style="DF")
        # Accent stripe
        self.set_fill_color(*ACCENT)
        self.rect(x0, y0, 1.2, h, style="F")

        self.set_xy(x0 + 5, y0 + 2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*PRIMARY)
        self.cell(w - 8, title_h, title)
        self.set_xy(x0 + 5, y0 + 2 + title_h)
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(*INK)
        self.multi_cell(w - 8, 5.2, body)
        # Move below box
        self.set_y(y0 + h + 2)

    def code(self, text):
        self.set_font("Courier", "", 10)
        self.set_text_color(*PRIMARY)
        self.set_fill_color(*BOX_BG)
        self.multi_cell(0, 5, text, fill=True)
        self.set_text_color(*INK)
        self.ln(1)

    # ---- Diagrams ---------------------------------------------------------
    def pipeline_diagram(self):
        """Top-down pipeline of the generator and discriminator."""
        self.ln(2)
        x0 = 18
        y0 = self.get_y()
        box_w = 38
        box_h = 12
        gap = 6

        def draw_box(x, y, w, h, label, sub=None, color=PRIMARY, fill=BOX_BG):
            self.set_draw_color(*color)
            self.set_fill_color(*fill)
            self.set_line_width(0.4)
            self.rect(x, y, w, h, style="DF")
            self.set_xy(x, y + (1 if sub else 3))
            self.set_text_color(*color)
            self.set_font("Helvetica", "B", 9.5)
            self.cell(w, 5, label, align="C")
            if sub:
                self.set_xy(x, y + 6)
                self.set_text_color(*MUTED)
                self.set_font("Helvetica", "", 8)
                self.cell(w, 4, sub, align="C")

        def arrow(x1, y1, x2, y2):
            self.set_draw_color(*ACCENT)
            self.set_line_width(0.5)
            self.line(x1, y1, x2, y2)
            # arrow head
            self.line(x2, y2, x2 - 2, y2 - 1.5)
            self.line(x2, y2, x2 - 2, y2 + 1.5)

        # Row 1: z -> mapping -> w
        draw_box(x0,                 y0, box_w, box_h, "z (noise)", "256-dim")
        draw_box(x0 + box_w + gap,   y0, box_w, box_h, "Mapping FC", "8 layers")
        draw_box(x0 + 2*(box_w+gap), y0, box_w, box_h, "w (style)", "256-dim")
        draw_box(x0 + 3*(box_w+gap), y0, box_w, box_h, "Synthesis", "4x4 -> 128x128", color=ACCENT)

        for i in range(3):
            arrow(x0 + (i+1)*box_w + i*gap, y0 + box_h/2,
                  x0 + (i+1)*box_w + (i+1)*gap, y0 + box_h/2)

        # Row 2: synthesis blocks
        y1 = y0 + box_h + 14
        block_w = 26
        block_h = 11
        labels = ["4x4", "8x8", "16x16", "32x32", "64x64", "128x128"]
        for i, lab in enumerate(labels):
            x = x0 + i * (block_w + 4)
            draw_box(x, y1, block_w, block_h, lab, "block", color=ACCENT)
            if i > 0:
                arrow(x - 4, y1 + block_h/2, x, y1 + block_h/2)

        # Label above the row
        self.set_xy(x0, y1 - 6)
        self.set_text_color(*MUTED)
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 4, "Synthesis blocks (each one doubles the resolution):")

        # Row 3: output image -> Discriminator -> real/fake score
        y2 = y1 + block_h + 14
        draw_box(x0,                 y2, box_w, box_h, "Fake image", "128x128 RGB")
        draw_box(x0 + box_w + gap,   y2, box_w, box_h, "Real image", "from FFHQ", fill=(252, 245, 240))
        draw_box(x0 + 2*(box_w+gap), y2, box_w, box_h, "Discriminator", "downsamples")
        draw_box(x0 + 3*(box_w+gap), y2, box_w, box_h, "Real or fake?", "logit score", color=ACCENT)

        # Both feed into D
        arrow(x0 + box_w, y2 + 3,
              x0 + 2*(box_w+gap), y2 + 3)
        arrow(x0 + box_w + gap + box_w, y2 + box_h - 3,
              x0 + 2*(box_w+gap), y2 + box_h - 3)
        arrow(x0 + 3*box_w + 2*gap, y2 + box_h/2,
              x0 + 3*(box_w+gap), y2 + box_h/2)

        self.set_y(y2 + block_h + 8)
        self.set_text_color(*MUTED)
        self.set_font("Helvetica", "I", 9)
        self.multi_cell(0, 4, "Figure 1. End-to-end pipeline. The Generator turns random noise "
                              "into a face. The Discriminator scores faces as real or fake.")
        self.ln(3)


def build():
    pdf = Doc(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    # ===================== Cover =====================
    pdf.add_page()
    pdf.ln(30)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 30)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(180, 14, "GAN Architecture")
    pdf.ln(14)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*ACCENT)
    pdf.cell(180, 12, "Explained Simply")
    pdf.ln(16)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*INK)
    pdf.multi_cell(180, 6,
        "A student-friendly walkthrough of the small-scale StyleGAN2 face "
        "generator in this repository. We answer: what each piece does, "
        "why it is there, and how the pieces fit together to turn random "
        "numbers into pictures of faces.")
    pdf.ln(10)

    # Hero box
    pdf.set_fill_color(*BOX_BG)
    pdf.set_draw_color(*RULE)
    pdf.rect(15, pdf.get_y(), 180, 46, style="DF")
    pdf.set_fill_color(*ACCENT)
    pdf.rect(15, pdf.get_y(), 1.2, 46, style="F")
    pdf.set_xy(20, pdf.get_y() + 3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(0, 6, "What you will learn")
    pdf.set_xy(20, pdf.get_y() + 6)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*INK)
    items = [
        "What a GAN is, in plain language with an analogy.",
        "The Generator: how noise becomes a face, layer by layer.",
        "The Discriminator: how it judges real vs. fake.",
        "How the two networks learn together (the training loop).",
        "The 'extras' that keep training stable: R1, path length, EMA.",
        "What changed from the earlier model that wasn't training, and why.",
    ]
    for it in items:
        y = pdf.get_y()
        pdf.set_xy(20, y)
        pdf.cell(4, 5.6, "-")
        pdf.set_xy(24, y)
        pdf.multi_cell(166, 5.6, it)
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5,
        "No prior GAN experience required. Some familiarity with neural "
        "networks (layers, weights, training) helps, but is not assumed.")

    # ===================== 1. The big idea =====================
    pdf.add_page()
    pdf.H1("1. The Big Idea")
    pdf.P(
        "A Generative Adversarial Network (GAN) is two neural networks "
        "playing a game against each other.")
    pdf.callout(
        "Analogy: the forger and the detective",
        "Picture an art forger trying to paint convincing fake portraits, "
        "and a detective trying to catch the forgeries. Every time the "
        "detective spots a fake, the forger learns from the mistake and "
        "paints better. Every time the forger fools the detective, the "
        "detective also learns and gets sharper. After many rounds, both "
        "get very good at their jobs. The forger is our Generator. The "
        "detective is our Discriminator.")
    pdf.H3("In this project specifically")
    pdf.P(
        "We are training on the FFHQ dataset (faces of people). The goal "
        "is for the Generator to learn to draw realistic 128x128 face "
        "images from pure random noise. The Discriminator's only job is "
        "to score images: 'this looks real' versus 'this looks fake'.")
    pdf.P(
        "The architecture is a smaller, single-GPU version of StyleGAN2 "
        "(Karras et al., 2020). 'Smaller' means we shrink the channel "
        "widths so it trains in a few hours on one GPU instead of a "
        "cluster, but the design choices match the paper.")

    pdf.H2("The whole pipeline at a glance")
    pdf.pipeline_diagram()

    # ===================== 2. The Generator =====================
    pdf.add_page()
    pdf.H1("2. The Generator: Noise to Face")
    pdf.P(
        "The Generator's job is simple to state: take a vector of random "
        "numbers and produce a 128x128 RGB image that looks like a face. "
        "It does this in three stages.")

    pdf.H2("Stage 1. The Mapping Network (z to w)")
    pdf.P(
        "We start with a noise vector z of length 256, drawn from a "
        "normal distribution. Different z vectors should produce "
        "different faces.")
    pdf.P(
        "But raw noise is a clumsy input. StyleGAN2's insight is to first "
        "pass z through 8 small fully-connected layers to get a new "
        "vector w, also of length 256. We call w the 'style' vector. "
        "The space of w is more disentangled than the space of z, "
        "meaning each direction in w tends to control one visual "
        "attribute (e.g. hair colour, age, pose).")
    pdf.callout(
        "Why this helps",
        "Imagine z is a pile of unsorted Lego pieces and w is a tidy "
        "drawer where each compartment holds one type of piece. It is "
        "easier to build something specific from the tidy drawer. "
        "That is what the mapping network gives us.")
    pdf.H3("In code")
    pdf.code("class MappingNetwork(nn.Module):  # models.py\n"
             "    # 8 layers of EqualizedLinear + LeakyReLU\n"
             "    # PixelNorm applied to z first\n"
             "    def forward(self, z):\n"
             "        return self.net(self.pixel_norm(z))")

    pdf.H2("Stage 2. The Synthesis Network (w to image)")
    pdf.P(
        "The synthesis network does NOT start from the noise vector. "
        "It starts from a small 4x4 'learned constant' image: a tensor "
        "of weights that the model trains alongside everything else. "
        "Think of it as a shared starting canvas, the same for every "
        "image we generate.")
    pdf.P(
        "We then walk up the resolution ladder, doubling the size each "
        "time:")
    for line in ["4x4  -> 8x8  -> 16x16  -> 32x32  -> 64x64  -> 128x128"]:
        pdf.set_font("Courier", "B", 11)
        pdf.set_text_color(*ACCENT)
        pdf.cell(0, 6, "    " + line)
        pdf.ln(8)
    pdf.set_text_color(*INK)
    pdf.P(
        "At each resolution there is a 'synthesis block' that does two "
        "things: it upsamples the feature map (the internal picture is "
        "now twice as big) and applies two 3x3 convolutions to refine "
        "it. The style vector w controls every one of these "
        "convolutions.")

    pdf.H3("How does w 'control' the convolution?")
    pdf.P(
        "This is the StyleGAN2 magic, called modulated convolution with "
        "weight demodulation. In a normal convolution, the filter "
        "weights are fixed. Here, before the convolution runs, we "
        "rescale each input channel by a number derived from w. "
        "Different w produces a different filter, so different style.")
    pdf.P(
        "After the rescaling, the filter is normalised again so its "
        "outputs do not blow up. This 'demodulation' replaces the "
        "instance-norm trick from StyleGAN1 and is faster.")

    pdf.add_page()

    pdf.H3("Noise injection: the small random details")
    pdf.P(
        "After every convolution, we add a single channel of fresh "
        "Gaussian noise, scaled by a learned weight. This lets the "
        "Generator paint stochastic details (individual hairs, freckles, "
        "stray pixels) without needing to encode every single one in w.")

    pdf.H3("Skip-to-RGB: every resolution contributes")
    pdf.P(
        "At every resolution we also project the feature map down to a "
        "3-channel image (the 'ToRGB' branch). We upsample the previous "
        "RGB image and add it on top. So the final picture is a sum of "
        "contributions from 4x4, 8x8, ..., 128x128. Lower resolutions "
        "draw the rough shape; higher resolutions add the fine detail.")
    pdf.callout(
        "Mental model",
        "Painting a portrait: first the sketch (4x4 block), then "
        "blocking in colours (8x8, 16x16), then features and shading "
        "(32x32, 64x64), then individual hairs and skin texture "
        "(128x128). Each layer adds its own contribution to the final "
        "image instead of overwriting the previous one.")

    # ===================== 3. The Discriminator =====================
    pdf.add_page()
    pdf.H1("3. The Discriminator: The Detective")
    pdf.P(
        "The Discriminator is essentially a classifier. It takes a "
        "128x128 image and outputs a single number: high means 'I think "
        "this is real', low means 'I think this is fake'.")
    pdf.P(
        "Structurally, it is the mirror image of the Generator. While "
        "the Generator climbs UP from 4x4 to 128x128, the Discriminator "
        "climbs DOWN from 128x128 to 4x4 using residual blocks that "
        "halve the resolution. At 4x4 it has a tiny but information-rich "
        "feature map, which it then flattens into a single score.")
    pdf.H3("Residual blocks")
    pdf.P(
        "Each downsample step has two paths: a skip path that just "
        "average-pools, and a main path with two 3x3 convolutions. "
        "Their outputs are added. Residual connections make the gradient "
        "flow cleanly through deep stacks, so the network trains "
        "stably.")

    pdf.H3("MinibatchStdDev: a clever trick against 'mode collapse'")
    pdf.callout(
        "What is mode collapse?",
        "If the Generator finds ONE face that fools the Discriminator, "
        "it might give up and produce the same face every time. That is "
        "called mode collapse and it is the most common GAN failure "
        "mode. We need a defense.")
    pdf.P(
        "Just before the final layer, the Discriminator computes the "
        "standard deviation of features across the batch and appends it "
        "as a new feature channel. If the Generator is producing nearly "
        "identical images, this stddev channel is small, and the "
        "Discriminator can use that as a tell: 'low diversity -> "
        "probably fake'. So mode-collapsed batches are easy to spot, "
        "which pushes the Generator to keep producing varied outputs.")

    # ===================== 4. Training =====================
    pdf.add_page()
    pdf.H1("4. How They Learn Together")
    pdf.P(
        "Training alternates between updating the Discriminator and "
        "updating the Generator. One iteration of the loop in train.py "
        "looks like this:")
    pdf.code(
        "for step in range(total_steps):\n"
        "    # 1. Discriminator step\n"
        "    real = next batch from FFHQ\n"
        "    fake = G(random z)\n"
        "    update D to push D(real) up and D(fake) down\n\n"
        "    # 2. Generator step\n"
        "    fake = G(random z)\n"
        "    update G to push D(fake) up\n\n"
        "    # 3. Slowly update G_ema (smoothed copy of G)")
    pdf.P(
        "The clever part is the loss function: we use the "
        "'non-saturating logistic' loss. In plain language: the "
        "Discriminator is rewarded for confident correct answers, and "
        "the Generator is rewarded specifically for making the "
        "Discriminator say 'real' about its fakes. We avoid older losses "
        "that produced near-zero gradients early in training.")

    pdf.H2("Three stabilisers you should know about")

    pdf.H3("R1 gradient penalty")
    pdf.P(
        "Every 16 steps, on real images only, we compute the gradient "
        "of the Discriminator output with respect to the input pixels "
        "and penalise its squared length. This stops the Discriminator "
        "from becoming too confident too quickly, which would starve "
        "the Generator of useful gradients.")

    pdf.H3("Path-length regularisation")
    pdf.P(
        "Every 4 steps, we measure how much the output image changes "
        "when we wiggle w by a tiny amount, and push that magnitude "
        "toward a moving average. This keeps the mapping from w to "
        "image smooth, so small style changes produce small image "
        "changes. Smooth latent spaces are easier to train and easier "
        "to interpolate in.")

    pdf.H3("Generator EMA (Exponential Moving Average)")
    pdf.P(
        "We keep a SECOND copy of the Generator, G_ema, whose weights "
        "are a slow-moving average of the live Generator's weights. "
        "Sample images and final inference both use G_ema, not G. The "
        "average smooths out the noisy step-to-step weight updates and "
        "produces visibly cleaner faces.")

    pdf.H2("Practical settings (config.py)")
    pdf.P(
        "All hyperparameters live in config.py. The defaults are tuned "
        "for one mid-range GPU (~12 GB):")
    pdf.code(
        "IMAGE_SIZE        = 128       # output resolution\n"
        "NOISE_SIZE        = 256       # length of z\n"
        "W_DIM             = 256       # length of w\n"
        "MAPPING_LAYERS    = 8\n"
        "CHANNEL_BASE      = 8192      # NF schedule (smaller = lighter)\n"
        "CHANNEL_MAX       = 256\n"
        "BATCH_SIZE        = 16\n"
        "TOTAL_STEPS       = 150_000\n"
        "LR_G = LR_D       = 2.5e-3\n"
        "LAMBDA_R1         = 1.0       # R1 strength\n"
        "LAMBDA_PL         = 2.0       # path-length strength\n"
        "EMA_KIMG          = 10.0      # G_ema decay")

    # ===================== 5. What changed =====================
    pdf.add_page()
    pdf.H1("5. What I Changed (and Why It Helped)")
    pdf.P(
        "The first version of this model didn't train well: it was "
        "unstable, slow, and the samples never sharpened up. The current "
        "architecture is the result of stripping it back to something "
        "smaller and more honest. This section is the diff, in plain "
        "language.")

    pdf.callout(
        "TL;DR",
        "The old model was a text-conditioned face generator with "
        "bounding-box layout control and TWO discriminators. It was "
        "trying to learn too many things at once. The new one drops "
        "text, drops bounding boxes, drops one of the discriminators, "
        "and switches to the cleaner FFHQ dataset. Less to balance "
        "means more capacity left for actually drawing faces.")

    pdf.H2("Side-by-side")
    rows = [
        ("Conditioning",   "z + CLIP text emb + bbox mask",     "z only (unconditional)"),
        ("Discriminators", "Two: full-image + face crop",       "One: full-image"),
        ("Spectral norm",  "On every D conv",                   "None (R1 alone)"),
        ("Extra losses",   "ROI loss + leakage loss + warmup",  "Adversarial loss only"),
        ("Augmentation",   "Adaptive Data Aug (ADA)",           "hflip + small crop jitter"),
        ("Path length",    "Disabled (broken with text input)", "Enabled, every 4 steps"),
        ("Dataset",        "CelebA",                            "FFHQ"),
        ("Schedule",       "300 epochs + LR decay",             "150k iters, fixed LR"),
        ("Synth layers",   "2 per resolution block",            "1 + 1 (upsample + refine)"),
        ("Block layout",   "Hard-coded for 128x128",            "Parameterised by resolution"),
    ]
    # Header row
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*PRIMARY)
    pdf.set_fill_color(*BOX_BG)
    pdf.cell(38, 7, "  Aspect", border=0, fill=True)
    pdf.cell(72, 7, "  Old (didn't work)", border=0, fill=True)
    pdf.cell(70, 7, "  New (works)", border=0, fill=True, ln=1)
    pdf.set_draw_color(*RULE)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*INK)
    for asp, old, new in rows:
        pdf.set_x(15)
        pdf.cell(38, 6.4, "  " + asp)
        pdf.set_text_color(*MUTED)
        pdf.cell(72, 6.4, "  " + old)
        pdf.set_text_color(*ACCENT)
        pdf.cell(70, 6.4, "  " + new, ln=1)
        pdf.set_text_color(*INK)
    pdf.ln(2)

    pdf.H2("The five changes that mattered most")

    pdf.H3("1. Removed text and bbox conditioning")
    pdf.P(
        "The old generator took three inputs: noise z, a CLIP text "
        "embedding, and a bounding-box mask injected at every "
        "resolution. The discriminator also used a 'projection' head "
        "that scored text-image alignment.")
    pdf.P(
        "The problem: the model was trying to learn TWO hard things at "
        "the same time. (a) draw a realistic face, and (b) make sure "
        "that face matched the text and lived inside the bbox. Until "
        "(a) was working, (b) had nothing to align to. The text "
        "projection also had a numerical issue documented in the old "
        "config: it amplified the R1 gradient by ~500x at "
        "initialisation, which destabilised early training.")
    pdf.P(
        "Fix: drop text, drop bbox. Make it an unconditional face GAN. "
        "Once an unconditional model works, conditioning can be added "
        "back as a separate project.")

    pdf.add_page()
    pdf.H3("2. Went from two discriminators to one")
    pdf.P(
        "The old setup had a 'global' discriminator that judged the "
        "whole image and a separate 'RoI' discriminator that judged a "
        "64x64 face crop. The motivation was to pressure the generator "
        "into making the face region especially sharp.")
    pdf.P(
        "In practice this meant THREE adversarial losses to balance: "
        "G vs. D_global, G vs. D_roi, plus the relative weight between "
        "them. There was also a 'leakage loss' nudging the background "
        "outside the bbox toward zero, with a hand-tuned warm-up "
        "schedule. Every extra loss is one more thing that can fight "
        "the others.")
    pdf.P(
        "Fix: single discriminator, single adversarial loss. The R1 "
        "penalty alone is enough to keep D from running away.")

    pdf.H3("3. Removed spectral normalisation from D")
    pdf.P(
        "Spectral normalisation (SN) and the R1 gradient penalty are "
        "both ways to stop the discriminator from becoming too sharp. "
        "Stacking both is double regularisation: D ends up too weak to "
        "give G a meaningful learning signal early on.")
    pdf.P(
        "Fix: keep R1 (it's the StyleGAN2 default), drop SN. The "
        "discriminator now actually does its job.")

    pdf.H3("4. Switched from CelebA to FFHQ")
    pdf.P(
        "CelebA is older, lower-resolution, and the alignment is "
        "approximate. FFHQ is curated specifically for face GANs: high "
        "resolution, good alignment, diverse demographics, and "
        "well-licensed.")
    pdf.P(
        "Fix: switched the data loader to FFHQ thumbnails (128x128). "
        "Same training code, dramatically higher-quality samples.")

    pdf.H3("5. Re-enabled path-length regularisation")
    pdf.P(
        "Path length pushes G to have a smooth latent-to-image map, "
        "which makes both training and interpolation behave better. "
        "The old config disabled it with the comment: 'PL w.r.t. raw "
        "noise z is ill-defined when z feeds two paths' - the two "
        "paths being z and text both flowing into the mapping network.")
    pdf.P(
        "Fix: removing text input also removed the dual-path issue, so "
        "PL now works exactly as in the StyleGAN2 reference.")

    pdf.H2("Smaller cleanups that also helped")
    items = [
        ("Iteration-based training",
         "Old: 300 epochs with linear LR decay. New: 150k iterations "
         "with fixed LR. Iterations make sample/checkpoint scheduling "
         "cleaner and remove the 'last epoch is short' artifact."),
        ("Parameterised block schedule",
         "Old: each of the 5 resolution blocks hard-coded as a "
         "named attribute (stem, up1, ..., up5). New: a single loop "
         "that builds blocks from log2(image_size). Switching to 64x64 "
         "or 256x256 is now a config change, not a code change."),
        ("Cached blur kernels",
         "The [1,2,1] FIR blur used during upsampling is now cached "
         "per (channels, device, dtype). Small change, noticeably "
         "faster synthesis."),
        ("Removed Adaptive Data Augmentation (ADA)",
         "ADA helps small datasets, but FFHQ is large enough that "
         "horizontal flip plus a tiny crop jitter is plenty. One "
         "fewer moving part."),
    ]
    for name, desc in items:
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*ACCENT)
        pdf.cell(0, 6, "  " + name, ln=1)
        pdf.set_x(15)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*INK)
        pdf.multi_cell(180, 5.6, "    " + desc)
        pdf.ln(1)

    pdf.callout(
        "The lesson",
        "Most of the wins came from REMOVING things, not adding them. "
        "When a GAN won't train, the instinct is to add another loss "
        "or another regulariser to fix the symptom. The opposite often "
        "works better: rip out anything that isn't pulling its weight, "
        "make the simplest unconditional version train cleanly, and "
        "only then add complexity back on a working baseline.")

    # ===================== 6. Cheat sheet =====================
    pdf.add_page()
    pdf.H1("6. Cheat Sheet")
    pdf.P("If you only remember a few things, remember these.")

    pdf.H3("The five pieces")
    items = [
        ("Mapping network",   "8-layer FC. Turns raw noise z into a tidy style vector w."),
        ("Synthesis network", "Climbs from 4x4 to 128x128. w controls every conv."),
        ("Modulated conv",    "Conv whose weights are rescaled by w, then re-normalised."),
        ("Discriminator",     "Mirror of synthesis. Scores images as real or fake."),
        ("Training tricks",   "R1 + path length + EMA keep training stable and smooth."),
    ]
    for name, desc in items:
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*PRIMARY)
        pdf.cell(45, 6, "  " + name)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*INK)
        pdf.multi_cell(135, 6, desc)
    pdf.ln(2)

    pdf.H3("Glossary")
    glossary = [
        ("Latent",       "A vector of numbers that represents an image abstractly. "
                         "Here z (raw) and w (styled) are both latents."),
        ("Logit",        "The Discriminator's raw output BEFORE turning it into a "
                         "probability. Higher logit means 'more real'."),
        ("Demodulation", "Re-normalising filter weights after multiplying them "
                         "by w, so output magnitudes do not explode."),
        ("R1",           "A penalty on the Discriminator's gradient on real images. "
                         "Stops it from over-fitting to real samples."),
        ("EMA",          "Exponential moving average. A smoothed copy of the "
                         "Generator weights, used at inference time."),
        ("Mode collapse","When the Generator produces only a few variations. The "
                         "MinibatchStdDev layer fights this."),
    ]
    for term, defn in glossary:
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(*ACCENT)
        pdf.cell(34, 5.6, "  " + term)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(*INK)
        pdf.multi_cell(146, 5.6, defn)

    pdf.H3("Where to look in the code")
    where = [
        ("config.py",   "All the knobs (image size, batch, learning rate)."),
        ("models.py",   "Generator, Discriminator, and all the building blocks."),
        ("dataset.py",  "FFHQ data loading and augmentation."),
        ("train.py",    "The training loop, losses, and checkpointing."),
        ("infer.py",    "Loads a checkpoint and generates samples."),
        ("utils.py",    "Loss functions and the image-grid saver."),
    ]
    for f, d in where:
        pdf.set_x(15)
        pdf.set_font("Courier", "B", 10.5)
        pdf.set_text_color(*PRIMARY)
        pdf.cell(34, 5.6, "  " + f)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(*INK)
        pdf.multi_cell(146, 5.6, d)

    pdf.ln(4)
    pdf.set_draw_color(*RULE)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5,
        "Reference: Karras et al., 'Analyzing and Improving the Image "
        "Quality of StyleGAN' (CVPR 2020). The original paper, if you "
        "want the deep dive after this overview.")

    out = "architecture_explained.pdf"
    pdf.output(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
