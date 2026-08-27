from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np

class Day64DiffusionModels(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # Set default colors for text and math
        Text.set_default(color=BLACK)
        Tex.set_default(color=BLACK)
        MathTex.set_default(color=BLACK)

        # ---------------------------------------------------------
        # SECTION 1: WHAT IS IT?
        # ---------------------------------------------------------

        title = Text("Diffusion Models", font_size=64, weight=BOLD).to_edge(UP)
        subtitle = Text("Forward and Reverse Process", font_size=36, color=BLUE).next_to(title, DOWN)

        with self.voiceover(text="Diffusion models represent a paradigm shift in generative artificial intelligence. These models learn to synthesize highly complex, realistic data from scratch by first mastering the process of destruction.") as tracker:
            self.play(Write(title))
            self.play(FadeIn(subtitle))

        # Forward Process Animation Setup
        square_data = Square(side_length=2.0, fill_opacity=1, fill_color=BLUE).shift(LEFT * 4 + DOWN * 1)
        original_data_label = Text("Original Data", font_size=24).next_to(square_data, DOWN)

        noisy_dots = VGroup()
        for _ in range(200):
            dot = Dot(radius=0.03, color=BLUE).move_to(square_data.get_center())
            noisy_dots.add(dot)

        with self.voiceover(text="Imagine taking a perfectly clear, high-resolution photograph and systematically, step by step, adding static noise to it until the original image is completely obliterated and all that remains is pure, structureless white noise.") as tracker:
            self.play(FadeIn(square_data), Write(original_data_label))

            # Step by step destruction
            self.play(FadeOut(square_data), FadeIn(noisy_dots), run_time=0.5)

            for _ in range(5):
                animations = []
                for dot in noisy_dots:
                    random_shift = np.array([
                        np.random.normal(0, 0.3),
                        np.random.normal(0, 0.3),
                        0
                    ])
                    # Transition to gray/black as it gets noisier
                    dot.generate_target()
                    dot.target.shift(random_shift)
                    dot.target.set_color(interpolate_color(dot.get_color(), GRAY, 0.2))
                    animations.append(MoveToTarget(dot))
                self.play(*animations, run_time=tracker.duration/6)

        forward_process_label = Text("Forward Process (Adding Noise)", font_size=24, color=RED).next_to(noisy_dots, UP).shift(UP*1)
        arrow_forward = Arrow(square_data.get_top() + UP*0.5, forward_process_label.get_left(), color=RED)

        with self.voiceover(text="This destructive phase is known as the Forward Process, or the diffusion process.") as tracker:
            self.play(Write(forward_process_label), GrowArrow(arrow_forward))

        with self.voiceover(text="Once the model has learned exactly how data degrades into noise, it is then trained to perform the exact opposite operation.") as tracker:
            self.wait(tracker.duration)

        reverse_process_label = Text("Reverse Process (Denoising)", font_size=24, color=GREEN).next_to(square_data, UP).shift(UP*1)

        with self.voiceover(text="It learns to take pure, chaotic noise and progressively denoise it, step by step, reconstructing the intricate patterns, edges, and colors until a brand new, coherent piece of data emerges.") as tracker:
            self.play(Write(reverse_process_label))

            for _ in range(5):
                animations = []
                for dot in noisy_dots:
                    # Move dots back towards square shape center, restoring color
                    dir_to_center = square_data.get_center() - dot.get_center()
                    shift_amount = dir_to_center * 0.3 + np.array([np.random.normal(0, 0.1), np.random.normal(0, 0.1), 0])
                    dot.generate_target()
                    dot.target.shift(shift_amount)
                    dot.target.set_color(interpolate_color(dot.get_color(), BLUE, 0.2))
                    animations.append(MoveToTarget(dot))
                self.play(*animations, run_time=tracker.duration/6)

            self.play(FadeOut(noisy_dots), FadeIn(square_data), run_time=1.0)

        with self.voiceover(text="This constructive phase is the Reverse Process.") as tracker:
            arrow_reverse = Arrow(forward_process_label.get_bottom(), reverse_process_label.get_right(), color=GREEN)
            self.play(GrowArrow(arrow_reverse))

        # Show Equation
        eq = MathTex(
            r"q", r"(x_t", r"\vert", r"x_{t-1})", r"=",
            r"\mathcal{N}", r"(x_t;", r"\sqrt{1 - \beta_t} x_{t-1},", r"\beta_t \mathbf{I})"
        ).scale(0.8).shift(RIGHT * 3 + DOWN * 1)

        with self.voiceover(text="By mastering this bidirectional journey—from order to chaos, and back from chaos to order—diffusion models can generate breathtakingly detailed images, synthesize realistic audio, and even model complex protein structures with a level of fidelity that previous generations of AI could only dream of.") as tracker:
            self.play(Write(eq[0:4]))
            self.wait(0.5)
            self.play(Write(eq[4]))
            self.wait(0.5)
            self.play(Write(eq[5:9]))

        self.wait(1.5)
        self.play(
            FadeOut(title), FadeOut(subtitle), FadeOut(square_data),
            FadeOut(original_data_label), FadeOut(forward_process_label),
            FadeOut(reverse_process_label), FadeOut(arrow_forward),
            FadeOut(arrow_reverse), FadeOut(eq)
        )
        self.wait(1.0)

        # ---------------------------------------------------------
        # SECTION 2: WHY DO WE NEED IT?
        # ---------------------------------------------------------

        why_title = Text("Why do we need Diffusion Models?", font_size=48, weight=BOLD).to_edge(UP)

        # GANs Side
        gans_group = VGroup()
        gans_title = Text("Before: GANs", font_size=36, color=RED)
        gans_bullet1 = Text("• Fast (Single Step)", font_size=24)
        gans_bullet2 = Text("• Mode Collapse", font_size=24)
        gans_bullet3 = Text("• Unstable Training", font_size=24)
        gans_group.add(gans_title, gans_bullet1, gans_bullet2, gans_bullet3).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        gans_group.shift(LEFT * 3)

        with self.voiceover(text="To appreciate why diffusion models are so revolutionary, we have to look back at what the industry was using before them. For years, Generative Adversarial Networks, commonly known as GANs, were the undisputed kings of image generation.") as tracker:
            self.play(Write(why_title))
            self.play(FadeIn(gans_title))

        with self.voiceover(text="GANs work by pitting two neural networks against each other in a relentless game of cat and mouse. While GANs are incredibly fast at generating images—often needing only a single step—they come with severe, fundamental drawbacks.") as tracker:
            self.play(Write(gans_bullet1))

        with self.voiceover(text="The most infamous of these is 'mode collapse'. This happens when the generator discovers a small handful of images that consistently fool the discriminator, and it simply gives up on learning anything else. It might learn to draw a perfect golden retriever, but it completely forgets how to draw any other breed of dog, or cats, or cars.") as tracker:
            self.play(Write(gans_bullet2))

        with self.voiceover(text="Furthermore, the adversarial training process is notoriously unstable; it requires meticulous hyperparameter tuning and often simply fails to converge.") as tracker:
            self.play(Write(gans_bullet3))

        # Diffusion Side
        diff_group = VGroup()
        diff_title = Text("After: Diffusion", font_size=36, color=GREEN)
        diff_bullet1 = Text("• High Diversity", font_size=24)
        diff_bullet2 = Text("• Stable Training", font_size=24)
        diff_bullet3 = Text("• Step-by-step control", font_size=24)
        diff_group.add(diff_title, diff_bullet1, diff_bullet2, diff_bullet3).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        diff_group.shift(RIGHT * 3)

        with self.voiceover(text="The AI community desperately needed a generative model that could provide much higher diversity, mathematically guaranteed stable training, and finer control over the generation process. Diffusion models solve this beautifully.") as tracker:
            self.play(FadeIn(diff_title))

        with self.voiceover(text="By replacing the chaotic adversarial game with a rigorous, step-by-step denoising process rooted in non-equilibrium thermodynamics, diffusion models provide unprecedented control and astonishingly high diversity.") as tracker:
            self.play(Write(diff_bullet1), Write(diff_bullet3))

        with self.voiceover(text="This leap in stability and quality is the direct technical catalyst for the AI art revolution we are experiencing right now.") as tracker:
            self.play(Write(diff_bullet2))

        self.wait(1.5)
        self.play(FadeOut(why_title), FadeOut(gans_group), FadeOut(diff_group))
        self.wait(1.0)

        # ---------------------------------------------------------
        # SECTION 3: USE CASES
        # ---------------------------------------------------------

        uc_title = Text("Use Cases", font_size=48, weight=BOLD).to_edge(UP)

        dalle_group = VGroup()
        dalle_header = Text("OpenAI's DALL-E 3", font_size=32, color=BLUE)
        dalle_desc = Text("Text-to-Image Generation", font_size=24)
        dalle_group.add(dalle_header, dalle_desc).arrange(DOWN, aligned_edge=LEFT)
        dalle_group.shift(LEFT*3 + UP*0.5)

        with self.voiceover(text="Today, Diffusion models have moved out of the research labs and are directly powering some of the most impressive and widely used AI products on the consumer market.") as tracker:
            self.play(Write(uc_title))

        with self.voiceover(text="Perhaps the most famous example is OpenAI's DALL-E 3, which leverages advanced, text-conditioned diffusion techniques to generate highly detailed, semantically accurate, and photorealistic images directly from natural language textual descriptions.") as tracker:
            self.play(FadeIn(dalle_group))

        mj_group = VGroup()
        mj_header = Text("Midjourney", font_size=32, color=PURPLE)
        mj_desc = Text("Concept Art & Prototyping", font_size=24)
        mj_group.add(mj_header, mj_desc).arrange(DOWN, aligned_edge=LEFT)
        mj_group.shift(RIGHT*3 + UP*0.5)

        with self.voiceover(text="Another massively popular use case is Midjourney, a platform heavily utilized by professional graphic designers, concept artists, and marketing agencies. Midjourney allows creators to rapidly prototype visual concepts, ranging from atmospheric video game assets to complex architectural visualizations, in a fraction of the time it would traditionally take.") as tracker:
            self.play(FadeIn(mj_group))

        with self.voiceover(text="But the applications don't stop at pixels. Companies in the biotech sector are actively adapting these exact same diffusion architectures to generate novel 3D molecular structures for drug discovery, proving that this technology is as versatile as it is powerful.") as tracker:
            self.wait(tracker.duration)

        self.wait(1.5)
        self.play(FadeOut(uc_title), FadeOut(dalle_group), FadeOut(mj_group))
        self.wait(1.0)

        # ---------------------------------------------------------
        # SECTION 4: KEY INTERVIEW INSIGHT
        # ---------------------------------------------------------

        insight_title = Text("Key Interview Insight", font_size=48, weight=BOLD, color=RED).to_edge(UP)

        insight_box = VGroup()
        tradeoff_text = Text("Tradeoff: Generation Quality vs. Inference Speed", font_size=32, weight=BOLD)
        bullet_mc = Text("• Markov chain requires hundreds of sequential steps.", font_size=24)
        bullet_slow = Text("• Extremely slow inference compared to GANs.", font_size=24)
        bullet_sol = Text("• Solutions: DDIM & Latent Diffusion.", font_size=24, color=BLUE)

        insight_content = VGroup(tradeoff_text, bullet_mc, bullet_slow, bullet_sol).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        rect = SurroundingRectangle(insight_content, color=RED, buff=0.5, stroke_width=4)
        insight_box.add(rect, insight_content)

        with self.voiceover(text="If you are interviewing for a specialized role in generative AI or machine learning engineering, you must be prepared for the most common 'gotcha' question about diffusion models.") as tracker:
            self.play(Write(insight_title))

        with self.voiceover(text="Interviewers will inevitably ask you about the fundamental tradeoff inherent in this architecture: the severe tradeoff between generation quality and inference speed.") as tracker:
            self.play(Write(tradeoff_text))
            self.play(Create(rect))

        with self.voiceover(text="The key insight you need to articulate is this: Because diffusion models generate data through a mathematical Markov chain consisting of hundreds or even thousands of sequential denoising steps, the inference process is notoriously slow and computationally expensive compared to single-step models like GANs or Variational Autoencoders.") as tracker:
            self.play(Write(bullet_mc))
            self.play(Write(bullet_slow))

        with self.voiceover(text="Interviewers want to know if you understand this critical bottleneck and how to solve it in a production environment. You should immediately mention acceleration techniques. For instance, you could discuss DDIM (Denoising Diffusion Implicit Models), which modifies the sampling process to allow for safely skipping steps.") as tracker:
            self.play(Write(bullet_sol))

        with self.voiceover(text="Even more importantly, you must mention Latent Diffusion, the core innovation behind models like Stable Diffusion. By moving the entire diffusion process out of the massive pixel space and into a much smaller, highly compressed lower-dimensional latent space, we can drastically speed up the reverse process without sacrificing the final image quality. Mastering this tradeoff is what separates theoretical knowledge from production-ready engineering.") as tracker:
            self.wait(tracker.duration)

        self.wait(2.0)
        self.play(FadeOut(insight_title), FadeOut(insight_box))
        self.wait(1.0)
