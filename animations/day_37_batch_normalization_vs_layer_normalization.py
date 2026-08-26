from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np

class BatchVsLayerNorm(VoiceoverScene):
    def construct(self):
        # Setup aesthetic
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # ==========================================================
        # 1. WHAT IS IT?
        # ==========================================================
        title = Text("Batch Normalization vs Layer Normalization", color=BLACK, weight=BOLD).scale(0.8)
        self.play(FadeIn(title, shift=UP))
        self.play(title.animate.to_edge(UP))

        what_title = Text("1. What is it?", color=BLUE, weight=BOLD).scale(0.7)
        what_title.next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="Welcome to this comprehensive technical overview. Today we will dive deep and compare two incredibly critical techniques used in deep learning: Batch Normalization and Layer Normalization. These operations are essentially the backbone of modern architectures, allowing us to train extremely deep neural networks without getting stuck.") as tracker:
            self.play(Write(what_title))
        self.wait(1.5)

        def_text1 = Text("Techniques to stabilize training", color=BLACK, font_size=24)
        def_text2 = Text("by normalizing intermediate activations.", color=BLACK, font_size=24)
        def_group = VGroup(def_text1, def_text2).arrange(DOWN, buff=0.2).next_to(what_title, DOWN, buff=0.5)

        with self.voiceover(text="Simply put, these are highly effective techniques implemented to stabilize and drastically accelerate the training process. They achieve this by consistently normalizing the intermediate activations of a neural network layer by layer. Let us take a moment to understand the mathematics behind this normalization.") as tracker:
            self.play(FadeIn(def_text1))
            self.play(FadeIn(def_text2))
        self.wait(2.0)

        # Mathematical definition with step-by-step derivation
        eq_step1 = MathTex(r"x", color=BLACK)
        eq_step2 = MathTex(r"x", r"- \mu", color=BLACK)
        # Fix missing braces in LaTeX strings
        eq_step3 = MathTex(r"\frac{x - \mu}{\sqrt{\sigma^2}}", color=BLACK)
        eq_step4 = MathTex(r"\hat{x}", r"=", r"\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}", color=BLACK)

        eq_step1.next_to(def_group, DOWN, buff=1.0)
        eq_step2.move_to(eq_step1, aligned_edge=UP)
        eq_step3.move_to(eq_step2, aligned_edge=UP)
        eq_step4.move_to(eq_step3, aligned_edge=UP)

        with self.voiceover(text="Consider an intermediate feature activation, denoted by x. First, to center our data, we subtract the mean, mu.") as tracker:
            self.play(Write(eq_step1))
            self.wait(1.0)
            self.play(TransformMatchingTex(eq_step1, eq_step2))
        self.wait(1.0)

        with self.voiceover(text="Next, to ensure uniform scale, we divide this centered value by the standard deviation, which is the square root of the variance.") as tracker:
            self.play(TransformMatchingTex(eq_step2, eq_step3))
        self.wait(1.0)

        with self.voiceover(text="We also add a very small constant epsilon to the variance before taking the root, purely for numerical stability to prevent division by zero. This gives us our normalized activation, x hat.") as tracker:
            self.play(TransformMatchingTex(eq_step3, eq_step4))
        self.wait(1.5)

        eq_scale = MathTex(r"y", r"=", r"\gamma", r"\hat{x}", r"+ \beta", color=BLACK).next_to(eq_step4, DOWN, buff=0.5)

        with self.voiceover(text="However, we don't want to permanently constrain the network's expressive power. So, the network learns two parameters: gamma to scale the output, and beta to shift it. This results in our final output y.") as tracker:
            self.play(Write(eq_scale))
        self.wait(2.0)

        # Visual distinction
        distinction_text = Text(
            "The core difference is HOW the mean and variance are calculated.",
            color=PURPLE, font_size=24
        ).next_to(eq_scale, DOWN, buff=1)

        with self.voiceover(text="Both Batch Normalization and Layer Normalization apply this exact mathematical foundation. The fundamental, defining difference between them is precisely which dimensions they compute that mean and variance over when operating on a batch of multi-dimensional tensor data.") as tracker:
            self.play(FadeIn(distinction_text))
        self.wait(2.0)

        self.play(
            FadeOut(VGroup(def_group, eq_step4, eq_scale, distinction_text))
        )
        self.wait(1.0)

        # ==========================================================
        # 2. WHY DO WE NEED IT? (Visual comparison)
        # ==========================================================
        why_title = Text("2. Why do we need it?", color=BLUE, weight=BOLD).scale(0.7)
        why_title.move_to(what_title)

        with self.voiceover(text="This brings us to the next critical question: why do we absolutely need normalization in the first place? What happens if we just don't use it? Let's visualize the problem known as internal covariate shift.") as tracker:
            self.play(Transform(what_title, why_title))
        self.wait(1.0)

        # Before (Without Norm)
        before_title = Text("Without Normalization", color=RED, font_size=24).shift(LEFT*3.5 + UP*1.5)

        # Simple curve showing exploding/shifting gradients
        axes_before = Axes(x_range=[0, 10, 2], y_range=[-5, 5, 2], x_length=4, y_length=3, axis_config={"color": BLACK}).next_to(before_title, DOWN)
        x_vals = np.linspace(0.1, 10, 100)
        y_vals_before = np.sin(x_vals) * np.exp(x_vals/4)
        curve_before = VMobject(color=RED)
        curve_before.set_points_smoothly([axes_before.c2p(x, y) for x, y in zip(x_vals, y_vals_before)])

        with self.voiceover(text="Without any normalization, as training data passes forward through many sequential layers of a deep network, the mathematical distribution of those internal activations begins to shift wildly. Because the weights are constantly updating, the input to each subsequent layer is a moving target. This causes the gradients to either explode or vanish, rendering the learning process extremely unstable, chaotic, and painfully slow.") as tracker:
            self.play(FadeIn(before_title))
            self.play(FadeIn(axes_before))
            self.play(Write(curve_before), run_time=3.5)
        self.wait(1.5)

        # After (With Norm)
        after_title = Text("With Normalization", color=GREEN, font_size=24).shift(RIGHT*3.5 + UP*1.5)
        axes_after = Axes(x_range=[0, 10, 2], y_range=[-5, 5, 2], x_length=4, y_length=3, axis_config={"color": BLACK}).next_to(after_title, DOWN)
        y_vals_after = np.sin(x_vals) # Stabilized
        curve_after = VMobject(color=GREEN)
        curve_after.set_points_smoothly([axes_after.c2p(x, y) for x, y in zip(x_vals, y_vals_after)])

        with self.voiceover(text="But with proper normalization applied, we explicitly constrain the activations to maintain a stable, predictable, and bounded distribution. This significantly smooths out the optimization landscape. As a result, we are safely allowed to use much higher learning rates, which leads to dramatically faster and more reliable convergence during model training.") as tracker:
            self.play(FadeIn(after_title))
            self.play(FadeIn(axes_after))
            self.play(Write(curve_after), run_time=3.5)
        self.wait(2.0)

        self.play(
            FadeOut(VGroup(before_title, axes_before, curve_before, after_title, axes_after, curve_after))
        )
        self.wait(1.0)

        # ==========================================================
        # 3. USE CASES
        # ==========================================================
        cases_title = Text("3. Use Cases", color=BLUE, weight=BOLD).scale(0.7)
        cases_title.move_to(why_title)

        with self.voiceover(text="Understanding the theory is great, but as an AI engineer, you must know when to actually use which technique. Let us examine the standard architectural use cases.") as tracker:
            self.play(Transform(what_title, cases_title))
        self.wait(1.0)

        # Batch Norm Use Case
        bn_title = Text("Batch Normalization", color=PURPLE, font_size=28, weight=BOLD).shift(LEFT*3.5 + UP*1.5)
        bn_desc1 = Text("Normalizes across the", color=BLACK, font_size=20)
        bn_desc2 = Text("Batch dimension.", color=BLACK, font_size=20, weight=BOLD)
        bn_desc_group = VGroup(bn_desc1, bn_desc2).arrange(DOWN, buff=0.1).next_to(bn_title, DOWN, buff=0.3)
        bn_use = Text("Best for: CNNs (Vision)", color=BLACK, font_size=20).next_to(bn_desc_group, DOWN, buff=0.5)
        bn_ex = Text("e.g., ResNet by Microsoft", color=GREEN, font_size=20).next_to(bn_use, DOWN, buff=0.2)

        with self.voiceover(text="Batch Normalization calculates the mean and variance across the batch dimension, meaning it aggregates statistics from multiple independent examples for each distinct feature channel. Historically, this is the absolute gold standard for Computer Vision architectures, particularly Convolutional Neural Networks. A famous real-world example is the ResNet architecture developed by Microsoft, which powers many image recognition systems today.") as tracker:
            self.play(FadeIn(bn_title))
            self.play(FadeIn(bn_desc_group))
            self.wait(1.0)
            self.play(FadeIn(bn_use))
            self.play(FadeIn(bn_ex))
        self.wait(2.0)

        # Layer Norm Use Case
        ln_title = Text("Layer Normalization", color=PURPLE, font_size=28, weight=BOLD).shift(RIGHT*3.5 + UP*1.5)
        ln_desc1 = Text("Normalizes across the", color=BLACK, font_size=20)
        ln_desc2 = Text("Feature dimension.", color=BLACK, font_size=20, weight=BOLD)
        ln_desc_group = VGroup(ln_desc1, ln_desc2).arrange(DOWN, buff=0.1).next_to(ln_title, DOWN, buff=0.3)
        ln_use = Text("Best for: Transformers (NLP)", color=BLACK, font_size=20).next_to(ln_desc_group, DOWN, buff=0.5)
        ln_ex = Text("e.g., ChatGPT by OpenAI", color=GREEN, font_size=20).next_to(ln_use, DOWN, buff=0.2)

        with self.voiceover(text="Layer Normalization, entirely conversely, computes the mean and variance across the feature dimension for each individual sequence element independently. It does not look at the batch at all. This independence makes it exceptionally well suited, and in fact necessary, for Recurrent Nets and modern Transformer architectures used in Natural Language Processing. A prime example is ChatGPT by OpenAI, which heavily relies on Layer Normalization.") as tracker:
            self.play(FadeIn(ln_title))
            self.play(FadeIn(ln_desc_group))
            self.wait(1.0)
            self.play(FadeIn(ln_use))
            self.play(FadeIn(ln_ex))
        self.wait(2.0)

        self.play(
            FadeOut(VGroup(bn_title, bn_desc_group, bn_use, bn_ex, ln_title, ln_desc_group, ln_use, ln_ex))
        )
        self.wait(1.0)

        # ==========================================================
        # 4. KEY INTERVIEW INSIGHT
        # ==========================================================
        insight_title = Text("4. Key Interview Insight", color=RED, weight=BOLD).scale(0.7)
        insight_title.move_to(cases_title)

        with self.voiceover(text="Now we arrive at the most crucial segment for your preparation: the key interview insight. Interviewers love to test your depth of understanding by asking about edge cases and trade-offs.") as tracker:
            self.play(Transform(what_title, insight_title))
        self.wait(1.0)

        insight_box = Rectangle(width=11, height=4.5, color=RED, fill_color=WHITE, fill_opacity=1)
        insight_box.next_to(insight_title, DOWN, buff=0.5)

        q_text = Text("Q: Why not use Batch Norm for Transformers?", color=BLACK, font_size=24, weight=BOLD)
        q_text.move_to(insight_box.get_center() + UP*1.5)

        with self.voiceover(text="A remarkably common and classic interview question you will face is this: We know Batch Normalization works wonderfully for vision, so why exactly do we abandon it and use Layer Norm instead for NLP and Transformer models?") as tracker:
            self.play(FadeIn(insight_box))
            self.play(Write(q_text))
        self.wait(2.0)

        a1_text = Text("1. Variable Sequence Lengths:", color=BLACK, font_size=20, weight=BOLD)
        a2_text = Text("   NLP sentences vary in length. Batch Norm struggles", color=BLACK, font_size=20)
        a3_text = Text("   calculating meaningful stats over padded, empty tokens.", color=BLACK, font_size=20)

        a4_text = Text("2. Small Batch Size Limitations:", color=BLACK, font_size=20, weight=BOLD)
        a5_text = Text("   Huge language models often require tiny batch sizes.", color=BLACK, font_size=20)
        a6_text = Text("   Batch Norm variance estimates fail with tiny samples.", color=BLACK, font_size=20)

        a_group = VGroup(a1_text, a2_text, a3_text, a4_text, a5_text, a6_text).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        # Add slight extra space between point 1 and 2
        a4_text.shift(DOWN*0.2)
        a5_text.shift(DOWN*0.2)
        a6_text.shift(DOWN*0.2)

        a_group.next_to(q_text, DOWN, buff=0.5)

        with self.voiceover(text="The answer is two-fold and centers around practical constraints. First, variable sequence lengths. Unlike standard images, NLP sentences are not uniform in length. We use padding to make them uniform in memory. Batch Norm seriously struggles here because it tries to compute meaningful statistical variance across a batch where a significant portion of the tokens are literally just zero padding, corrupting the statistics.") as tracker:
            self.play(FadeIn(a1_text))
            self.play(FadeIn(a2_text))
            self.play(FadeIn(a3_text))
        self.wait(2.0)

        with self.voiceover(text="Second, we have severe batch size limitations. Training massive transformer models like GPT consumes immense GPU memory, which often forcibly restricts researchers to using very small batch sizes, sometimes just two or four. Batch Norm completely fails in this regime because attempting to estimate a population variance from merely a few samples is highly inaccurate and introduces destructive noise. Layer Norm, by operating exclusively on a per-instance basis across features, gracefully sidesteps both of these issues entirely.") as tracker:
            self.play(FadeIn(a4_text))
            self.play(FadeIn(a5_text))
            self.play(FadeIn(a6_text))
        self.wait(3.0)

        with self.voiceover(text="If you can clearly articulate these failure modes of Batch Normalization, and explain why Layer Normalization's per-instance calculation solves them, you will easily impress your interviewer. Thank you very much for watching, and good luck with your system design and machine learning interviews.") as tracker:
            self.play(FadeOut(VGroup(insight_box, q_text, a_group, what_title, title)))
        self.wait(2.0)
