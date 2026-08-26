from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class GradientClippingScene(VoiceoverScene):
    def construct(self):
        # Configure Voiceover Service
        self.set_speech_service(GTTSService())

        # Set white background (as required by aesthetic guidelines)
        self.camera.background_color = WHITE

        # --- Section 1: What is it? ---
        title = Text("Gradient Clipping", color=BLACK, font_size=48).to_edge(UP)
        definition = Text(
            "A technique to cap the size of gradients during backpropagation,\n"
            "preventing them from growing out of control.",
            color=BLACK, font_size=28
        ).next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="Welcome back! Today we are discussing Gradient Clipping. Gradient Clipping is a technique used in deep learning to cap the maximum size of gradients during backpropagation. This prevents the gradients from growing out of control, a phenomenon known as exploding gradients.") as tracker:
            self.play(Write(title))
            self.wait(1)
            self.play(FadeIn(definition))
            self.wait(1.5)

        # Mathematical definition derivation using MathTex
        eq_text = Text("Mathematically:", color=BLACK, font_size=32).next_to(definition, DOWN, buff=0.8).align_to(definition, LEFT)

        eq_start = MathTex(
            r"g",
            color=BLACK, font_size=36
        ).next_to(eq_text, DOWN, buff=0.5)

        eq_clip = MathTex(
            r"g", r"\leftarrow", r"g", r"\cdot \min\left(1, \frac{\text{threshold}}{\|g\|}\right)",
            color=BLACK, font_size=36
        ).move_to(eq_start)

        with self.voiceover(text="Mathematically, we scale the gradient vector g down if its norm exceeds a predefined threshold. This ensures its direction is maintained, but its magnitude is kept in check.") as tracker:
            self.play(FadeIn(eq_text))
            self.play(Write(eq_start))
            self.wait(1)
            self.play(TransformMatchingTex(eq_start, eq_clip))
            self.wait(1.5)

        self.play(FadeOut(VGroup(definition, eq_text, eq_clip)))

        # --- Section 2: Why do we need it? ---
        # Before (Exploding Gradients) vs After (Clipped)
        subtitle_why = Text("Why do we need it?", color=BLACK, font_size=40).next_to(title, DOWN, buff=0.5)

        # Diagram: Unclipped gradient step
        axes_unclipped = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": BLACK, "include_numbers": False}
        ).shift(LEFT * 3 + DOWN * 1)

        axes_unclipped_title = Text("Without Clipping", color=RED, font_size=28).next_to(axes_unclipped, UP)

        # We pre-calculate points to avoid lambda cache bugs
        x_vals = [x/10 for x in range(101)]
        y_vals = [(x-5)**2 / 4 + 1 for x in x_vals] # A simple parabola

        # A steep "cliff" representing a chaotic loss landscape common in RNNs
        curve_points = [axes_unclipped.c2p(x, (1/(0.5 + abs(x-5))) * 5 + (x-2)**2 * 0.1) for x in range(11)]
        loss_curve = VMobject(color=BLACK)
        loss_curve.set_points_smoothly(curve_points)

        dot_start = Dot(axes_unclipped.c2p(6, 4.6), color=BLUE)
        arrow_exploding = Arrow(
            dot_start.get_center(),
            axes_unclipped.c2p(1, 9),
            buff=0, color=RED, max_tip_length_to_length_ratio=0.1
        )
        dot_explode = Dot(axes_unclipped.c2p(1, 9), color=RED)

        with self.voiceover(text="Why do we need this? In architectures like Recurrent Neural Networks, gradients can multiply repeatedly through layers. Without clipping, a steep loss landscape can cause a massive gradient update.") as tracker:
            self.play(FadeIn(subtitle_why))
            self.wait(1)
            self.play(FadeIn(VGroup(axes_unclipped, axes_unclipped_title, loss_curve)))
            self.play(FadeIn(dot_start))
            self.wait(0.5)
            self.play(GrowArrow(arrow_exploding))
            self.play(FadeIn(dot_explode))
            self.wait(1.5)

        with self.voiceover(text="This massive update throws our model parameters far away from the optimal region, potentially causing numerical instability or NaN errors. This is the exploding gradient problem.") as tracker:
            self.wait(2) # Give viewer time to absorb

        # Diagram: Clipped gradient step
        axes_clipped = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": BLACK, "include_numbers": False}
        ).shift(RIGHT * 3 + DOWN * 1)

        axes_clipped_title = Text("With Clipping", color=GREEN, font_size=28).next_to(axes_clipped, UP)

        loss_curve_clipped = VMobject(color=BLACK)
        loss_curve_clipped.set_points_smoothly([axes_clipped.c2p(x, (1/(0.5 + abs(x-5))) * 5 + (x-2)**2 * 0.1) for x in range(11)])

        dot_start_c = Dot(axes_clipped.c2p(6, 4.6), color=BLUE)

        # Direction is same as arrow_exploding, but magnitude is scaled down
        vector_unclipped = axes_unclipped.c2p(1, 9) - axes_unclipped.c2p(6, 4.6)
        clipped_end_point = dot_start_c.get_center() + vector_unclipped * 0.25 # Clip magnitude

        arrow_clipped = Arrow(
            dot_start_c.get_center(),
            clipped_end_point,
            buff=0, color=GREEN, max_tip_length_to_length_ratio=0.2
        )
        dot_clip = Dot(clipped_end_point, color=GREEN)

        with self.voiceover(text="Now, let's look at the same scenario with Gradient Clipping applied. When that same steep gradient is calculated, we cap its magnitude. The direction of the update remains exactly the same, but the step size is safe and controlled, keeping our optimization on track.") as tracker:
            self.play(FadeIn(VGroup(axes_clipped, axes_clipped_title, loss_curve_clipped)))
            self.play(FadeIn(dot_start_c))
            self.wait(0.5)
            self.play(GrowArrow(arrow_clipped))
            self.play(FadeIn(dot_clip))
            self.wait(1.5)

        self.play(FadeOut(VGroup(
            subtitle_why, axes_unclipped, axes_unclipped_title, loss_curve, dot_start, arrow_exploding, dot_explode,
            axes_clipped, axes_clipped_title, loss_curve_clipped, dot_start_c, arrow_clipped, dot_clip
        )))

        # --- Section 3: Use Cases ---
        subtitle_uses = Text("Use Cases", color=BLACK, font_size=40).next_to(title, DOWN, buff=0.5)

        use_case_1 = Text("1. Training Large Language Models (LLMs)\n   e.g., OpenAI uses it to stabilize GPT training.", color=BLACK, font_size=28, t2c={"OpenAI": BLUE, "GPT": PURPLE}).move_to(UP * 0.5)
        use_case_2 = Text("2. Training Recurrent Neural Networks (RNNs/LSTMs)\n   e.g., Spotify uses it for sequential music recommendation models.", color=BLACK, font_size=28, t2c={"Spotify": GREEN, "RNNs/LSTMs": PURPLE}).next_to(use_case_1, DOWN, buff=1.0)

        with self.voiceover(text="So, where is this actually used in the real world? First, it is practically mandatory when training Large Language Models. Companies like OpenAI rely on gradient clipping to prevent massive spikes in loss that would destabilize the training of models like GPT.") as tracker:
            self.play(FadeIn(subtitle_uses))
            self.wait(0.5)
            self.play(Write(use_case_1))
            self.wait(1.5)

        with self.voiceover(text="Second, it is crucial for sequential models like RNNs or LSTMs. For instance, Spotify might use recurrent models for generating sequential music playlists, and gradient clipping is needed to stop gradients from blowing up over long sequence lengths.") as tracker:
            self.play(Write(use_case_2))
            self.wait(1.5)

        self.play(FadeOut(VGroup(subtitle_uses, use_case_1, use_case_2)))

        # --- Section 4: Key Interview Insight ---
        subtitle_insight = Text("Key Interview Insight", color=RED, font_size=40).next_to(title, DOWN, buff=0.5)

        # Callout box
        insight_text = Text(
            "Tradeoff: Value of the Threshold\n\n"
            "• Too High: Fails to prevent exploding gradients.\n"
            "• Too Low: Destroys useful gradient signal, making\n  training agonizingly slow.",
            color=BLACK, font_size=28
        )

        callout_box = SurroundingRectangle(insight_text, color=RED, fill_color=WHITE, fill_opacity=1, buff=0.5)
        callout_group = VGroup(callout_box, insight_text).move_to(DOWN * 0.5)

        with self.voiceover(text="Finally, let's cover the key interview insight. If an interviewer asks you about the downsides or tradeoffs of gradient clipping, they want to hear about the threshold hyperparameter.") as tracker:
            self.play(FadeIn(subtitle_insight))
            self.wait(1)

        with self.voiceover(text="Choosing the clipping threshold is a delicate balance. If you set the threshold too high, it effectively does nothing, and your gradients will still explode.") as tracker:
            self.play(FadeIn(callout_box))
            self.play(Write(insight_text[0:31])) # Title + Too High bullet
            self.wait(1.5)

        with self.voiceover(text="But, if you set the threshold too low, you are constantly squashing valid, helpful gradient signals. This artificially limits your step size everywhere, destroying the model's ability to learn and making training agonizingly slow. You often have to find the sweet spot empirically.") as tracker:
            self.play(Write(insight_text[31:])) # Too Low bullet
            self.wait(2)

        with self.voiceover(text="And that wraps up our deep dive into Gradient Clipping. Best of luck on your interviews, and happy building!") as tracker:
            self.wait(2)
            self.play(FadeOut(VGroup(title, subtitle_insight, callout_group)))
