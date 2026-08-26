from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class Day34LRSchedulers(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # Helper text stylings
        def create_title(text):
            return Text(text, color=BLACK, font_size=40, weight=BOLD).to_edge(UP)

        # ------------------------------------------------------------
        # Intro
        # ------------------------------------------------------------
        with self.voiceover(text="Welcome back to our AI Engineering series. Today, we're diving into Learning Rate Schedulers, specifically Warmup and Cosine Decay.") as tracker:
            intro_title = create_title("Learning Rate Schedulers")
            subtitle = Text("Warmup & Cosine Decay", color=BLUE, font_size=32).next_to(intro_title, DOWN)
            self.play(Write(intro_title))
            self.play(FadeIn(subtitle))

        self.wait(1.5)
        self.play(FadeOut(VGroup(intro_title, subtitle)))

        # ------------------------------------------------------------
        # Section 1: What is it?
        # ------------------------------------------------------------
        with self.voiceover(text="What is a learning rate scheduler? It is a mechanism that adjusts the learning rate during training, rather than keeping it constant. Specifically, Warmup starts the learning rate very low and gradually increases it, while Cosine Decay slowly reduces it following a cosine curve.") as tracker:
            sec1_title = create_title("What is it?")
            self.play(Write(sec1_title))

            definition1 = Text("Warmup: Gradually increases LR from near-zero.", color=BLACK, font_size=28).move_to(UP)
            definition2 = Text("Cosine Decay: Gradually decreases LR following a cosine curve.", color=BLACK, font_size=28).next_to(definition1, DOWN, buff=0.5)

            self.play(FadeIn(definition1))
            self.play(FadeIn(definition2))

        self.wait(1.5)
        self.play(FadeOut(VGroup(definition1, definition2)))

        # Visualizing the schedule
        with self.voiceover(text="Let's see what this looks like visually. On the x-axis, we have training steps, and on the y-axis, the learning rate.") as tracker:
            axes = Axes(
                x_range=[0, 10, 2],
                y_range=[0, 1.2, 0.5],
                x_length=7,
                y_length=4,
                axis_config={"color": BLACK},
                tips=False
            ).next_to(sec1_title, DOWN, buff=0.5)

            labels = axes.get_axis_labels(
                x_label=Text("Steps", color=BLACK, font_size=24),
                y_label=Text("Learning Rate", color=BLACK, font_size=24)
            )
            self.play(Write(axes), FadeIn(labels))

        with self.voiceover(text="During the warmup phase, the learning rate scales up linearly to a target peak. This prevents large, destabilizing weight updates early in training when the model weights are essentially random.") as tracker:
            warmup_curve = axes.plot(lambda x: x, x_range=[0, 1], color=RED)
            warmup_label = Text("Warmup", color=RED, font_size=20).next_to(axes.c2p(0.5, 0.5), UP, buff=0.2)
            self.play(Write(warmup_curve))
            self.play(FadeIn(warmup_label))

        with self.voiceover(text="After the warmup, the cosine decay phase takes over, smoothly lowering the learning rate to zero. The smooth transition allows the model to fine-tune its parameters without getting stuck.") as tracker:
            # We use a parametric curve for cosine decay to avoid lambda issues with caching
            import numpy as np
            x_vals = np.linspace(1, 10, 100)
            y_vals = 0.5 * (1 + np.cos(np.pi * (x_vals - 1) / 9))
            cosine_curve = VMobject(color=BLUE)
            cosine_curve.set_points_smoothly([axes.c2p(x, y) for x, y in zip(x_vals, y_vals)])

            decay_label = Text("Cosine Decay", color=BLUE, font_size=20).next_to(axes.c2p(5, 0.5), UP, buff=0.2)
            self.play(Write(cosine_curve))
            self.play(FadeIn(decay_label))

        self.wait(1.5)
        self.play(FadeOut(VGroup(sec1_title, axes, labels, warmup_curve, warmup_label, cosine_curve, decay_label)))

        # ------------------------------------------------------------
        # Section 2: Why do we need it?
        # ------------------------------------------------------------
        with self.voiceover(text="Why do we need this complexity? Let's look at the problem without a learning rate scheduler.") as tracker:
            sec2_title = create_title("Why do we need it?")
            self.play(Write(sec2_title))

        with self.voiceover(text="Without a scheduler, a constant high learning rate can cause the loss to spike early on, or bounce around the minimum later in training without ever settling.") as tracker:
            axes2 = Axes(
                x_range=[0, 10, 2],
                y_range=[0, 5, 1],
                x_length=7,
                y_length=4,
                axis_config={"color": BLACK},
                tips=False
            ).next_to(sec2_title, DOWN, buff=0.5)
            labels2 = axes2.get_axis_labels(
                x_label=Text("Steps", color=BLACK, font_size=24),
                y_label=Text("Loss", color=BLACK, font_size=24)
            )

            # Bad loss curve: bounces around
            np.random.seed(42)
            x_bad = np.linspace(0, 10, 50)
            y_bad = 3 * np.exp(-0.2 * x_bad) + np.random.normal(0, 0.4, len(x_bad)) + 1
            bad_curve = VMobject(color=RED)
            bad_curve.set_points_smoothly([axes2.c2p(x, y) for x, y in zip(x_bad, y_bad)])

            bad_label = Text("Constant LR (Spikes & Bounces)", color=RED, font_size=24).to_corner(DR)

            self.play(Write(axes2), FadeIn(labels2))
            self.play(Write(bad_curve), FadeIn(bad_label))

        with self.voiceover(text="Now, let's see what happens with Warmup and Cosine Decay. The loss starts decreasing smoothly without early spikes, and eventually settles into a deep minimum as the learning rate decays.") as tracker:
            # Good loss curve: smooth
            x_good = np.linspace(0, 10, 50)
            y_good = 4 * np.exp(-0.5 * x_good) + 0.5
            good_curve = VMobject(color=GREEN)
            good_curve.set_points_smoothly([axes2.c2p(x, y) for x, y in zip(x_good, y_good)])

            good_label = Text("With Warmup + Decay", color=GREEN, font_size=24).next_to(bad_label, UP, aligned_edge=RIGHT)

            self.play(Transform(bad_curve, good_curve))
            self.play(FadeIn(good_label))

        self.wait(1.5)
        self.play(FadeOut(VGroup(sec2_title, axes2, labels2, bad_curve, bad_label, good_label)))

        # ------------------------------------------------------------
        # Section 3: Use Cases
        # ------------------------------------------------------------
        with self.voiceover(text="Where is this used in the real world? Almost everywhere in modern deep learning.") as tracker:
            sec3_title = create_title("Use Cases")
            self.play(Write(sec3_title))

        with self.voiceover(text="For example, OpenAI used warmup and cosine decay when training GPT-3 and ChatGPT. Transformers are notoriously sensitive during early training, so warmup is essential.") as tracker:
            case1 = Text("1. OpenAI (ChatGPT / GPT-3): Stabilizes Transformer training.", color=BLACK, font_size=28).move_to(UP)
            self.play(FadeIn(case1))

        with self.voiceover(text="Another example is Spotify, which uses these schedulers in their recommendation systems to fine-tune collaborative filtering embeddings without overshooting optimal weights.") as tracker:
            case2 = Text("2. Spotify: Fine-tunes embeddings in recommendation systems.", color=BLACK, font_size=28).next_to(case1, DOWN, buff=0.5)
            self.play(FadeIn(case2))

        self.wait(1.5)
        self.play(FadeOut(VGroup(sec3_title, case1, case2)))

        # ------------------------------------------------------------
        # Math Deep Dive
        # ------------------------------------------------------------
        with self.voiceover(text="Let's look closely at the math for Cosine Decay, as it often comes up in technical deep dives.") as tracker:
            math_title = create_title("Cosine Decay Formula")
            self.play(Write(math_title))

        with self.voiceover(text="The learning rate at step t, represented as eta sub t, is a function of the initial learning rate, eta initial.") as tracker:
            eq1 = MathTex(r"\eta_t", r"=", r"\eta_{initial}").set_color(BLACK)
            self.play(Write(eq1))

        with self.voiceover(text="We multiply this by a factor that scales from 1 down to 0.") as tracker:
            eq2 = MathTex(r"\eta_t", r"=", r"\eta_{initial}", r"\times \frac{1}{2} \left( 1 + \dots \right)").set_color(BLACK)
            self.play(TransformMatchingTex(eq1, eq2))

        with self.voiceover(text="The internal part is a cosine function based on the current step T relative to the total steps T max.") as tracker:
            eq3 = MathTex(r"\eta_t", r"=", r"\eta_{initial}", r"\times \frac{1}{2} \left( 1 + \cos\left( \pi \frac{T}{T_{max}} \right) \right)").set_color(BLACK)
            self.play(TransformMatchingTex(eq2, eq3))

        self.wait(1.5)
        self.play(FadeOut(VGroup(math_title, eq3)))

        # ------------------------------------------------------------
        # Section 4: Key Interview Insight
        # ------------------------------------------------------------
        with self.voiceover(text="Finally, let's discuss the Key Interview Insight. What is the most common gotcha interviewers will test you on?") as tracker:
            sec4_title = create_title("Key Interview Insight")
            self.play(Write(sec4_title))

        with self.voiceover(text="The most critical tradeoff is setting the Warmup Steps. If your warmup is too short, the model diverges early due to massive gradients. If it's too long, you waste compute and learn too slowly.") as tracker:
            insight_box = Rectangle(width=10, height=4, color=PURPLE, fill_color=PURPLE, fill_opacity=0.1)
            insight_text_1 = Text("Gotcha: Sizing the Warmup Phase", color=BLACK, font_size=32, weight=BOLD).next_to(insight_box.get_top(), DOWN, buff=0.5)
            insight_text_2 = Text("• Too Short: Model diverges instantly (gradient explosion).", color=BLACK, font_size=24).next_to(insight_text_1, DOWN, buff=0.5, aligned_edge=LEFT)
            insight_text_3 = Text("• Too Long: Wastes compute; struggles to escape local minima.", color=BLACK, font_size=24).next_to(insight_text_2, DOWN, buff=0.3, aligned_edge=LEFT)

            self.play(FadeIn(insight_box))
            self.play(Write(insight_text_1))
            self.play(FadeIn(insight_text_2))
            self.play(FadeIn(insight_text_3))

        with self.voiceover(text="A common rule of thumb is to use 5 to 10 percent of your total training steps for warmup, then transition to cosine decay. Remember this heuristic for your system design interviews.") as tracker:
            rule_of_thumb = Text("Rule of Thumb: Warmup for ~5-10% of total steps.", color=PURPLE, font_size=28, weight=BOLD).next_to(insight_text_3, DOWN, buff=0.5)
            self.play(FadeIn(rule_of_thumb))

        self.wait(2)

        with self.voiceover(text="That covers the essentials of Warmup and Cosine Decay schedulers. Thank you for watching, and see you in the next session.") as tracker:
            self.play(FadeOut(VGroup(sec4_title, insight_box, insight_text_1, insight_text_2, insight_text_3, rule_of_thumb)))

        self.wait(1)
