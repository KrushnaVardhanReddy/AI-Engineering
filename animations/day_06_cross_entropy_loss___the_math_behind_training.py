from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class CrossEntropyLossScene(VoiceoverScene):
    def construct(self):
        # Setup aesthetic (Whiteboard Style)
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # We will track our main title
        title = Text("Cross-Entropy Loss", font_size=48, color=BLACK, weight=BOLD)
        title.to_edge(UP)

        self.play(Write(title))
        self.wait(3.0)

        # ---------------------------------------------------------
        # Section 1: What is it?
        # ---------------------------------------------------------
        what_is_it_title = Text("What is it?", font_size=36, color=BLUE, weight=BOLD)
        what_is_it_title.next_to(title, DOWN, buff=0.5)
        what_is_it_title.to_edge(LEFT, buff=1.0)

        def_text = Text(
            "A metric that measures the difference between two probability distributions.",
            font_size=28, color=BLACK
        )
        def_text.next_to(what_is_it_title, DOWN, buff=0.5).align_to(what_is_it_title, LEFT)

        with self.voiceover(text="What is Cross-Entropy Loss? It is a metric that measures the difference between two probability distributions.") as tracker:
            self.play(FadeIn(what_is_it_title))
            self.play(Write(def_text))

        with self.voiceover(text="Specifically, it compares the true labels of our data with the predictions made by our model.") as tracker:
            # Let's show two distributions
            true_dist = VGroup(
                Text("True: ", font_size=32, color=BLACK),
                MathTex("[1.0, 0.0, 0.0]", color=GREEN).scale(0.8)
            ).arrange(RIGHT)

            pred_dist = VGroup(
                Text("Predicted: ", font_size=32, color=BLACK),
                MathTex("[0.7, 0.2, 0.1]", color=RED).scale(0.8)
            ).arrange(RIGHT)

            dists = VGroup(true_dist, pred_dist).arrange(DOWN, aligned_edge=LEFT)
            dists.next_to(def_text, DOWN, buff=1.0).align_to(what_is_it_title, LEFT)

            self.play(FadeIn(true_dist))
            self.play(FadeIn(pred_dist))

        with self.voiceover(text="The mathematical formula calculates the negative sum of the true probabilities multiplied by the log of the predicted probabilities.") as tracker:
            formula = MathTex(
                "L", "=", "-", "\\sum_{i=1}^{N}", "y_i", "\\log(", "\\hat{y}_i", ")",
                color=BLACK
            ).scale(1.2)

            # Color code formula parts
            formula[4].set_color(GREEN) # y_i
            formula[6].set_color(RED)   # \hat{y}_i

            formula.next_to(dists, RIGHT, buff=2.0)

            self.play(Write(formula[0:4]))
            self.play(Write(formula[4]))
            self.play(Write(formula[5]))
            self.play(Write(formula[6]))
            self.play(Write(formula[7]))

        self.wait(3.0)

        self.play(
            FadeOut(what_is_it_title),
            FadeOut(def_text),
            FadeOut(dists),
            FadeOut(formula)
        )
        self.wait(1.5)

        # ---------------------------------------------------------
        # Section 2: Why do we need it?
        # ---------------------------------------------------------
        why_title = Text("Why do we need it?", font_size=36, color=BLUE, weight=BOLD)
        why_title.next_to(title, DOWN, buff=0.5)
        why_title.to_edge(LEFT, buff=1.0)

        with self.voiceover(text="Why do we need Cross-Entropy Loss? Let's look at the problem without it.") as tracker:
            self.play(FadeIn(why_title))

        with self.voiceover(text="Imagine we use a simple metric like Classification Accuracy.") as tracker:
            acc_title = Text("Using Classification Accuracy", font_size=28, color=BLACK, weight=BOLD)
            acc_title.next_to(why_title, DOWN, buff=0.5).align_to(why_title, LEFT)
            self.play(Write(acc_title))

        with self.voiceover(text="If Model A is 51 percent confident, and Model B is 99 percent confident, both predict the correct class and get 100 percent accuracy.") as tracker:
            model_a = Text("Model A: 51% confident", font_size=28, color=RED)
            model_b = Text("Model B: 99% confident", font_size=28, color=GREEN)

            models = VGroup(model_a, model_b).arrange(DOWN, aligned_edge=LEFT)
            models.next_to(acc_title, DOWN, buff=0.5).align_to(why_title, LEFT)

            acc_result = Text("Accuracy: 100% for both!", font_size=28, color=BLACK)
            acc_result.next_to(models, DOWN, buff=0.5).align_to(why_title, LEFT)

            self.play(FadeIn(model_a))
            self.play(FadeIn(model_b))
            self.play(Write(acc_result))

        with self.voiceover(text="But clearly, Model B is much better. Accuracy is too blunt; it doesn't reward the model for being more confident.") as tracker:
            cross_mark = Cross(acc_result, stroke_color=RED)
            self.play(Create(cross_mark))
            self.wait(3.0)

        self.play(FadeOut(acc_title), FadeOut(models), FadeOut(acc_result), FadeOut(cross_mark))

        with self.voiceover(text="Now, let's see how Cross-Entropy Loss solves this.") as tracker:
            ce_title = Text("Using Cross-Entropy Loss", font_size=28, color=BLACK, weight=BOLD)
            ce_title.next_to(why_title, DOWN, buff=0.5).align_to(why_title, LEFT)
            self.play(Write(ce_title))

        with self.voiceover(text="Because of the logarithm, a prediction of 99 percent yields a loss close to zero, while a prediction of 51 percent yields a much higher loss.") as tracker:
            axes = Axes(
                x_range=[0, 1.1, 0.2],
                y_range=[0, 5, 1],
                x_length=5,
                y_length=3,
                axis_config={"color": BLACK},
            )
            # Add labels separately
            axes_labels = axes.get_axis_labels(
                Tex("Predicted Probability $\\hat{y}$", color=BLACK).scale(0.6),
                Tex("Loss", color=BLACK).scale(0.6)
            )

            log_curve = axes.plot(lambda x: -np.log(x + 1e-5), color=BLUE, x_range=[0.01, 1.0])

            plot_group = VGroup(axes, axes_labels, log_curve)
            plot_group.next_to(ce_title, DOWN, buff=0.5).align_to(why_title, LEFT)

            self.play(Create(axes), Write(axes_labels))
            self.play(Create(log_curve))

            # Show points
            point_99 = Dot(axes.c2p(0.99, -np.log(0.99)), color=GREEN)
            label_99 = Tex("0.99", color=GREEN).scale(0.6).next_to(point_99, UP)

            point_51 = Dot(axes.c2p(0.51, -np.log(0.51)), color=RED)
            label_51 = Tex("0.51", color=RED).scale(0.6).next_to(point_51, RIGHT)

            self.play(FadeIn(point_99), Write(label_99))
            self.play(FadeIn(point_51), Write(label_51))

        with self.voiceover(text="This continuous, smooth gradient allows neural networks to learn efficiently through backpropagation.") as tracker:
            self.wait(3.0)

        self.play(
            FadeOut(why_title), FadeOut(ce_title),
            FadeOut(plot_group), FadeOut(point_99), FadeOut(label_99),
            FadeOut(point_51), FadeOut(label_51)
        )
        self.wait(1.5)

        # ---------------------------------------------------------
        # Section 3: Use Cases
        # ---------------------------------------------------------
        use_cases_title = Text("Use Cases", font_size=36, color=BLUE, weight=BOLD)
        use_cases_title.next_to(title, DOWN, buff=0.5)
        use_cases_title.to_edge(LEFT, buff=1.0)

        with self.voiceover(text="Where is Cross-Entropy Loss actually used?") as tracker:
            self.play(FadeIn(use_cases_title))

        with self.voiceover(text="ChatGPT uses it during training to predict the next word in a sequence across tens of thousands of possible vocabulary words.") as tracker:
            uc1 = VGroup(
                Text("1. ChatGPT (Language Modeling)", font_size=28, color=BLACK, weight=BOLD),
                Text("Predicts the next word across 50,000+ vocabulary choices.", font_size=24, color=BLACK)
            ).arrange(DOWN, aligned_edge=LEFT)
            uc1.next_to(use_cases_title, DOWN, buff=0.5).align_to(use_cases_title, LEFT)
            self.play(Write(uc1))

        with self.voiceover(text="Spotify uses it in its recommendation models to classify whether a user will skip a song or listen to it entirely.") as tracker:
            uc2 = VGroup(
                Text("2. Spotify (Recommendation Systems)", font_size=28, color=BLACK, weight=BOLD),
                Text("Classifies if a user will 'Skip' or 'Listen' to a track.", font_size=24, color=BLACK)
            ).arrange(DOWN, aligned_edge=LEFT)
            uc2.next_to(uc1, DOWN, buff=1.0).align_to(use_cases_title, LEFT)
            self.play(Write(uc2))

        self.wait(3.0)
        self.play(FadeOut(use_cases_title), FadeOut(uc1), FadeOut(uc2))
        self.wait(1.5)

        # ---------------------------------------------------------
        # Section 4: Key Interview Insight
        # ---------------------------------------------------------
        insight_title = Text("Key Interview Insight", font_size=36, color=BLUE, weight=BOLD)
        insight_title.next_to(title, DOWN, buff=0.5)
        insight_title.to_edge(LEFT, buff=1.0)

        with self.voiceover(text="Finally, what is the most common interview question about Cross-Entropy Loss?") as tracker:
            self.play(FadeIn(insight_title))

        insight_box = Rectangle(width=10, height=4, color=PURPLE, stroke_width=4)
        insight_box.next_to(insight_title, DOWN, buff=0.5)

        insight_text_1 = Text(
            "Softmax and Cross-Entropy are a package deal.",
            font_size=28, color=BLACK, weight=BOLD
        )
        insight_text_2 = Text(
            "Interviewers often ask why we use them together.",
            font_size=24, color=BLACK
        )
        insight_text_group = VGroup(insight_text_1, insight_text_2).arrange(DOWN)
        insight_text_group.move_to(insight_box.get_center())

        with self.voiceover(text="Interviewers often test if you know that Softmax and Cross-Entropy are practically a package deal.") as tracker:
            self.play(Create(insight_box))
            self.play(Write(insight_text_group))

        with self.voiceover(text="The insight is mathematical stability. Softmax involves exponentials, and Cross-Entropy involves logarithms. When combined, they cancel out extreme values, preventing numerical overflow during backpropagation.") as tracker:
            insight_math_1 = MathTex("e^{z_i}", color=RED).scale(1.2)
            insight_math_2 = MathTex("\\log(x)", color=BLUE).scale(1.2)
            insight_math_group = VGroup(insight_math_1, MathTex("+", color=BLACK), insight_math_2).arrange(RIGHT, buff=0.5)

            insight_math_result = Text("Numerical Stability!", font_size=32, color=GREEN, weight=BOLD)

            insight_math_group.next_to(insight_text_group, DOWN, buff=0.5)
            insight_math_result.next_to(insight_math_group, DOWN, buff=0.5)

            self.play(Write(insight_math_group))
            self.play(Write(insight_math_result))

        with self.voiceover(text="This combined operation is often called 'LogSumExp'. Mentioning this in an interview shows you understand the deep learning framework implementation details, not just the theory.") as tracker:
            self.wait(3.0)

        self.play(
            FadeOut(insight_title), FadeOut(insight_box),
            FadeOut(insight_text_group), FadeOut(insight_math_group),
            FadeOut(insight_math_result), FadeOut(title)
        )
        self.wait(4.0)

        with self.voiceover(text="That wraps up our dive into Cross-Entropy Loss.") as tracker:
            end_text = Text("Keep Learning!", font_size=48, color=BLUE, weight=BOLD)
            self.play(Write(end_text))
            self.wait(4.0)
            self.play(FadeOut(end_text))
