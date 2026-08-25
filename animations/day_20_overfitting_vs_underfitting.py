from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class OverfittingVsUnderfitting(VoiceoverScene):
    def construct(self):
        # Set up voiceover
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # --- Section 1: What is it? ---
        title = Text("Overfitting vs. Underfitting", color=BLACK, font_size=48)
        self.play(Write(title))
        self.wait(1.5)

        with self.voiceover(text="Welcome to our AI and ML interview prep series. Today, we are exploring one of the most fundamental concepts in machine learning: Overfitting and Underfitting.") as tracker:
            self.play(title.animate.to_edge(UP))
            self.wait(1)

        # Definition section
        definition_title = Text("What is it?", color=BLUE, font_size=36).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(definition_title))

        underfitting_def = Text("Underfitting: Model is too simple, missing patterns.", color=BLACK, font_size=28).next_to(definition_title, DOWN, buff=0.5)
        overfitting_def = Text("Overfitting: Model is too complex, memorizing noise.", color=BLACK, font_size=28).next_to(underfitting_def, DOWN, buff=0.3)
        good_def = Text("Optimal: Model learns true underlying patterns.", color=BLACK, font_size=28).next_to(overfitting_def, DOWN, buff=0.3)

        with self.voiceover(text="In simple terms, underfitting occurs when a machine learning model is too simple to capture the underlying structure of the data. It essentially misses the patterns completely. Overfitting, on the other hand, happens when a model is far too complex. Instead of learning the general patterns, it starts to memorize the training data, including all the random noise and outliers.") as tracker:
            self.play(Write(underfitting_def))
            self.wait(0.5)
            self.play(Write(overfitting_def))
            self.wait(0.5)

        with self.voiceover(text="The goal of any machine learning model is to find the sweet spot, the optimal balance where the model learns the true underlying patterns so it can generalize well to unseen data.") as tracker:
            self.play(Write(good_def))
            self.wait(1.5)

        self.play(FadeOut(VGroup(definition_title, underfitting_def, overfitting_def, good_def)))

        # Visualizing the 3 scenarios
        ax1 = Axes(x_range=[0, 10, 2], y_range=[0, 10, 2], x_length=3, y_length=3,
                   axis_config={"color": BLACK, "include_numbers": False, "include_ticks": False})
        ax2 = ax1.copy()
        ax3 = ax1.copy()

        graphs = VGroup(ax1, ax2, ax3).arrange(RIGHT, buff=1).shift(DOWN*0.5)

        label1 = Text("Underfitting", font_size=24, color=RED).next_to(ax1, UP)
        label2 = Text("Optimal", font_size=24, color=GREEN).next_to(ax2, UP)
        label3 = Text("Overfitting", font_size=24, color=BLUE).next_to(ax3, UP)

        self.play(FadeIn(graphs), FadeIn(VGroup(label1, label2, label3)))

        # Generate some dummy data points (a quadratic curve + noise)
        np.random.seed(42)
        x_vals = np.linspace(1, 9, 15)
        y_vals = -0.15 * (x_vals - 5)**2 + 8 + np.random.normal(0, 0.8, 15)

        points1 = VGroup(*[Dot(ax1.c2p(x, y), color=BLACK, radius=0.06) for x, y in zip(x_vals, y_vals)])
        points2 = VGroup(*[Dot(ax2.c2p(x, y), color=BLACK, radius=0.06) for x, y in zip(x_vals, y_vals)])
        points3 = VGroup(*[Dot(ax3.c2p(x, y), color=BLACK, radius=0.06) for x, y in zip(x_vals, y_vals)])

        with self.voiceover(text="Let us visualize this with a simple regression problem. Here we have some data points that clearly follow a curved trajectory.") as tracker:
            self.play(FadeIn(points1), FadeIn(points2), FadeIn(points3))
            self.wait(1)

        # Underfitting line
        line1 = ax1.plot(lambda x: 5, color=RED, x_range=[1, 9])
        with self.voiceover(text="If we use a model that is too simple, like a flat horizontal line, it completely fails to capture the curve. This is underfitting. It has high bias, making strong, incorrect assumptions about the data.") as tracker:
            self.play(Write(line1))
            self.wait(1.5)

        # Optimal curve
        curve2 = ax2.plot(lambda x: -0.15 * (x - 5)**2 + 8, color=GREEN, x_range=[1, 9])
        with self.voiceover(text="An optimal model, perhaps a simple polynomial, captures the general shape of the data perfectly without chasing every single point. It will generalize well to new data.") as tracker:
            self.play(Write(curve2))
            self.wait(1.5)

        # Overfitting curve (a complex curve that wiggles to hit every point)
        curve3 = VMobject(color=BLUE)
        curve3.set_points_smoothly([ax3.c2p(x, y) for x, y in zip(x_vals, y_vals)])
        with self.voiceover(text="But if we use a model that is too complex, it will wiggle wildly to touch every single data point perfectly. This is overfitting. While it has zero error on the training data, it has high variance, and will perform terribly on any new, unseen data points.") as tracker:
            self.play(Write(curve3))
            self.wait(1.5)

        self.play(FadeOut(VGroup(graphs, label1, label2, label3, points1, points2, points3, line1, curve2, curve3)))

        # --- Section 2: Why do we need it? ---
        why_title = Text("Why do we need to care?", color=BLUE, font_size=36).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(why_title))

        with self.voiceover(text="Why is understanding this balance so critical? Because deploying an overfitted or underfitted model into production can be disastrous. Let's look at the training process.") as tracker:
            self.wait(1)

        # Training vs Validation loss curve
        loss_ax = Axes(
            x_range=[0, 100, 20], y_range=[0, 10, 2],
            x_length=6, y_length=4,
            axis_config={"color": BLACK, "include_numbers": False},
            x_axis_config={"numbers_to_include": [0, 20, 40, 60, 80, 100]},
            y_axis_config={"numbers_to_include": [0, 5, 10]}
        ).shift(DOWN*0.5)

        x_label = Text("Model Complexity / Epochs", font_size=18, color=BLACK).next_to(loss_ax.x_axis, DOWN)
        y_label = Text("Error", font_size=18, color=BLACK).next_to(loss_ax.y_axis, LEFT).rotate(PI/2)

        self.play(FadeIn(VGroup(loss_ax, x_label, y_label)))

        train_curve = loss_ax.plot(lambda x: 8 * np.exp(-0.05 * x) + 0.5, color=BLUE, x_range=[0, 100])
        val_curve = loss_ax.plot(lambda x: 8 * np.exp(-0.05 * x) + 0.5 + 0.001 * x**2, color=RED, x_range=[0, 100])

        train_label = Text("Training Error", color=BLUE, font_size=20).next_to(train_curve.points[-1], RIGHT)
        val_label = Text("Validation Error", color=RED, font_size=20).next_to(val_curve.points[-1], UP)

        with self.voiceover(text="As we train a model, or increase its complexity, the error on the training data consistently goes down. The model gets better and better at memorizing the data it sees.") as tracker:
            self.play(Write(train_curve))
            self.play(FadeIn(train_label))
            self.wait(1.5)

        with self.voiceover(text="However, we evaluate true performance using validation error on data the model hasn't seen before. At first, validation error drops along with training error.") as tracker:
            self.play(Write(val_curve.get_subcurve(0, 0.4))) # up to x=40 approx
            self.wait(1)

        with self.voiceover(text="But eventually, as the model starts overfitting and memorizing noise, the validation error starts to creep back up, even as training error continues to fall. This divergence is the classic signature of overfitting.") as tracker:
            self.play(Write(val_curve.get_subcurve(0.4, 1.0)))
            self.play(FadeIn(val_label))
            self.wait(1.5)

        optimum_line = DashedLine(
            start=loss_ax.c2p(45, 0),
            end=loss_ax.c2p(45, 10),
            color=GREEN
        )
        optimum_text = Text("Sweet Spot", color=GREEN, font_size=20).next_to(optimum_line, UP)

        with self.voiceover(text="We need to carefully monitor this to find the sweet spot, the point of lowest validation error, before the model begins to overfit.") as tracker:
            self.play(Write(optimum_line), FadeIn(optimum_text))
            self.wait(1.5)

        self.play(FadeOut(VGroup(why_title, loss_ax, x_label, y_label, train_curve, val_curve, train_label, val_label, optimum_line, optimum_text)))

        # --- Section 3: Use Cases ---
        usecase_title = Text("Real-World Use Cases", color=BLUE, font_size=36).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(usecase_title))

        chatgpt_box = RoundedRectangle(width=5, height=2, color=PURPLE, corner_radius=0.2).shift(LEFT*3 + DOWN*0.5)
        chatgpt_text = Text("Large Language Models (e.g. ChatGPT)", color=BLACK, font_size=20).move_to(chatgpt_box).shift(UP*0.5)
        chatgpt_desc = Text("Use techniques like Dropout\nand Regularization to prevent\nmemorizing specific articles.", color=BLACK, font_size=16).next_to(chatgpt_text, DOWN)

        spotify_box = RoundedRectangle(width=5, height=2, color=GREEN, corner_radius=0.2).shift(RIGHT*3 + DOWN*0.5)
        spotify_text = Text("Recommender Systems (e.g. Spotify)", color=BLACK, font_size=20).move_to(spotify_box).shift(UP*0.5)
        spotify_desc = Text("Stop training early to avoid\noverfitting to a user's recent\nlistening history.", color=BLACK, font_size=16).next_to(spotify_text, DOWN)

        with self.voiceover(text="How is this handled in the real world? Let's look at two practical examples.") as tracker:
            self.wait(1)

        with self.voiceover(text="Large Language Models, like the architecture behind ChatGPT, have billions of parameters, making them highly prone to overfitting. Engineers use techniques like Dropout and Weight Regularization to ensure the model learns language patterns rather than memorizing entire Wikipedia articles verbatim.") as tracker:
            self.play(FadeIn(chatgpt_box), Write(chatgpt_text))
            self.play(FadeIn(chatgpt_desc))
            self.wait(1.5)

        with self.voiceover(text="For Recommender Systems, like Spotify's music recommendations, an overfitted model might only suggest songs you listened to yesterday. By using techniques like Early Stopping, they prevent the model from overfitting to recent noise and instead capture your long-term, underlying musical tastes.") as tracker:
            self.play(FadeIn(spotify_box), Write(spotify_text))
            self.play(FadeIn(spotify_desc))
            self.wait(1.5)

        self.play(FadeOut(VGroup(usecase_title, chatgpt_box, chatgpt_text, chatgpt_desc, spotify_box, spotify_text, spotify_desc)))

        # --- Section 4: Key Interview Insight ---
        insight_title = Text("Key Interview Insight", color=RED, font_size=36).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(insight_title))

        callout_box = RoundedRectangle(width=10, height=5, color=RED, corner_radius=0.3).shift(DOWN*0.5)
        insight_heading = Text("The Bias-Variance Tradeoff", color=BLACK, font_size=32).move_to(callout_box).shift(UP*1.5)

        with self.voiceover(text="Finally, let's discuss the most important concept for your interviews. When interviewers ask about overfitting and underfitting, they are actually testing your understanding of the Bias-Variance Tradeoff.") as tracker:
            self.play(FadeIn(callout_box))
            self.play(Write(insight_heading))
            self.wait(1)

        eq1 = MathTex("Error", "=", "Bias^2", "+", "Variance", "+", "Noise", color=BLACK, font_size=40).move_to(callout_box).shift(UP*0.3)

        with self.voiceover(text="The total error of any model can be mathematically decomposed into three components: Bias squared, Variance, and Irreducible Noise.") as tracker:
            self.play(Write(eq1))
            self.wait(1.5)

        eq2 = MathTex("Error", "=", "\\text{Underfitting}", "+", "Variance", "+", "Noise", color=BLACK, font_size=40).move_to(callout_box).shift(UP*0.3)
        eq3 = MathTex("Error", "=", "\\text{Underfitting}", "+", "\\text{Overfitting}", "+", "Noise", color=BLACK, font_size=40).move_to(callout_box).shift(UP*0.3)

        with self.voiceover(text="High Bias corresponds to Underfitting. The model makes strong assumptions and fails to capture complexity.") as tracker:
            self.play(TransformMatchingTex(eq1, eq2))
            self.wait(1.5)

        with self.voiceover(text="High Variance corresponds to Overfitting. The model is highly sensitive to the specific training data and fluctuations.") as tracker:
            self.play(TransformMatchingTex(eq2, eq3))
            self.wait(1.5)

        tradeoff_text1 = Text("As you decrease Bias (make model complex),", color=BLACK, font_size=24).move_to(callout_box).shift(DOWN*0.7)
        tradeoff_text2 = Text("Variance increases.", color=RED, font_size=24).next_to(tradeoff_text1, DOWN, buff=0.2)

        with self.voiceover(text="The key insight to explain to your interviewer is the tradeoff. As you decrease bias by making a model more complex, you inherently increase its variance.") as tracker:
            self.play(Write(tradeoff_text1))
            self.play(Write(tradeoff_text2))
            self.wait(1.5)

        solutions_text = Text("Solutions: Regularization, More Data, Cross-Validation", color=BLUE, font_size=24).next_to(tradeoff_text2, DOWN, buff=0.5)

        with self.voiceover(text="Always conclude your answer by mentioning practical solutions to manage this tradeoff, such as L1 or L2 regularization, gathering more diverse training data, and using cross-validation. This demonstrates true engineering maturity.") as tracker:
            self.play(FadeIn(solutions_text))
            self.wait(2)

        self.play(FadeOut(VGroup(insight_title, callout_box, insight_heading, eq3, tradeoff_text1, tradeoff_text2, solutions_text, title)))

        final_text = Text("Good luck with your interviews!", color=BLACK, font_size=48)
        with self.voiceover(text="Keep practicing, and good luck with your AI engineering interviews!") as tracker:
            self.play(Write(final_text))
            self.wait(2)
            self.play(FadeOut(final_text))
