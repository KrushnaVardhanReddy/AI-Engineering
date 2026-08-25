import numpy as np
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class BiasVarianceTradeoff(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Colors
        COLOR_PRIMARY = BLACK
        COLOR_BIAS = BLUE
        COLOR_VAR = RED
        COLOR_NOISE = GREEN
        COLOR_HIGHLIGHT = PURPLE

        # ---------------------------------------------------------
        # SECTION 1: What is it?
        # ---------------------------------------------------------
        title = Text("The Bias-Variance Tradeoff", color=COLOR_PRIMARY, font_size=48).to_edge(UP)
        with self.voiceover(text="Welcome to Day 19 of our AI Engineering Masterclass. Today, we are diving deep into one of the most fundamental concepts in machine learning and artificial intelligence: The Bias-Variance Tradeoff.") as tracker:
            self.play(Write(title))
            self.wait(1.5)

        def_text = Text("The balance between two sources of error\npreventing models from generalizing.", color=COLOR_PRIMARY, font_size=32).next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="What exactly is the Bias-Variance Tradeoff? Simply put, it is the delicate balance between two sources of error that prevent supervised learning algorithms from generalizing beyond their training data.") as tracker:
            self.play(FadeIn(def_text, shift=DOWN))
            self.wait(1.5)

        bias_def = MarkupText("<b>Bias:</b> Error from oversimplifying the problem (Underfitting).", color=COLOR_BIAS, font_size=28).next_to(def_text, DOWN, buff=1)
        var_def = MarkupText("<b>Variance:</b> Error from sensitivity to noise (Overfitting).", color=COLOR_VAR, font_size=28).next_to(bias_def, DOWN, buff=0.5)

        with self.voiceover(text="Let's formally define these terms. Bias is the error introduced by approximating a real-world problem, which may be highly complex, by a much simpler model. High bias means the model is paying very little attention to the training data and oversimplifies the problem. This is also known as underfitting.") as tracker:
            self.play(Write(bias_def))
            self.wait(1.5)

        with self.voiceover(text="On the other hand, Variance is the error introduced by the model's sensitivity to small fluctuations in the training set. A model with high variance pays too much attention to the training data, capturing random noise as if it were a true signal. This is known as overfitting.") as tracker:
            self.play(Write(var_def))
            self.wait(1.5)

        self.play(FadeOut(def_text), FadeOut(bias_def), FadeOut(var_def))

        # Bullseye diagrams
        def create_bullseye(center_pos, label_text, darts_spread, darts_offset):
            group = VGroup()
            colors = [COLOR_PRIMARY, WHITE, COLOR_PRIMARY, WHITE, COLOR_VAR]
            radii = [1.0, 0.8, 0.6, 0.4, 0.2]
            for r, c in zip(radii, colors):
                circle = Circle(radius=r, color=COLOR_PRIMARY, fill_color=c, fill_opacity=1 if c != WHITE else 0).move_to(center_pos)
                if c == WHITE:
                    circle.set_fill(WHITE, opacity=1)
                group.add(circle)

            label = Text(label_text, color=COLOR_PRIMARY, font_size=20).next_to(group, DOWN)
            group.add(label)

            darts = VGroup()
            import random
            random.seed(42) # Deterministic
            for _ in range(7):
                dx = random.uniform(-darts_spread, darts_spread)
                dy = random.uniform(-darts_spread, darts_spread)
                dart = Dot(color=COLOR_HIGHLIGHT, radius=0.05).move_to(center_pos + np.array([darts_offset[0] + dx, darts_offset[1] + dy, 0]))
                darts.add(dart)
            group.add(darts)
            return group, darts

        b1, d1 = create_bullseye(np.array([-4, -1, 0]), "Low Bias, Low Variance", 0.15, [0, 0])
        b2, d2 = create_bullseye(np.array([-1.33, -1, 0]), "High Bias, Low Variance", 0.15, [0.6, 0.6])
        b3, d3 = create_bullseye(np.array([1.33, -1, 0]), "Low Bias, High Variance", 0.6, [0, 0])
        b4, d4 = create_bullseye(np.array([4, -1, 0]), "High Bias, High Variance", 0.6, [0.6, 0.6])

        bullseyes = VGroup(b1, b2, b3, b4)

        with self.voiceover(text="To visualize this, imagine a dartboard. The center of the target represents a model that perfectly predicts the correct values. Low bias and low variance means all our darts hit the bullseye. This is our ultimate goal.") as tracker:
            self.play(FadeIn(b1))
            self.wait(1.5)

        with self.voiceover(text="If we have high bias but low variance, our darts are clustered together, but far from the bullseye. The model is consistent, but consistently wrong.") as tracker:
            self.play(FadeIn(b2))
            self.wait(1.5)

        with self.voiceover(text="If we have low bias but high variance, the darts are scattered around the bullseye. On average they might center on the target, but any single prediction could be wildly off.") as tracker:
            self.play(FadeIn(b3))
            self.wait(1.5)

        with self.voiceover(text="And finally, high bias and high variance means the darts are scattered far from the target. The worst of both worlds.") as tracker:
            self.play(FadeIn(b4))
            self.wait(1.5)

        self.play(FadeOut(bullseyes))

        # Math Equation
        eq_text = Text("Expected Test Error Decomposition:", color=COLOR_PRIMARY, font_size=32).next_to(title, DOWN, buff=0.5)

        eq1 = MathTex("Err(x)", "=", "E[(Y - \\hat{f}(x))^2]", color=COLOR_PRIMARY, font_size=40)
        eq2 = MathTex("Err(x)", "=", "\\text{Bias}^2", "+", "\\text{Variance}", "+", "\\text{Irreducible Error}", color=COLOR_PRIMARY, font_size=40)
        eq2[2].set_color(COLOR_BIAS)
        eq2[4].set_color(COLOR_VAR)
        eq2[6].set_color(COLOR_NOISE)

        with self.voiceover(text="The mathematical foundation of this is the error decomposition: The expected test error of our model squared error loss can be written mathematically.") as tracker:
            self.play(Write(eq_text))
            self.play(Write(eq1))
            self.wait(1.5)

        with self.voiceover(text="This error expands directly into three components: Bias squared, plus Variance, plus Irreducible Error which is just the inherent noise in the data itself.") as tracker:
            self.play(TransformMatchingTex(eq1, eq2))
            self.wait(1.5)

        self.play(FadeOut(title), FadeOut(eq_text), FadeOut(eq2))

        # ---------------------------------------------------------
        # SECTION 2: Why do we need it?
        # ---------------------------------------------------------
        title2 = Text("Why do we need it?", color=COLOR_PRIMARY, font_size=48).to_edge(UP)
        with self.voiceover(text="So, why do we actually need to understand this tradeoff? Why can't we just build models with zero bias and zero variance? The answer lies in the problem of generalization.") as tracker:
            self.play(Write(title2))
            self.wait(1.5)

        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 20, 5],
            axis_config={"color": COLOR_PRIMARY, "include_numbers": False, "tip_shape": StealthTip}
        ).scale(0.6).shift(LEFT * 3 + DOWN * 0.5)
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y").set_color(COLOR_PRIMARY)

        # Data points
        x_vals = np.linspace(0, 4, 15)
        np.random.seed(42)
        y_vals = x_vals**2 + np.random.normal(0, 2, 15)
        dots = VGroup(*[Dot(axes.c2p(x, y), color=COLOR_PRIMARY) for x, y in zip(x_vals, y_vals)])

        with self.voiceover(text="Let's look at a concrete example. Suppose we have some data points representing housing prices over time. Our goal is to predict future prices based on this historical data.") as tracker:
            self.play(Create(axes), Write(axes_labels))
            self.play(FadeIn(dots, lag_ratio=0.1))
            self.wait(1.5)

        # Underfit line
        underfit_line = axes.plot(lambda x: 3*x, color=COLOR_BIAS)
        underfit_label = Text("High Bias (Underfitting)", color=COLOR_BIAS, font_size=24).next_to(underfit_line, UP)

        with self.voiceover(text="First, let's see what happens without understanding this concept. If we fit a simple straight line, a linear regression model, it fails to capture the underlying curve of the data. The model is too rigid. This is High Bias. It's underfitting the data, and its predictions will be poor.") as tracker:
            self.play(Create(underfit_line), Write(underfit_label))
            self.wait(1.5)

        # Overfit line
        self.play(FadeOut(underfit_line), FadeOut(underfit_label))

        # Wiggly line through points
        overfit_line = VMobject(color=COLOR_VAR)
        pts = [axes.c2p(x, y) for x, y in zip(x_vals, y_vals)]
        overfit_line.set_points_smoothly(pts)
        overfit_label = Text("High Variance (Overfitting)", color=COLOR_VAR, font_size=24).next_to(axes, UP).shift(RIGHT)

        with self.voiceover(text="Now, suppose we decide to use a highly complex polynomial model to hit every single data point perfectly. Look at this wild, wiggly line. It has zero error on the training data! But it's capturing the random noise. This is High Variance. On new data, it will fail spectacularly.") as tracker:
            self.play(Create(overfit_line), Write(overfit_label))
            self.wait(1.5)

        # Sweet spot
        self.play(FadeOut(overfit_line), FadeOut(overfit_label))

        sweet_line = axes.plot(lambda x: x**2, color=COLOR_HIGHLIGHT)
        sweet_label = Text("Optimal Tradeoff", color=COLOR_HIGHLIGHT, font_size=24).next_to(sweet_line, UP)

        with self.voiceover(text="This is why we need the tradeoff. We must find the sweet spot in model complexity. We want a smooth curve that captures the true underlying pattern, ignoring the noise but remaining flexible enough to learn.") as tracker:
            self.play(Create(sweet_line), Write(sweet_label))
            self.wait(1.5)

        # Complexity vs Error Graph
        axes_comp = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": COLOR_PRIMARY, "include_numbers": False, "tip_shape": StealthTip}
        ).scale(0.6).shift(RIGHT * 3 + DOWN * 0.5)
        axes_comp_labels = axes_comp.get_axis_labels(x_label="Complexity", y_label="Error").set_color(COLOR_PRIMARY)

        train_err = axes_comp.plot(lambda x: 8 * np.exp(-0.4 * x) + 0.5, color=COLOR_BIAS, x_range=[0, 9])
        test_err = axes_comp.plot(lambda x: 8 * np.exp(-0.4 * x) + 0.1 * (x - 4)**2 + 1, color=COLOR_VAR, x_range=[0, 9])

        train_label = Text("Training Error", color=COLOR_BIAS, font_size=20).next_to(axes_comp.c2p(8, 8 * np.exp(-0.4 * 8) + 0.5), DOWN)
        test_label = Text("Test Error", color=COLOR_VAR, font_size=20).next_to(axes_comp.c2p(8, 8 * np.exp(-0.4 * 8) + 0.1 * (8 - 4)**2 + 1), UP)

        with self.voiceover(text="We can visualize this relationship on a graph comparing Model Complexity against Prediction Error. As model complexity increases, our Training Error consistently goes down. The model gets better at memorizing the training data.") as tracker:
            self.play(Create(axes_comp), Write(axes_comp_labels))
            self.play(Create(train_err), Write(train_label))
            self.wait(1.5)

        with self.voiceover(text="However, the Test Error, which represents performance on new data, initially decreases, but then starts to increase again as the model begins to overfit.") as tracker:
            self.play(Create(test_err), Write(test_label))
            self.wait(1.5)

        optimum_x = 5.5
        optimum_line = DashedLine(axes_comp.c2p(optimum_x, 0), axes_comp.c2p(optimum_x, 10), color=COLOR_HIGHLIGHT)
        opt_label = Text("Sweet Spot", color=COLOR_HIGHLIGHT, font_size=20).next_to(optimum_line, UP)

        with self.voiceover(text="The Bias-Variance Tradeoff dictates that we must choose the model complexity that minimizes the total Test Error, right at the bottom of this U-shaped curve.") as tracker:
            self.play(Create(optimum_line), Write(opt_label))
            self.wait(2)

        self.play(FadeOut(VGroup(axes, axes_labels, dots, sweet_line, sweet_label, axes_comp, axes_comp_labels, train_err, test_err, train_label, test_label, optimum_line, opt_label, title2)))

        # ---------------------------------------------------------
        # SECTION 3: Use Cases
        # ---------------------------------------------------------
        title3 = Text("Real-World Use Cases", color=COLOR_PRIMARY, font_size=48).to_edge(UP)
        with self.voiceover(text="Understanding this tradeoff is crucial in real-world AI applications. Let's look at two specific use cases from the industry.") as tracker:
            self.play(Write(title3))
            self.wait(1)

        uc1_title = Text("1. Spotify Music Recommendation", color=COLOR_HIGHLIGHT, font_size=32).shift(UP * 2)
        uc1_desc1 = Text("Simple Linear Model (High Bias)", color=COLOR_BIAS, font_size=24).next_to(uc1_title, DOWN, buff=0.5)
        uc1_desc2 = Text("Recommends only Top 50 pop songs, ignoring niche tastes.", color=COLOR_PRIMARY, font_size=24).next_to(uc1_desc1, DOWN, buff=0.2)

        with self.voiceover(text="First, let's consider Spotify's music recommendation engine. If Spotify used a very simple, high-bias model, like recommending only the global top 50 pop songs to every user, it would severely underfit. It completely ignores individual user tastes and niche preferences.") as tracker:
            self.play(Write(uc1_title))
            self.play(FadeIn(uc1_desc1), FadeIn(uc1_desc2))
            self.wait(1.5)

        uc1_desc3 = Text("Deep Learning / Collab Filtering (Higher Variance)", color=COLOR_VAR, font_size=24).next_to(uc1_desc2, DOWN, buff=0.5)
        uc1_desc4 = Text("Captures complex, personalized user-item interactions.", color=COLOR_PRIMARY, font_size=24).next_to(uc1_desc3, DOWN, buff=0.2)

        with self.voiceover(text="To solve this, they use collaborative filtering and deep learning models with higher variance capacity to capture complex user-item interactions.") as tracker:
            self.play(FadeIn(uc1_desc3), FadeIn(uc1_desc4))
            self.wait(1.5)

        self.play(FadeOut(uc1_title, uc1_desc1, uc1_desc2, uc1_desc3, uc1_desc4))

        uc2_title = Text("2. Medical Diagnosis & LLMs", color=COLOR_HIGHLIGHT, font_size=32).shift(UP * 2)
        uc2_desc1 = Text("Deep Decision Tree (High Variance)", color=COLOR_VAR, font_size=24).next_to(uc2_title, DOWN, buff=0.5)
        uc2_desc2 = Text("Memorizes patient records exactly, fails on new patients.", color=COLOR_PRIMARY, font_size=24).next_to(uc2_desc1, DOWN, buff=0.2)

        with self.voiceover(text="Our second example involves medical diagnosis systems or even Large Language Models like ChatGPT. Imagine training a deep Decision Tree on a hospital's patient records. If the tree is allowed to grow infinitely deep, it will perfectly memorize every patient's symptoms, which is high variance. But when a new patient arrives with a slight variation, it misdiagnoses them.") as tracker:
            self.play(Write(uc2_title))
            self.play(FadeIn(uc2_desc1), FadeIn(uc2_desc2))
            self.wait(1.5)

        uc2_desc3 = Text("Random Forest (Ensemble Technique)", color=COLOR_BIAS, font_size=24).next_to(uc2_desc2, DOWN, buff=0.5)
        uc2_desc4 = Text("Averages many trees to reduce variance, keeping bias low.", color=COLOR_PRIMARY, font_size=24).next_to(uc2_desc3, DOWN, buff=0.2)

        with self.voiceover(text="To fix this overfitting, machine learning engineers use Random Forests. By training many deep decision trees on different subsets of data and averaging their predictions, Random Forests significantly reduce variance while keeping the low bias of the individual trees. This ensemble technique is a direct application of managing the bias-variance tradeoff.") as tracker:
            self.play(FadeIn(uc2_desc3), FadeIn(uc2_desc4))
            self.wait(1.5)

        self.play(FadeOut(uc2_title, uc2_desc1, uc2_desc2, uc2_desc3, uc2_desc4, title3))

        # ---------------------------------------------------------
        # SECTION 4: Key Interview Insight
        # ---------------------------------------------------------
        title4 = Text("Key Interview Insight", color=COLOR_PRIMARY, font_size=48).to_edge(UP)
        with self.voiceover(text="Now, let's discuss the most critical part for your career: The Key Interview Insight. When you are interviewing for an AI Engineering or Data Science role, interviewers love to test your practical intuition of this tradeoff.") as tracker:
            self.play(Write(title4))
            self.wait(1)

        q_text = Text("\"How does regularization or adding data affect this tradeoff?\"", color=COLOR_VAR, font_size=32).next_to(title4, DOWN, buff=0.5)
        with self.voiceover(text="Here is the most common gotcha they will throw at you. They will ask: 'How does adding regularization to your model affect bias and variance?' or 'Will adding more training data solve a high bias problem?'") as tracker:
            self.play(FadeIn(q_text))
            self.wait(1.5)

        # Callout Box
        box = Rectangle(width=12, height=3, color=COLOR_HIGHLIGHT, fill_color=WHITE, fill_opacity=1, stroke_width=4).shift(DOWN)
        box_title = Text("INTERVIEW INSIGHT", color=COLOR_HIGHLIGHT, font_size=28, weight=BOLD).next_to(box.get_top(), DOWN, buff=0.2)

        insight1 = MarkupText("<b>1. Regularization (L1/L2):</b> Deliberately <b>increases Bias</b>\nto significantly <b>decrease Variance</b>.", color=COLOR_PRIMARY, font_size=24).next_to(box_title, DOWN, buff=0.3).align_to(box.get_left() + RIGHT*0.5, LEFT)

        insight2 = MarkupText("<b>2. More Training Data:</b> Decreases Variance (harder to memorize noise),\nbut <b>does NOT fix High Bias</b> (model is still too simple).", color=COLOR_PRIMARY, font_size=24).next_to(insight1, DOWN, buff=0.3).align_to(box.get_left() + RIGHT*0.5, LEFT)

        with self.voiceover(text="Let's put the answer in a dedicated callout box for you to remember.") as tracker:
            self.play(Create(box), Write(box_title))
            self.wait(1)

        with self.voiceover(text="First Insight: Regularization, such as L1 or L2 penalties, deliberately increases Bias in order to significantly decrease Variance. By forcing the model's weights to be smaller, you are restricting its complexity, moving left on the complexity curve to escape overfitting.") as tracker:
            self.play(FadeIn(insight1))
            self.wait(1.5)

        with self.voiceover(text="Second Insight: Adding more training data will reduce Variance, because it becomes harder for the model to memorize the noise in a larger dataset. However, adding more data will NOT fix a high bias problem. If your model is a simple straight line, giving it a million data points will still result in a straight line.") as tracker:
            self.play(FadeIn(insight2))
            self.wait(1.5)

        summary = Text("Underfitting -> More Complex Model\nOverfitting -> More Data / Regularization", color=COLOR_PRIMARY, font_size=28).next_to(box, DOWN, buff=0.5)

        with self.voiceover(text="Remember this golden rule: If your model is underfitting, you need a more complex model or better features. If it is overfitting, you need more data, regularization, or a simpler model.") as tracker:
            self.play(Write(summary))
            self.wait(1.5)

        with self.voiceover(text="Mastering this intuition is what separates junior practitioners from senior AI engineers. Thank you for joining Day 19. Keep building, and see you in the next lesson!") as tracker:
            self.wait(3)
            self.play(FadeOut(Group(*self.mobjects)))
