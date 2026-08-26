from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class GradientDescentOptimizers(VoiceoverScene):
    def construct(self):
        # 4. Aesthetic (Whiteboard Style): Set background to white
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Title Screen
        title = Text("Gradient Descent:", font_size=48, color=BLACK)
        subtitle = Text("SGD vs RMSProp vs Adam", font_size=36, color=BLUE)
        VGroup(title, subtitle).arrange(DOWN, buff=0.5)

        with self.voiceover(text="Welcome to our comprehensive deep dive into Gradient Descent optimizers. Over the next several minutes, we will explore the critical differences between Stochastic Gradient Descent, RMSProp, and Adam.") as tracker:
            self.play(Write(title))
            self.play(FadeIn(subtitle))
            self.wait(1.5)

        with self.voiceover(text="These optimization algorithms are the foundational engines that train modern neural networks. Understanding their internal mechanisms, strengths, and weaknesses is absolutely essential for any AI engineer, especially when heading into advanced technical interviews.") as tracker:
            self.wait(1.5)

        with self.voiceover(text="Let us begin our exploration by establishing exactly what these algorithms do, and why the field has evolved beyond the simplest approaches.") as tracker:
            self.play(FadeOut(VGroup(title, subtitle)))

        # --- Section 1: What is it? ---
        section1_title = Text("What are Gradient Descent Optimizers?", font_size=36, color=BLACK).to_edge(UP)

        definition = Text(
            "Algorithms that adjust neural network weights to minimize loss.",
            font_size=24, color=BLACK, weight=BOLD
        ).next_to(section1_title, DOWN, buff=0.5)

        with self.voiceover(text="Let's start with a core question: What exactly are gradient descent optimizers? In the context of deep learning, they are mathematical algorithms that iteratively adjust the weights and biases of a neural network in order to minimize a specific loss function.") as tracker:
            self.play(Write(section1_title))
            self.play(FadeIn(definition))
            self.wait(1.5)

        # Visual for "What is it?" - A 2D contour plot analogy
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 5, 1],
            x_length=6,
            y_length=4,
            axis_config={"color": BLACK, "include_numbers": False},
        ).shift(DOWN * 1)

        labels = axes.get_axis_labels(x_label=Tex("$w$", color=BLACK), y_label=Tex("$L(w)$", color=BLACK))

        parabola = axes.plot(lambda x: 0.5 * x**2, color=BLUE)

        start_point = axes.c2p(-2.5, 0.5 * (-2.5)**2)
        dot = Dot(start_point, color=RED, radius=0.1)

        with self.voiceover(text="Imagine you are standing near the peak of a foggy mountain, and your singular objective is to reach the lowest possible point in the valley below. In this analogy, the altitude represents your model's error, or loss, and your geographic coordinates represent the model's parameters or weights.") as tracker:
            self.play(Write(axes), FadeIn(labels))
            self.play(Write(parabola))
            self.play(FadeIn(dot))
            self.wait(1.5)

        with self.voiceover(text="Vanilla Gradient Descent calculates the slope, or gradient, of the entire mountain using all your data at once, and takes a deliberate step downhill. However, what if your dataset contains millions of images? Computing the gradient over the entire dataset for a single step is computationally prohibitive.") as tracker:
            self.wait(1.5)

        with self.voiceover(text="To solve this, we introduce Stochastic Gradient Descent, or SGD. Instead of using the whole dataset, SGD takes steps using only a single random data point, or a small batch of data points, called a mini-batch. This makes each step faster, though significantly noisier.") as tracker:
            # Simulate SGD steps (noisy path)
            steps = [-2.5, -1.8, -2.1, -1.0, -1.3, -0.4, 0.1, 0.0]
            path = VGroup()
            prev_point = start_point

            for i, x in enumerate(steps[1:]):
                next_point = axes.c2p(x, 0.5 * x**2)
                line = Arrow(prev_point, next_point, buff=0, color=RED, stroke_width=2, max_tip_length_to_length_ratio=0.15)
                path.add(line)
                prev_point = next_point

            self.play(Write(path, run_time=4))
            self.wait(1.5)

        with self.voiceover(text="RMSProp and Adam build upon this basic concept by dynamically adapting the step size for each individual weight. They maintain a moving average of past gradients, which allows them to speed up movement in flat regions and slow down where the slope is dangerously steep.") as tracker:
            self.wait(2)

        with self.voiceover(text="Let's remove this basic visualization and examine why these adaptive methods are so crucial.") as tracker:
            self.play(FadeOut(VGroup(axes, labels, parabola, dot, path, definition)))

        # --- Section 2: Why do we need it? ---
        section2_title = Text("Why do we need advanced optimizers?", font_size=36, color=BLACK).to_edge(UP)

        with self.voiceover(text="So, why do we need advanced optimizers like Adam or RMSProp in the first place? Why isn't simple Stochastic Gradient Descent always enough for deep learning models?") as tracker:
            self.play(Transform(section1_title, section2_title))
            self.wait(1.5)

        # Before vs After Visual (SGD stuck in saddle point vs Adam escaping)
        axes_left = Axes(
            x_range=[-2, 2, 1], y_range=[-1, 3, 1],
            x_length=4, y_length=3,
            axis_config={"color": BLACK, "include_numbers": False}
        ).shift(LEFT * 3 + DOWN * 0.5)

        axes_right = Axes(
            x_range=[-2, 2, 1], y_range=[-1, 3, 1],
            x_length=4, y_length=3,
            axis_config={"color": BLACK, "include_numbers": False}
        ).shift(RIGHT * 3 + DOWN * 0.5)

        title_left = Text("Without Adam (SGD)", font_size=24, color=RED).next_to(axes_left, UP)
        title_right = Text("With Adam", font_size=24, color=GREEN).next_to(axes_right, UP)

        curve_left = axes_left.plot(lambda x: x**3 if x > 0 else 0.1*x**3, color=BLACK)
        curve_right = axes_right.plot(lambda x: x**3 if x > 0 else 0.1*x**3, color=BLACK)

        dot_left = Dot(axes_left.c2p(-1.5, 0.1*(-1.5)**3), color=RED)
        dot_right = Dot(axes_right.c2p(-1.5, 0.1*(-1.5)**3), color=GREEN)

        with self.voiceover(text="Consider a complex loss landscape that contains flat regions or ravines, such as a saddle point. When we use standard SGD, the gradients calculated in these flat regions become infinitesimally small. Let's see this in action on the left.") as tracker:
            self.play(Write(axes_left), Write(axes_right), FadeIn(title_left), FadeIn(title_right))
            self.play(Write(curve_left), Write(curve_right))
            self.play(FadeIn(dot_left), FadeIn(dot_right))
            self.wait(1.5)

        with self.voiceover(text="Because the computed gradient is nearly zero, SGD takes incredibly tiny steps. It gets effectively stuck, learning almost nothing. It wastes precious compute time and potentially fails to converge to a meaningful solution entirely.") as tracker:
            stuck_path = VGroup()
            prev_p = axes_left.c2p(-1.5, 0.1*(-1.5)**3)
            for x in [-1.4, -1.35, -1.32, -1.31, -1.305]:
                new_p = axes_left.c2p(x, 0.1*(x)**3)
                stuck_path.add(Arrow(prev_p, new_p, buff=0, color=RED, max_tip_length_to_length_ratio=0.2))
                prev_p = new_p
            self.play(Write(stuck_path, run_time=3))
            self.wait(1.5)

        with self.voiceover(text="Now, look at what happens with Adam on the right side. Adam uses momentum, which acts much like a heavy ball rolling down a physical hill. It accumulates velocity from past gradients. Even if the current gradient is extremely small, the built-up momentum pushes the optimizer through the flat region.") as tracker:
            escape_path = VGroup()
            prev_p = axes_right.c2p(-1.5, 0.1*(-1.5)**3)
            for x in [-1.0, -0.2, 0.5, 1.2]:
                new_p = axes_right.c2p(x, x**3 if x > 0 else 0.1*x**3)
                escape_path.add(Arrow(prev_p, new_p, buff=0, color=GREEN, max_tip_length_to_length_ratio=0.15))
                prev_p = new_p
            self.play(Write(escape_path, run_time=3))
            self.wait(1.5)

        with self.voiceover(text="Furthermore, Adam intelligently divides the learning rate by an exponentially decaying average of past squared gradients. This means it automatically increases the step size for parameters with small, consistent gradients, and decreases the step size for parameters with large, erratic gradients. This stark contrast highlights precisely why advanced optimizers are necessary for deep, complex network architectures.") as tracker:
            self.wait(2)

        with self.voiceover(text="We'll now clear the screen to discuss where these algorithms are used in practice.") as tracker:
            self.play(
                FadeOut(VGroup(axes_left, axes_right, title_left, title_right,
                curve_left, curve_right, dot_left, dot_right,
                stuck_path, escape_path))
            )

        # --- Section 3: Use Cases ---
        section3_title = Text("Real-World Use Cases", font_size=36, color=BLACK).to_edge(UP)

        with self.voiceover(text="Let's ground this theoretical knowledge in reality by looking at where these specific algorithms are deployed in major production systems.") as tracker:
            self.play(Transform(section1_title, section3_title))
            self.wait(1.5)

        case1 = Text("1. Transformer Models (ChatGPT)", font_size=28, color=BLACK, weight=BOLD)
        desc1 = Text("Uses Adam (or AdamW) to handle highly non-convex\nlandscapes and sparse gradients in NLP.", font_size=24, color=PURPLE)
        group1 = VGroup(case1, desc1).arrange(DOWN, aligned_edge=LEFT).shift(UP * 1 + LEFT * 2)

        case2 = Text("2. Recommendation Systems (Spotify/Netflix)", font_size=28, color=BLACK, weight=BOLD)
        desc2 = Text("Often use RMSProp or Adagrad to handle sparse\nfeatures like user-item interactions effectively.", font_size=24, color=GREEN)
        group2 = VGroup(case2, desc2).arrange(DOWN, aligned_edge=LEFT).shift(DOWN * 1.5 + LEFT * 2)

        with self.voiceover(text="First, consider large language models like OpenAI's ChatGPT or Google's BERT. These massive transformer models use variations of Adam, specifically a variant called AdamW. They do this because transformer architectures feature highly non-convex loss landscapes and incredibly sparse gradients. Adam has become practically the default standard for natural language processing tasks.") as tracker:
            self.play(FadeIn(case1))
            self.play(Write(desc1))
            self.wait(1.5)

        with self.voiceover(text="Second, look at large-scale recommendation systems at companies like Spotify, Netflix, or Amazon. These systems deal with heavily sparse features, such as specific user-item interactions where the vast majority of users have never interacted with the vast majority of items. Here, RMSProp or Adagrad are heavily favored because they adjust learning rates per-parameter, ensuring that rare, yet informative features get meaningful weight updates without being drowned out by common features.") as tracker:
            self.play(FadeIn(case2))
            self.play(Write(desc2))
            self.wait(1.5)

        with self.voiceover(text="Understanding these industry standards will help you justify your architectural choices.") as tracker:
            self.play(FadeOut(VGroup(group1, group2)))

        # --- Section 4: Key Interview Insight ---
        section4_title = Text("Key Interview Insight", font_size=36, color=BLACK).to_edge(UP)

        with self.voiceover(text="Finally, let's discuss the most common, challenging interview question you will face regarding these optimizers.") as tracker:
            self.play(Transform(section1_title, section4_title))
            self.wait(1.5)

        box = Rectangle(width=10, height=4, color=RED, fill_opacity=0.05).shift(DOWN * 0.5)
        insight_title = Text("The Generalization Gap Tradeoff", font_size=30, color=RED, weight=BOLD).next_to(box.get_top(), DOWN, buff=0.2)

        insight_text = Text(
            "Adam converges FASTER, but standard SGD (with momentum)\noften generalizes BETTER on unseen validation data.",
            font_size=24, color=BLACK, line_spacing=1.5
        ).next_to(insight_title, DOWN, buff=0.5)

        with self.voiceover(text="Senior interviewers love to ask: If Adam is so incredibly fast and adaptive, why do researchers ever bother to use standard SGD? The correct answer is known as the Generalization Gap tradeoff.") as tracker:
            self.play(Write(box))
            self.play(FadeIn(insight_title))
            self.wait(1.5)

        with self.voiceover(text="While Adam converges much faster during the training phase, standard SGD with momentum often finds flatter, wider minima in the loss landscape. Research has repeatedly shown that these flatter minima tend to generalize significantly better to unseen validation data. Adam, on the other hand, can sometimes converge into sharp minima that memorize the training data but perform poorly in production.") as tracker:
            self.play(Write(insight_text))
            self.wait(1.5)

        # Math animation using TransformMatchingTex
        math_title = Text("Adam Update Rule Insight:", font_size=20, color=BLUE).next_to(insight_text, DOWN, buff=0.5).align_to(insight_text, LEFT)

        # We will animate the math step-by-step
        eq_step1 = MathTex("w_{t+1}", "=", "w_t", "-", "\\text{step}", color=BLACK).scale(0.8).next_to(math_title, RIGHT, buff=0.3)
        eq_step2 = MathTex("w_{t+1}", "=", "w_t", "-", "\\alpha \\cdot \\text{gradient}", color=BLACK).scale(0.8).next_to(math_title, RIGHT, buff=0.3)
        eq_step3 = MathTex("w_{t+1}", "=", "w_t", "-", "\\frac{\\alpha}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t", color=BLACK).scale(0.8).next_to(math_title, RIGHT, buff=0.3)

        with self.voiceover(text="If an interviewer asks you to write out Adam's update rule, you can break it down logically. First, we start with the basic weight update: the new weight equals the old weight minus a step size.") as tracker:
            self.play(FadeIn(math_title))
            self.play(Write(eq_step1))
            self.wait(1.5)

        with self.voiceover(text="In standard gradient descent, that step is just the learning rate alpha multiplied by the raw gradient.") as tracker:
            self.play(TransformMatchingTex(eq_step1, eq_step2))
            self.wait(1.5)

        with self.voiceover(text="But in Adam, we replace the raw gradient with a highly sophisticated term. It updates the weight using the learning rate alpha, divided by the square root of the bias-corrected second moment, v_t. This provides the adaptive learning rate per parameter. This is then multiplied by the bias-corrected first moment, m_t, which acts as the momentum. The tiny epsilon term in the denominator is simply there to prevent division by zero—a crucial, practical detail to mention to your interviewer.") as tracker:
            self.play(TransformMatchingTex(eq_step2, eq_step3))
            self.wait(1.5)

        with self.voiceover(text="Understanding this tradeoff—the speed of convergence versus the ultimate quality of generalization—demonstrates to a hiring manager that you don't just know how to call a PyTorch API, but that you deeply understand the underlying mathematics and engineering tradeoffs required to train robust neural networks.") as tracker:
            self.wait(2)

        with self.voiceover(text="Let's wrap up our session for today.") as tracker:
            self.play(FadeOut(VGroup(section1_title, box, insight_title, insight_text, math_title, eq_step3)))

        # Outro
        outro = Text("Happy Tuning!", font_size=48, color=BLUE)
        with self.voiceover(text="That concludes our extensive breakdown of Stochastic Gradient Descent, RMSProp, and Adam. Good luck with training your deep learning models and acing your upcoming machine learning interviews. Happy tuning!") as tracker:
            self.play(Write(outro))
            self.wait(2)

        with self.voiceover(text="See you in the next lesson.") as tracker:
            self.play(FadeOut(outro))
