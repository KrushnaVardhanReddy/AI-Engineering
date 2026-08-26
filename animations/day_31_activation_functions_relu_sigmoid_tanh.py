from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np

class ActivationFunctions(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Title
        title = Text("Activation Functions", color=BLACK).scale(1.2).to_edge(UP)
        subtitle = Text("ReLU, Sigmoid, and Tanh", color=BLUE).scale(0.8).next_to(title, DOWN)

        with self.voiceover(text="Welcome to Day 31 of AI Engineering Mastery. Today, we are diving deep into Activation Functions, specifically ReLU, Sigmoid, and Tanh. We will explore what they are, why they are absolutely crucial for neural networks, real-world use cases, and the most important insights you need to know for your AI interviews.") as tracker:
            self.play(Write(title))
            self.play(FadeIn(subtitle))
            self.wait(1.5)

        with self.voiceover(text="Let's begin.") as tracker:
            self.play(FadeOut(title), FadeOut(subtitle))

        # --- Section 1: What is it? ---
        section1_title = Text("1. What is an Activation Function?", color=BLACK).scale(1.1).to_edge(UP)
        definition_text = Text(
            "A mathematical gate that decides whether a neuron should 'fire' or not.",
            color=BLACK, font_size=28
        ).next_to(section1_title, DOWN, buff=0.5)

        with self.voiceover(text="Let us start with a fundamental question: What is an activation function? Simply put, it is a mathematical gate that sits at the end of a neuron. It takes the weighted sum of inputs and decides whether the neuron should activate, or 'fire', passing its signal to the next layer in the network. You can think of it as a decision-maker.") as tracker:
            self.play(Write(section1_title))
            self.play(FadeIn(definition_text))
            self.wait(1.5)

        # Neuron Visual
        circle = Circle(radius=1.0, color=BLACK, fill_opacity=0.1, fill_color=BLUE).shift(LEFT * 2)

        eq_parts = [r"\Sigma", r"(w", r"\cdot", r"x", r"+", r"b)"]
        sigma_base = MathTex(*eq_parts, color=BLACK).move_to(circle.get_center())
        sigma = MathTex(r"\Sigma (w_i x_i + b)", color=BLACK).move_to(circle.get_center())

        arrow_in1 = Arrow(start=LEFT * 5 + UP * 1, end=circle.get_left() + UP * 0.5, color=BLACK)
        arrow_in2 = Arrow(start=LEFT * 5 + DOWN * 1, end=circle.get_left() + DOWN * 0.5, color=BLACK)
        x1_label = MathTex("x_1", color=BLACK).next_to(arrow_in1, LEFT)
        x2_label = MathTex("x_2", color=BLACK).next_to(arrow_in2, LEFT)

        activation_box = Rectangle(width=2, height=1, color=BLUE).next_to(circle, RIGHT, buff=1)
        f_x = MathTex("f(z)", color=BLACK).move_to(activation_box.get_center())

        arrow_mid = Arrow(start=circle.get_right(), end=activation_box.get_left(), color=BLACK)
        z_label = MathTex("z", color=BLACK).next_to(arrow_mid, UP, buff=0.1)

        arrow_out = Arrow(start=activation_box.get_right(), end=RIGHT * 5, color=BLACK)
        y_label = MathTex("y = f(z)", color=BLACK).next_to(arrow_out, RIGHT)

        with self.voiceover(text="Here is a visual representation. Inputs x1 and x2 enter the neuron, multiplied by their weights. Inside the neuron, we calculate the linear combination, the sum of weights times inputs plus a bias. Let's call this intermediate result z.") as tracker:
            self.play(Create(arrow_in1), Create(arrow_in2), Write(x1_label), Write(x2_label))
            self.play(Create(circle), Write(sigma_base))
            self.play(TransformMatchingTex(sigma_base, sigma))
            self.play(Create(arrow_mid), Write(z_label))
            self.wait(1.5)

        with self.voiceover(text="This raw value z is then passed into our activation function, represented by f of z. The function transforms z into the final output y. Without this step, the neuron would just output a simple linear calculation, no matter how complex the input data is.") as tracker:
            self.play(Create(activation_box), Write(f_x))
            self.play(Create(arrow_out), Write(y_label))
            self.wait(1.5)

        with self.voiceover(text="Let's move on to the next section to understand this concept deeper.") as tracker:
            self.play(
                FadeOut(section1_title), FadeOut(definition_text),
                FadeOut(circle), FadeOut(sigma), FadeOut(arrow_in1), FadeOut(arrow_in2),
                FadeOut(x1_label), FadeOut(x2_label), FadeOut(activation_box), FadeOut(f_x),
                FadeOut(arrow_mid), FadeOut(z_label), FadeOut(arrow_out), FadeOut(y_label)
            )

        # --- Section 2: Why do we need it? ---
        section2_title = Text("2. Why do we need it?", color=BLACK).scale(1.1).to_edge(UP)
        why_text = Text(
            "To introduce non-linearity into the network.",
            color=BLUE, font_size=32
        ).next_to(section2_title, DOWN, buff=0.5)

        with self.voiceover(text="So, why do we need activation functions? The single most important reason is to introduce non-linearity into the network. This is a crucial concept.") as tracker:
            self.play(Write(section2_title))
            self.play(FadeIn(why_text))
            self.wait(1.5)

        # Before (Linear) vs After (Non-linear)
        axes_linear = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=4, y_length=4,
            axis_config={"color": BLACK}
        ).shift(LEFT * 3 + DOWN * 1)
        axes_nonlinear = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=4, y_length=4,
            axis_config={"color": BLACK}
        ).shift(RIGHT * 3 + DOWN * 1)

        label_linear = Text("Without Activation (Linear)", color=RED, font_size=24).next_to(axes_linear, UP)
        label_nonlinear = Text("With Activation (Non-linear)", color=GREEN, font_size=24).next_to(axes_nonlinear, UP)

        line_linear = axes_linear.plot(lambda x: 0.5 * x + 0.5, color=RED)

        # Draw points manually to avoid lambda caching issues with complex functions
        curve_pts = [axes_nonlinear.c2p(x, np.sin(x) + 0.5 * x) for x in np.linspace(-3, 3, 50)]
        curve_nonlinear = VMobject(color=GREEN).set_points_smoothly(curve_pts)

        with self.voiceover(text="Consider a neural network without any activation functions. No matter how many layers or neurons you stack together, a linear function of a linear function is still just a linear function. The entire network collapses into the equivalent of a single linear regression model. It can only draw straight lines to separate data, which is completely useless for complex, real-world tasks like image recognition or natural language processing.") as tracker:
            self.play(Create(axes_linear), Write(label_linear))
            self.play(Create(line_linear))
            self.wait(1.5)

        with self.voiceover(text="However, when we introduce non-linear activation functions after each layer, the network gains the ability to learn complex, curvy representations. It can now approximate any continuous function, a property known as the Universal Approximation Theorem. This is the secret sauce that makes deep learning so incredibly powerful.") as tracker:
            self.play(Create(axes_nonlinear), Write(label_nonlinear))
            self.play(Create(curve_nonlinear))
            self.wait(1.5)

        with self.voiceover(text="Now let's explore the three most common activation functions used today.") as tracker:
            self.play(
                FadeOut(section2_title), FadeOut(why_text),
                FadeOut(axes_linear), FadeOut(label_linear), FadeOut(line_linear),
                FadeOut(axes_nonlinear), FadeOut(label_nonlinear), FadeOut(curve_nonlinear)
            )

        # --- Section 3: The Big Three ---
        section3_title = Text("3. The Big Three: ReLU, Sigmoid, Tanh", color=BLACK).scale(1.1).to_edge(UP)

        with self.voiceover(text="We call them the big three.") as tracker:
            self.play(Write(section3_title))

        # ReLU
        relu_axes = Axes(x_range=[-3, 3, 1], y_range=[-1, 3, 1], x_length=4, y_length=3, axis_config={"color": BLACK}).shift(LEFT * 4 + UP * 0.5)
        relu_pts = [relu_axes.c2p(x, max(0, x)) for x in np.linspace(-3, 3, 100)]
        relu_curve = VMobject(color=BLUE).set_points_as_corners(relu_pts)
        relu_label = Text("ReLU", color=BLUE, font_size=28).next_to(relu_axes, UP)
        relu_eq = MathTex(r"f(x) = \max(0, x)", color=BLACK, font_size=24).next_to(relu_axes, DOWN)

        with self.voiceover(text="Let's look at the three most common activation functions. First is ReLU, which stands for Rectified Linear Unit. The equation is incredibly simple: f of x equals the maximum of zero and x. If the input is negative, it outputs zero. If positive, it passes the value through unchanged.") as tracker:
            self.play(Create(relu_axes), Write(relu_label))
            self.play(Create(relu_curve), Write(relu_eq))
            self.wait(1.5)

        # Sigmoid
        sig_axes = Axes(x_range=[-5, 5, 1], y_range=[-0.2, 1.2, 0.5], x_length=4, y_length=3, axis_config={"color": BLACK}).shift(RIGHT * 0 + UP * 0.5)
        sig_pts = [sig_axes.c2p(x, 1 / (1 + np.exp(-x))) for x in np.linspace(-5, 5, 100)]
        sig_curve = VMobject(color=RED).set_points_smoothly(sig_pts)
        sig_label = Text("Sigmoid", color=RED, font_size=28).next_to(sig_axes, UP)
        sig_eq = MathTex(r"f(x) = \frac{1}{1 + e^{-x}}", color=BLACK, font_size=24).next_to(sig_axes, DOWN)

        with self.voiceover(text="Next is the Sigmoid function. It squashes any real-valued number into a range strictly between zero and one. Because it maps values to probabilities, it was historically very popular.") as tracker:
            self.play(Create(sig_axes), Write(sig_label))
            self.play(Create(sig_curve), Write(sig_eq))
            self.wait(1.5)

        # Tanh
        tanh_axes = Axes(x_range=[-5, 5, 1], y_range=[-1.2, 1.2, 1], x_length=4, y_length=3, axis_config={"color": BLACK}).shift(RIGHT * 4 + UP * 0.5)
        tanh_pts = [tanh_axes.c2p(x, np.tanh(x)) for x in np.linspace(-5, 5, 100)]
        tanh_curve = VMobject(color=GREEN).set_points_smoothly(tanh_pts)
        tanh_label = Text("Tanh", color=GREEN, font_size=28).next_to(tanh_axes, UP)
        tanh_eq = MathTex(r"f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}", color=BLACK, font_size=24).next_to(tanh_axes, DOWN)

        with self.voiceover(text="Finally, we have Tanh, or the hyperbolic tangent. It is mathematically very similar to Sigmoid, but it squashes values between negative one and positive one. It is zero-centered, which often makes optimization easier than with Sigmoid.") as tracker:
            self.play(Create(tanh_axes), Write(tanh_label))
            self.play(Create(tanh_curve), Write(tanh_eq))
            self.wait(1.5)

        with self.voiceover(text="Moving onto some practical applications.") as tracker:
            self.play(
                FadeOut(section3_title),
                FadeOut(relu_axes), FadeOut(relu_curve), FadeOut(relu_label), FadeOut(relu_eq),
                FadeOut(sig_axes), FadeOut(sig_curve), FadeOut(sig_label), FadeOut(sig_eq),
                FadeOut(tanh_axes), FadeOut(tanh_curve), FadeOut(tanh_label), FadeOut(tanh_eq)
            )

        # --- Section 4: Real-world Use Cases ---
        section4_title = Text("4. Real-world Use Cases", color=BLACK).scale(1.1).to_edge(UP)

        uc1_title = Text("Use Case 1: ChatGPT (Transformers)", color=BLUE, font_size=32).shift(UP * 1 + LEFT * 2)
        uc1_desc = Text("- Uses GELU/ReLU in feed-forward layers\n- Fast to compute, prevents vanishing gradients", color=BLACK, font_size=24).next_to(uc1_title, DOWN, aligned_edge=LEFT)

        uc2_title = Text("Use Case 2: Spotify (Recommenders)", color=GREEN, font_size=32).next_to(uc1_desc, DOWN, buff=1, aligned_edge=LEFT)
        uc2_desc = Text("- Uses Sigmoid for binary classification\n- E.g., 'Will the user like this song?' (0 to 1)", color=BLACK, font_size=24).next_to(uc2_title, DOWN, aligned_edge=LEFT)

        with self.voiceover(text="Let's ground this in reality with some real-world use cases. First, consider Large Language Models like ChatGPT, which are based on the Transformer architecture. Deep inside these models, in their feed-forward layers, they heavily rely on ReLU or its smooth variant, GELU. Why? Because these functions are computationally incredibly fast, and crucially, they help prevent the vanishing gradient problem in very deep networks.") as tracker:
            self.play(Write(section4_title))
            self.play(FadeIn(uc1_title))
            self.play(Write(uc1_desc))
            self.wait(1.5)

        with self.voiceover(text="For a second use case, look at recommendation systems like the ones used by Spotify. When the system wants to predict if a user will skip a song or like it, that is a binary classification problem. The final output layer will typically use a Sigmoid activation function to squash the network's prediction into a neat probability between zero and one, representing the likelihood of a 'like'.") as tracker:
            self.play(FadeIn(uc2_title))
            self.play(Write(uc2_desc))
            self.wait(1.5)

        with self.voiceover(text="Now let's review the most critical concept for interviews.") as tracker:
            self.play(
                FadeOut(section4_title), FadeOut(uc1_title), FadeOut(uc1_desc),
                FadeOut(uc2_title), FadeOut(uc2_desc)
            )

        # --- Section 5: Key Interview Insight ---
        section5_title = Text("5. Key Interview Insight", color=BLACK).scale(1.1).to_edge(UP)

        box = Rectangle(width=10, height=4, color=RED, fill_opacity=0.05).move_to(ORIGIN)
        insight_title = Text("The Vanishing Gradient Problem", color=RED, font_size=36).next_to(box.get_top(), DOWN, buff=0.3)

        insight_text_1 = Text("Sigmoid and Tanh gradients approach 0 at the extremes.", color=BLACK, font_size=26).next_to(insight_title, DOWN, buff=0.4)
        insight_text_2 = Text("If gradients are 0, weights don't update. Learning stops.", color=BLACK, font_size=26).next_to(insight_text_1, DOWN, buff=0.2)
        insight_text_3 = Text("Solution: Use ReLU for hidden layers!", color=BLUE, font_size=28).next_to(insight_text_2, DOWN, buff=0.5)

        with self.voiceover(text="Finally, the most important part: what will you actually be asked in an AI engineering interview? The most common question regarding activation functions revolves around their trade-offs, specifically the Vanishing Gradient Problem.") as tracker:
            self.play(Write(section5_title))
            self.play(Create(box))
            self.play(FadeIn(insight_title))
            self.wait(1.5)

        with self.voiceover(text="Interviewers want you to know that while Sigmoid and Tanh look nice, their gradients flatten out and approach zero for very high or very low input values. During backpropagation, we multiply these gradients together.") as tracker:
            self.play(Write(insight_text_1))
            self.wait(1.5)

        with self.voiceover(text="If you multiply many tiny numbers close to zero, the final gradient becomes effectively zero. When the gradient is zero, the neural network's weights do not update. The network completely stops learning. This is fatal for deep networks.") as tracker:
            self.play(Write(insight_text_2))
            self.wait(1.5)

        with self.voiceover(text="This is exactly why modern deep learning architectures almost exclusively use ReLU for hidden layers. The derivative of ReLU is a constant one for positive inputs, which allows gradients to flow backwards through the network without vanishing. Knowing this trade-off is essential for any AI engineer.") as tracker:
            self.play(Write(insight_text_3))
            self.wait(2)

        with self.voiceover(text="Let's summarize.") as tracker:
            self.play(
                FadeOut(VGroup(section5_title, box, insight_title, insight_text_1, insight_text_2, insight_text_3))
            )

        # Conclusion
        outro_title = Text("Summary", color=BLACK).scale(1.2).to_edge(UP)
        s1 = Text("1. Activation functions add non-linearity.", color=BLACK, font_size=32).shift(UP * 1)
        s2 = Text("2. ReLU is the standard for hidden layers.", color=BLUE, font_size=32).next_to(s1, DOWN, buff=0.5)
        s3 = Text("3. Sigmoid/Tanh suffer from vanishing gradients.", color=RED, font_size=32).next_to(s2, DOWN, buff=0.5)

        with self.voiceover(text="To summarize our deep dive today: First, activation functions are essential because they introduce non-linearity, allowing networks to learn complex patterns.") as tracker:
            self.play(Write(outro_title))
            self.play(FadeIn(s1))
            self.wait(1)

        with self.voiceover(text="Second, ReLU and its variants are the default standard for hidden layers in modern architectures due to their speed and robust gradient flow.") as tracker:
            self.play(FadeIn(s2))
            self.wait(1)

        with self.voiceover(text="And third, always remember that Sigmoid and Tanh are prone to the vanishing gradient problem, making them unsuitable for deep networks, though they are still useful in specific output layers. Thank you for joining Day 31 of AI Engineering Mastery. Happy coding!") as tracker:
            self.play(FadeIn(s3))
            self.wait(2)

        with self.voiceover(text="Goodbye.") as tracker:
            self.play(FadeOut(VGroup(outro_title, s1, s2, s3)))
