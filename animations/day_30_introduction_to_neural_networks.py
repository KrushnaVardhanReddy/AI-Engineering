from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class IntroductionToNeuralNetworks(VoiceoverScene):
    def construct(self):
        # Setup speech service
        self.set_speech_service(GTTSService())

        # Set aesthetic
        self.camera.background_color = WHITE

        # --- Section 1: What is it? ---
        title = Text("Neural Networks", font_size=60, color=BLACK).to_edge(UP)

        with self.voiceover(text="Welcome to Day 30! Today we are discussing Introduction to Neural Networks.") as tracker:
            self.play(Write(title))
            self.wait(1.5)

        def_text = Text(
            "A computing system inspired by the human brain that learns\npatterns from data to make predictions or decisions.",
            font_size=32, color=BLACK, t2c={"human brain": BLUE, "learns": GREEN, "patterns": GREEN}
        ).next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="What is a neural network? It is a computing system inspired by the human brain that learns patterns from data to make predictions or decisions. Let's see this in action.") as tracker:
            self.play(FadeIn(def_text, shift=DOWN))
            self.wait(1.5)

        # Draw a simple neural network
        layers = [3, 4, 2]
        neurons = VGroup()
        edges = VGroup()

        layer_spacing = 2.5
        neuron_spacing = 1.0

        for i, num_neurons in enumerate(layers):
            layer_neurons = VGroup()
            for j in range(num_neurons):
                neuron = Circle(radius=0.3, color=BLUE, fill_opacity=0.2, stroke_color=BLUE)
                neuron.move_to(
                    RIGHT * (i - len(layers)/2 + 0.5) * layer_spacing +
                    UP * (j - num_neurons/2 + 0.5) * neuron_spacing
                )
                layer_neurons.add(neuron)
            neurons.add(layer_neurons)

        for i in range(len(layers) - 1):
            for n1 in neurons[i]:
                for n2 in neurons[i+1]:
                    edge = Line(n1.get_center(), n2.get_center(), color=BLACK, stroke_width=1.5, stroke_opacity=0.5)
                    edges.add(edge)

        network_diagram = VGroup(edges, neurons).next_to(def_text, DOWN, buff=1.0)

        with self.voiceover(text="A neural network consists of interconnected nodes called artificial neurons, arranged in layers. We have an input layer that receives data, hidden layers that process it, and an output layer that produces the final prediction.") as tracker:
            self.play(Create(neurons), run_time=2)
            self.play(Create(edges), run_time=2)
            self.wait(1.5)

        # Highlight layers
        input_label = Text("Input Layer", font_size=24, color=BLUE).next_to(neurons[0], DOWN)
        hidden_label = Text("Hidden Layer", font_size=24, color=PURPLE).next_to(neurons[1], DOWN)
        output_label = Text("Output Layer", font_size=24, color=GREEN).next_to(neurons[2], DOWN)

        with self.voiceover(text="Information flows from the input layer on the left, through the hidden layers in the middle where the actual computation happens, and finally to the output layer on the right. During training, the network adjusts the strength of the connections between these neurons to minimize errors in its predictions.") as tracker:
            self.play(Write(input_label))
            self.play(neurons[0].animate.set_color(BLUE))
            self.wait(0.5)
            self.play(Write(hidden_label))
            self.play(neurons[1].animate.set_color(PURPLE))
            self.wait(0.5)
            self.play(Write(output_label))
            self.play(neurons[2].animate.set_color(GREEN))
            self.wait(1.5)

        with self.voiceover(text="Now that we know what a neural network is structurally, let's clear the board and understand why we actually need them in the first place.") as tracker:
            self.play(
                FadeOut(def_text),
                FadeOut(network_diagram),
                FadeOut(input_label),
                FadeOut(hidden_label),
                FadeOut(output_label)
            )
            self.wait(1.5)

        # --- Section 2: Why do we need it? ---
        why_title = Text("Why do we need it?", font_size=40, color=BLUE).to_edge(UP).shift(DOWN*0.5)

        with self.voiceover(text="Why do we need Neural Networks when we already have traditional programming and simpler machine learning models?") as tracker:
            self.play(Transform(title, why_title))
            self.wait(1.5)

        # Before (Traditional ML / Linear) vs After (Neural Networks / Non-linear)
        axes_before = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=4, y_length=4,
            axis_config={"color": BLACK}
        ).shift(LEFT * 3 + DOWN * 0.5)

        axes_after = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=4, y_length=4,
            axis_config={"color": BLACK}
        ).shift(RIGHT * 3 + DOWN * 0.5)

        label_before = Text("Traditional Models", font_size=28, color=BLACK).next_to(axes_before, UP)
        label_after = Text("Neural Networks", font_size=28, color=BLACK).next_to(axes_after, UP)

        # Data points for a non-linear problem (e.g. XOR or concentric circles)
        # Inner circle (class 0, RED)
        inner_points = [
            Circle(radius=0.08, color=RED, fill_opacity=1).move_to(axes_before.c2p(x, y))
            for x, y in [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (0, 1), (1, 0), (0, -1), (-1, 0)]
        ]
        # Outer circle (class 1, BLUE)
        outer_points = [
            Circle(radius=0.08, color=BLUE, fill_opacity=1).move_to(axes_before.c2p(x, y))
            for x, y in [(2, 2), (-2, 2), (-2, -2), (2, -2), (0, 2.5), (2.5, 0), (0, -2.5), (-2.5, 0)]
        ]

        inner_points_after = [p.copy().move_to(axes_after.c2p(axes_before.p2c(p.get_center())[0], axes_before.p2c(p.get_center())[1])) for p in inner_points]
        outer_points_after = [p.copy().move_to(axes_after.c2p(axes_before.p2c(p.get_center())[0], axes_before.p2c(p.get_center())[1])) for p in outer_points]

        with self.voiceover(text="Consider a complex, non-linear classification problem. Here, we have red points clustered in the center, surrounded by blue points. The relationship between the features is complex.") as tracker:
            self.play(FadeIn(axes_before), Write(label_before))
            self.play(FadeIn(VGroup(*inner_points, *outer_points)))
            self.wait(1.5)

        linear_line = Line(axes_before.c2p(-3, -2), axes_before.c2p(3, 2), color=GREEN, stroke_width=4)

        with self.voiceover(text="Traditional linear models, like logistic regression, try to separate the data with a straight line. As you can see, no straight line can accurately separate the red points from the blue points. They fail on highly non-linear data.") as tracker:
            self.play(Create(linear_line))
            self.wait(1.5)

        with self.voiceover(text="Now let's look at neural networks. We'll plot the exact same dataset.") as tracker:
            self.play(FadeIn(axes_after), Write(label_after))
            self.play(FadeIn(VGroup(*inner_points_after, *outer_points_after)))
            self.wait(1.5)

        circle_boundary = Circle(radius=1.5, color=PURPLE, stroke_width=4).move_to(axes_after.c2p(0, 0))

        with self.voiceover(text="Because neural networks use multiple layers of neurons with non-linear activation functions, they can learn highly complex, non-linear decision boundaries. The network easily learns to draw a circle around the inner points, solving the problem perfectly.") as tracker:
            self.play(Create(circle_boundary))
            self.wait(1.5)

        with self.voiceover(text="This ability to approximate almost any function, known as the Universal Approximation Theorem, is why neural networks are the foundation of modern deep learning. Let's move on to see where this is applied in the real world.") as tracker:
            self.play(
                FadeOut(axes_before), FadeOut(label_before), FadeOut(VGroup(*inner_points, *outer_points)), FadeOut(linear_line),
                FadeOut(axes_after), FadeOut(label_after), FadeOut(VGroup(*inner_points_after, *outer_points_after)), FadeOut(circle_boundary)
            )
            self.wait(1.5)

        # --- Section 3: Use Cases ---
        uc_title = Text("Use Cases", font_size=40, color=BLUE).to_edge(UP).shift(DOWN*0.5)

        with self.voiceover(text="Neural networks are everywhere today. Let's look at two prominent examples.") as tracker:
            self.play(Transform(title, uc_title))
            self.wait(1.5)

        uc1 = Text("1. ChatGPT (OpenAI)", font_size=32, color=BLACK, weight=BOLD).shift(UP*1.5 + LEFT*3)
        uc1_desc = Text("Uses Large Language Models (Transformer Neural\nNetworks) to understand context and generate\nhuman-like text responses.", font_size=24, color=BLACK).next_to(uc1, DOWN, aligned_edge=LEFT)

        with self.voiceover(text="First, ChatGPT by OpenAI. It is powered by a type of neural network called a Transformer. It processes massive amounts of text data to understand context, syntax, and semantics, allowing it to generate incredibly human-like responses to our prompts.") as tracker:
            self.play(Write(uc1))
            self.play(FadeIn(uc1_desc))
            self.wait(1.5)

        uc2 = Text("2. Spotify Recommendations", font_size=32, color=BLACK, weight=BOLD).next_to(uc1_desc, DOWN, buff=1.0).align_to(uc1, LEFT)
        uc2_desc = Text("Uses Deep Neural Networks to analyze listening\nhistory, audio features, and user behavior to predict\nand recommend songs you will like.", font_size=24, color=BLACK).next_to(uc2, DOWN, aligned_edge=LEFT)

        with self.voiceover(text="Second, consider Spotify. They use deep neural networks for their recommendation engine. By analyzing your listening history alongside the audio features of millions of songs, the network learns complex patterns in your taste and accurately predicts which new songs you will enjoy.") as tracker:
            self.play(Write(uc2))
            self.play(FadeIn(uc2_desc))
            self.wait(1.5)

        with self.voiceover(text="From generating text and images to driving autonomous vehicles, neural networks excel whenever there is a massive amount of complex data to learn from. But they aren't perfect.") as tracker:
            self.play(FadeOut(uc1), FadeOut(uc1_desc), FadeOut(uc2), FadeOut(uc2_desc))
            self.wait(1.5)

        # --- Section 4: Key Interview Insight ---
        insight_title = Text("Key Interview Insight", font_size=40, color=BLUE).to_edge(UP).shift(DOWN*0.5)

        with self.voiceover(text="If you are asked about Neural Networks in a machine learning interview, there is a very common tradeoff you must be prepared to discuss.") as tracker:
            self.play(Transform(title, insight_title))
            self.wait(1.5)

        insight_box = Rectangle(width=10, height=4, color=RED, stroke_width=3, fill_color=WHITE, fill_opacity=1)
        insight_label = Text("Interpretability vs. Performance", font_size=36, color=RED, weight=BOLD).next_to(insight_box.get_top(), DOWN, buff=0.3)

        insight_content = Text(
            "Neural Networks are often \"Black Boxes\".\n\n"
            "• High Performance: Great at complex patterns.\n"
            "• Low Interpretability: Hard to explain *why* they\n  made a specific decision.",
            font_size=28, color=BLACK, t2c={"Black Boxes": RED, "High Performance": GREEN, "Low Interpretability": RED}
        ).next_to(insight_label, DOWN, buff=0.5)

        with self.voiceover(text="The most critical concept to mention is the tradeoff between Interpretability and Performance. While neural networks achieve state-of-the-art performance on complex tasks like image and text recognition, they are notoriously known as 'Black Boxes'.") as tracker:
            self.play(Create(insight_box))
            self.play(Write(insight_label))
            self.wait(1.5)

        with self.voiceover(text="In a traditional model like a decision tree, you can easily trace the exact path taken to arrive at a prediction. In a deep neural network with millions of parameters, it is incredibly difficult to explain exactly *why* the model made a specific decision. If you are building a medical diagnosis system or a loan approval model, this lack of interpretability can be a major regulatory and ethical problem.") as tracker:
            self.play(FadeIn(insight_content))
            self.wait(1.5)

        # Show a math derivation of a neuron to show the "black box" complexity conceptually
        math_title = Text("Inside a single neuron:", font_size=24, color=BLACK).next_to(insight_box, DOWN, buff=0.5).align_to(insight_box, LEFT)

        # Step-by-step MathTex
        eq1 = MathTex("y", "=", "f(", "w_1 x_1 + w_2 x_2 + b", ")", color=BLACK).next_to(math_title, RIGHT, buff=0.5)
        eq2 = MathTex("y", "=", "f(", "\\sum_{i=1}^{n} w_i x_i + b", ")", color=BLACK).move_to(eq1)

        with self.voiceover(text="Even a single neuron performs a weighted sum of inputs plus a bias, passed through an activation function. When you stack thousands or millions of these neurons together, the function becomes so complex that humans can no longer intuitively parse the logic. Mentioning this tradeoff shows you understand practical AI engineering, not just theory.") as tracker:
            self.play(Write(math_title))
            self.play(Write(eq1))
            self.wait(1.5)
            self.play(TransformMatchingTex(eq1, eq2))
            self.wait(1.5)

        with self.voiceover(text="That concludes our introduction to Neural Networks for Day 30. We've covered what they are, why their non-linear capabilities are essential, real world use cases, and the critical black box tradeoff. Keep practicing, and I'll see you in the next lesson!") as tracker:
            self.play(FadeOut(VGroup(title, insight_box, insight_label, insight_content, math_title, eq2)))

            final_text = Text("Day 30 Complete!", font_size=60, color=BLUE)
            self.play(Write(final_text))
            self.wait(2)
            self.play(FadeOut(final_text))
