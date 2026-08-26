from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class BackpropChainRule(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Section 1: What is it?
        self.section_what_is_it()

        # Section 2: Why do we need it?
        self.section_why_do_we_need_it()

        # Section 3: Use Cases
        self.section_use_cases()

        # Section 4: Key Interview Insight
        self.section_key_insight()

    def section_what_is_it(self):
        title = Text("Backpropagation & The Chain Rule", color=BLACK, font_size=48, weight=BOLD)
        title.to_edge(UP)

        with self.voiceover(text="Welcome to our deep dive into the core engine of deep learning: Backpropagation and the Chain Rule.") as tracker:
            self.play(Write(title))
            self.wait(1.5)

        def_text = Text(
            "What is it?\n"
            "A method to calculate how much each weight\n"
            "in a neural network contributed to the final error.",
            color=BLACK, font_size=32, line_spacing=1.2
        ).next_to(title, DOWN, buff=1.0)

        with self.voiceover(text="What exactly is backpropagation? In a single sentence, it is a method to calculate exactly how much each weight in a neural network contributed to the final error. It tells the network how to adjust its internal parameters to make better predictions.") as tracker:
            self.play(FadeIn(def_text))
            self.wait(1.5)

        # Simple Computational Graph
        x_node = Circle(radius=0.5, color=BLUE).set_fill(BLUE, opacity=0.2).move_to(LEFT * 4 + DOWN * 1)
        x_label = MathTex("x", color=BLACK).move_to(x_node)

        w_node = Circle(radius=0.5, color=BLUE).set_fill(BLUE, opacity=0.2).move_to(LEFT * 4 + DOWN * 3)
        w_label = MathTex("w", color=BLACK).move_to(w_node)

        z_node = Circle(radius=0.6, color=GREEN).set_fill(GREEN, opacity=0.2).move_to(LEFT * 0 + DOWN * 2)
        z_label = MathTex("z = w \\cdot x", color=BLACK).move_to(z_node)

        L_node = Circle(radius=0.6, color=RED).set_fill(RED, opacity=0.2).move_to(RIGHT * 4 + DOWN * 2)
        L_label = MathTex("L(z)", color=BLACK).move_to(L_node)

        arrow_xw_z = VGroup(
            Arrow(x_node.get_right(), z_node.get_left(), color=BLACK),
            Arrow(w_node.get_right(), z_node.get_left(), color=BLACK)
        )
        arrow_z_L = Arrow(z_node.get_right(), L_node.get_left(), color=BLACK)

        graph = VGroup(x_node, x_label, w_node, w_label, z_node, z_label, L_node, L_label, arrow_xw_z, arrow_z_L)

        with self.voiceover(text="Let's look at a very simple neural network layer, represented as a computational graph. We have an input x and a weight w. They are multiplied to create a hidden state z. Finally, z is passed into a loss function L to measure our error.") as tracker:
            self.play(FadeOut(def_text))
            self.play(FadeIn(x_node, x_label, w_node, w_label))
            self.play(Write(arrow_xw_z), FadeIn(z_node, z_label))
            self.play(Write(arrow_z_L), FadeIn(L_node, L_label))
            self.wait(1.5)

        chain_rule_title = Text("The Chain Rule of Calculus", color=BLACK, font_size=36, weight=BOLD).next_to(title, DOWN, buff=0.5)

        eq1 = MathTex(r"\frac{\partial L}{\partial w}", r"=", r"?", color=BLACK).move_to(UP * 1)
        eq2 = MathTex(r"\frac{\partial L}{\partial w}", r"=", r"\frac{\partial L}{\partial z}", r"\cdot", r"\frac{\partial z}{\partial w}", color=BLACK).move_to(UP * 1)

        with self.voiceover(text="Our goal is to find out how a tiny change in the weight w affects the final loss L. In calculus notation, we want to compute the partial derivative of L with respect to w.") as tracker:
            self.play(Write(chain_rule_title))
            self.play(Write(eq1))
            self.wait(1.5)

        with self.voiceover(text="Because w only affects L indirectly through z, we must use the Chain Rule. The Chain Rule states that the derivative of a composite function is the product of the derivatives of its parts. So, the effect of w on L is the effect of z on L multiplied by the effect of w on z.") as tracker:
            self.play(TransformMatchingTex(eq1, eq2))
            self.wait(1.5)

        with self.voiceover(text="Backpropagation is simply applying this chain rule repeatedly, backwards through the graph, from the output all the way back to the earliest weights.") as tracker:
            # Animate the reverse flow
            flow1 = Arrow(L_node.get_top(), z_node.get_top(), color=PURPLE, path_arc=0.5)
            flow1_label = MathTex(r"\frac{\partial L}{\partial z}", color=PURPLE).next_to(flow1, UP, buff=0.1)

            flow2 = Arrow(z_node.get_bottom(), w_node.get_right(), color=PURPLE, path_arc=0.5)
            flow2_label = MathTex(r"\frac{\partial z}{\partial w}", color=PURPLE).next_to(flow2, DOWN, buff=0.1)

            self.play(Write(flow1), FadeIn(flow1_label))
            self.wait(1.0)
            self.play(Write(flow2), FadeIn(flow2_label))
            self.wait(1.5)

        self.play(FadeOut(VGroup(title, chain_rule_title, eq2, graph, flow1, flow1_label, flow2, flow2_label)))


    def section_why_do_we_need_it(self):
        title = Text("Why do we need it?", color=BLACK, font_size=48, weight=BOLD).to_edge(UP)

        with self.voiceover(text="So, why do we need backpropagation? Why can't we just figure out the weights without it?") as tracker:
            self.play(Write(title))
            self.wait(1.5)

        # Before (Without Backprop)
        subtitle_without = Text("Without Backprop: Random Guessing", color=RED, font_size=36).next_to(title, DOWN, buff=0.5)

        box = Rectangle(width=4, height=3, color=BLACK).move_to(LEFT * 3)
        box_text = Text("Neural Network\n(1 Million Weights)", color=BLACK, font_size=24).move_to(box)

        guess_text = Text("Guess w1 = 0.5?\nGuess w2 = -0.1?\n...", color=BLACK, font_size=24).next_to(box, DOWN)

        with self.voiceover(text="Imagine a modern neural network with millions or billions of weights. If we didn't have a structured way to calculate gradients, we would essentially be guessing. We might try tweaking a weight randomly, running the entire network to see if the loss goes down, and repeating.") as tracker:
            self.play(Write(subtitle_without))
            self.play(FadeIn(box, box_text))
            self.play(Write(guess_text))
            self.wait(1.5)

        with self.voiceover(text="This approach is computationally impossible. For a million weights, you would have to run the network a million times just to figure out one update step.") as tracker:
            cross = Cross(VGroup(box, box_text, guess_text), stroke_color=RED, stroke_width=8)
            self.play(Write(cross))
            self.wait(1.5)

        self.play(FadeOut(VGroup(subtitle_without, box, box_text, guess_text, cross)))

        # After (With Backprop)
        subtitle_with = Text("With Backprop: Directed Learning", color=GREEN, font_size=36).next_to(title, DOWN, buff=0.5)

        graph_box = Rectangle(width=6, height=3, color=BLACK).move_to(DOWN * 1)
        graph_layers = VGroup(
            Circle(radius=0.2, color=BLUE, fill_opacity=1).move_to(graph_box.get_left() + RIGHT * 1 + UP * 0.5),
            Circle(radius=0.2, color=BLUE, fill_opacity=1).move_to(graph_box.get_left() + RIGHT * 1 + DOWN * 0.5),

            Circle(radius=0.2, color=GREEN, fill_opacity=1).move_to(graph_box.get_center() + UP * 0.5),
            Circle(radius=0.2, color=GREEN, fill_opacity=1).move_to(graph_box.get_center() + DOWN * 0.5),

            Circle(radius=0.2, color=RED, fill_opacity=1).move_to(graph_box.get_right() + LEFT * 1)
        )

        lines = VGroup(
            Line(graph_layers[0].get_right(), graph_layers[2].get_left(), color=BLACK, stroke_width=2),
            Line(graph_layers[0].get_right(), graph_layers[3].get_left(), color=BLACK, stroke_width=2),
            Line(graph_layers[1].get_right(), graph_layers[2].get_left(), color=BLACK, stroke_width=2),
            Line(graph_layers[1].get_right(), graph_layers[3].get_left(), color=BLACK, stroke_width=2),

            Line(graph_layers[2].get_right(), graph_layers[4].get_left(), color=BLACK, stroke_width=2),
            Line(graph_layers[3].get_right(), graph_layers[4].get_left(), color=BLACK, stroke_width=2)
        )

        with self.voiceover(text="With backpropagation, we solve this elegantly. We do one forward pass to calculate the loss, and then one backward pass to compute the exact gradient for every single weight in the network simultaneously.") as tracker:
            self.play(Write(subtitle_with))
            self.play(FadeIn(graph_box, graph_layers, lines))

            # Forward pass animation
            forward_runner = Dot(color=YELLOW).move_to(graph_layers[0].get_center())
            self.play(MoveAlongPath(forward_runner, Line(graph_layers[0].get_center(), graph_layers[2].get_center())), run_time=0.5)
            self.play(MoveAlongPath(forward_runner, Line(graph_layers[2].get_center(), graph_layers[4].get_center())), run_time=0.5)
            self.play(FadeOut(forward_runner))

            # Backward pass animation
            backward_runners = VGroup(
                Dot(color=PURPLE).move_to(graph_layers[4].get_center()),
                Dot(color=PURPLE).move_to(graph_layers[4].get_center())
            )
            self.play(
                MoveAlongPath(backward_runners[0], Line(graph_layers[4].get_center(), graph_layers[2].get_center())),
                MoveAlongPath(backward_runners[1], Line(graph_layers[4].get_center(), graph_layers[3].get_center())),
                run_time=0.5
            )
            self.play(
                MoveAlongPath(backward_runners[0], Line(graph_layers[2].get_center(), graph_layers[0].get_center())),
                MoveAlongPath(backward_runners[1], Line(graph_layers[3].get_center(), graph_layers[1].get_center())),
                run_time=0.5
            )
            self.play(FadeOut(backward_runners))

            self.wait(1.5)

        with self.voiceover(text="This mathematical efficiency is the only reason deep learning is possible today. It scales perfectly to billions of parameters.") as tracker:
            eff_text = Text("1 Forward Pass + 1 Backward Pass = All Gradients", color=PURPLE, font_size=28).next_to(graph_box, DOWN, buff=0.5)
            self.play(Write(eff_text))
            self.wait(2.0)

        self.play(FadeOut(VGroup(title, subtitle_with, graph_box, graph_layers, lines, eff_text)))


    def section_use_cases(self):
        title = Text("Real-World Use Cases", color=BLACK, font_size=48, weight=BOLD).to_edge(UP)

        with self.voiceover(text="Where is backpropagation used in the real world? The short answer is: everywhere deep learning is used.") as tracker:
            self.play(Write(title))
            self.wait(1.0)

        # Use Case 1
        chatgpt_logo = Text("ChatGPT (OpenAI)", color=BLUE, font_size=36, weight=BOLD).move_to(UP * 1 + LEFT * 3)
        chatgpt_desc = Text(
            "Training Large Language Models.\n"
            "Backprop updates billions of weights\n"
            "to predict the next word accurately.",
            color=BLACK, font_size=24, line_spacing=1.2
        ).next_to(chatgpt_logo, DOWN, buff=0.5)

        with self.voiceover(text="For instance, OpenAI uses backpropagation to train models like ChatGPT. During training, the model tries to predict the next word. If it gets it wrong, backpropagation uses the chain rule to adjust billions of weights across dozens of transformer layers so it does better next time.") as tracker:
            self.play(FadeIn(chatgpt_logo))
            self.play(Write(chatgpt_desc))
            self.wait(1.5)

        # Use Case 2
        spotify_logo = Text("Spotify (Recommender Systems)", color=GREEN, font_size=36, weight=BOLD).move_to(UP * 1 + RIGHT * 3)
        spotify_desc = Text(
            "Training embedding models.\n"
            "Backprop tunes neural networks\n"
            "to map users and songs to vectors.",
            color=BLACK, font_size=24, line_spacing=1.2
        ).next_to(spotify_logo, DOWN, buff=0.5)

        with self.voiceover(text="Similarly, Spotify uses backpropagation to train their music recommendation algorithms. They use deep neural networks to create embeddings—mathematical representations of users and songs. Backprop adjusts the network so that songs you like end up close to your user profile in vector space.") as tracker:
            self.play(FadeIn(spotify_logo))
            self.play(Write(spotify_desc))
            self.wait(2.0)

        self.play(FadeOut(VGroup(title, chatgpt_logo, chatgpt_desc, spotify_logo, spotify_desc)))


    def section_key_insight(self):
        title = Text("Key Interview Insight", color=BLACK, font_size=48, weight=BOLD).to_edge(UP)

        with self.voiceover(text="Finally, let's cover the most important insight you need for AI engineering interviews regarding backpropagation.") as tracker:
            self.play(Write(title))
            self.wait(1.0)

        box = Rectangle(width=10, height=5, color=RED, fill_opacity=0.05).move_to(DOWN * 0.5)
        insight_title = Text("The Vanishing Gradient Problem", color=RED, font_size=36, weight=BOLD).move_to(box.get_top() + DOWN * 0.7)

        eq = MathTex(
            r"\frac{\partial L}{\partial w_1}", r"=",
            r"0.1 \times 0.2 \times 0.05 \times 0.1 \times \frac{\partial L}{\partial y}",
            color=BLACK
        ).next_to(insight_title, DOWN, buff=0.8)

        result_eq = MathTex(
            r"\frac{\partial L}{\partial w_1}", r"\approx", r"0.0001",
            color=RED
        ).next_to(insight_title, DOWN, buff=0.8)

        with self.voiceover(text="Interviewers will often ask you about the Vanishing Gradient Problem. Because backpropagation relies heavily on multiplying chain rule terms together, what happens when those terms are very small?") as tracker:
            self.play(FadeIn(box))
            self.play(Write(insight_title))
            self.play(Write(eq))
            self.wait(1.5)

        with self.voiceover(text="If you multiply many small numbers together—for example, gradients passing through a long chain of Sigmoid or Tanh activation functions—the final gradient becomes exponentially small.") as tracker:
            self.play(TransformMatchingTex(eq, result_eq))
            self.wait(1.5)

        solution_text = Text(
            "Solution: Use ReLU activations or Residual Connections (ResNets)",
            color=BLACK, font_size=28, weight=BOLD
        ).next_to(result_eq, DOWN, buff=1.0)

        with self.voiceover(text="By the time the gradient reaches the earliest layers of a deep network, it is effectively zero, meaning those layers stop learning entirely. The standard solution, which you must mention in an interview, is to use ReLU activation functions, which pass gradients of exactly 1.0, or to use Residual Connections like in ResNets, which provide a shortcut for gradients to bypass the multiplication chain.") as tracker:
            self.play(Write(solution_text))
            self.wait(3.0)

        with self.voiceover(text="Understanding not just how backpropagation works, but how its multiplication chains can fail, is the hallmark of a senior AI engineer. Thank you for watching.") as tracker:
            self.wait(2.0)

        self.play(FadeOut(VGroup(title, box, insight_title, result_eq, solution_text)))
        self.wait(1.0)
