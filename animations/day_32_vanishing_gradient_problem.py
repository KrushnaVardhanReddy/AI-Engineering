import numpy as np
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class VanishingGradientProblem(VoiceoverScene):
    def construct(self):
        # Setup aesthetic
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Overall title
        title = Text("The Vanishing Gradient Problem", font_size=50, color=BLACK)
        title.to_edge(UP)

        # ---------------------------------------------------------
        # SECTION 1: WHAT IS IT?
        # ---------------------------------------------------------
        with self.voiceover(text="Welcome back to our comprehensive AI and machine learning interview preparation series. Today, we are going to dive deep into a foundational concept that every machine learning engineer must thoroughly understand: the Vanishing Gradient Problem. This problem has historically been one of the biggest roadblocks to training deep neural networks. So, what exactly is the vanishing gradient problem, and why has it caused so much trouble for researchers over the years?") as tracker:
            self.play(Write(title))
            self.wait(1.5)

        def_text = Text(
            "During backpropagation, gradients get smaller and\nsmaller as they propagate backward to earlier layers.",
            font_size=30, color=BLACK, t2c={"smaller": RED, "gradients": BLUE}
        )
        def_text.next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="In simple terms, the vanishing gradient problem occurs during the training phase of artificial neural networks, specifically when we are using the backpropagation algorithm. As the network attempts to learn, it calculates the error at the output layer and propagates this error backward through the network to update the weights. However, as these error gradients are passed backward from layer to layer, they can become increasingly smaller, eventually shrinking down to practically zero. This effectively prevents the weights of those earlier layers from updating at all, meaning the first few layers of your deep network simply stop learning, freezing the network in an untrained state.") as tracker:
            self.play(FadeIn(def_text))
            self.wait(1.5)

        # Visualizing a neural network
        layers = VGroup()
        layer_sizes = [4, 4, 4, 4, 1]

        for i, size in enumerate(layer_sizes):
            layer = VGroup(*[Circle(radius=0.2, color=BLACK, fill_opacity=0.1) for _ in range(size)])
            layer.arrange(DOWN, buff=0.3)
            layers.add(layer)
        layers.arrange(RIGHT, buff=1.5)
        layers.next_to(def_text, DOWN, buff=1.0)

        connections = VGroup()
        for i in range(len(layers) - 1):
            for n1 in layers[i]:
                for n2 in layers[i+1]:
                    connections.add(Line(n1.get_right(), n2.get_left(), stroke_width=1, color=BLACK))

        with self.voiceover(text="Let us visualize this by looking at a standard deep neural network architecture. On the far left side, we have our input layer where the initial data enters the network. In the middle, we have several hidden layers responsible for extracting complex features. And finally, on the far right, we have our output layer which provides the final prediction. In a deep network, you might have dozens or even hundreds of these hidden layers stacked sequentially.") as tracker:
            self.play(FadeIn(connections))
            self.play(FadeIn(layers))
            self.wait(1.5)

        # Backpropagation animation
        arrows = VGroup()
        for i in range(len(layers) - 1, 0, -1):
            arrow = Arrow(
                start=layers[i].get_left() + LEFT * 0.2,
                end=layers[i-1].get_right() + RIGHT * 0.2,
                color=RED, stroke_width=4
            )
            arrows.add(arrow)

        with self.voiceover(text="When the network makes a prediction, we compare it against the actual ground truth to calculate the total loss at the output layer. To minimize this loss, we use the chain rule from calculus to backpropagate the error backwards through the network. At each layer, we calculate the partial derivative of the loss with respect to the weights, and we adjust those weights accordingly using an optimizer like gradient descent.") as tracker:
            self.play(GrowArrow(arrows[0]))
            self.wait(1.5)

        with self.voiceover(text="However, here is where the critical issue arises. Traditional activation functions, like the Sigmoid or Tanh functions, map large input spaces into a very small, constrained range. Because of this squashing effect, their derivatives are always significantly less than one. As we backpropagate through a deep network, the chain rule requires us to multiply these small derivative fractions together repeatedly for every single layer. When you multiply a small fraction by another small fraction many times over, the resulting gradient signal shrinks exponentially, causing it to completely vanish before it ever reaches the early layers.") as tracker:
            for i in range(1, len(arrows)):
                arrows[i].set_opacity(1.0 - (i * 0.25)) # Gradient gets smaller/fainter
                self.play(GrowArrow(arrows[i]))
            self.wait(1.5)

        self.play(FadeOut(VGroup(def_text, layers, connections, arrows)))
        self.wait(1.0)

        # ---------------------------------------------------------
        # SECTION 2: WHY DO WE NEED TO SOLVE IT? (BEFORE vs AFTER)
        # ---------------------------------------------------------
        why_title = Text("Why do we need it? (The Solution)", font_size=40, color=BLUE)
        why_title.next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="Why is this such a big deal in the grand scheme of artificial intelligence? Why do we absolutely need reliable solutions to overcome the vanishing gradient problem? The answer lies in the very nature of deep learning. Without a solution, we simply cannot train deep networks. If the early layers fail to learn the basic, foundational features of the input data, the later layers will have nothing meaningful to build upon, completely destroying the performance of the entire model.") as tracker:
            self.play(Write(why_title))
            self.wait(1.5)

        before_title = Text("Before (Sigmoid)", font_size=30, color=BLACK)
        before_title.move_to(LEFT * 3.5 + UP * 1)

        after_title = Text("After (ReLU)", font_size=30, color=BLACK)
        after_title.move_to(RIGHT * 3.5 + UP * 1)

        axes_before = Axes(
            x_range=[-5, 5, 1], y_range=[-0.2, 1.2, 0.5],
            x_length=4, y_length=3, axis_config={"color": BLACK}
        ).next_to(before_title, DOWN)

        axes_after = Axes(
            x_range=[-5, 5, 1], y_range=[-1, 5, 1],
            x_length=4, y_length=3, axis_config={"color": BLACK}
        ).next_to(after_title, DOWN)

        sigmoid_graph = axes_before.plot(lambda x: 1 / (1 + np.exp(-x)), color=RED)
        relu_graph = axes_after.plot(lambda x: max(0, x), color=GREEN)

        with self.voiceover(text="To truly understand the solution, let us look at the before and after states. Before the advent of modern deep learning architectures, researchers heavily relied on the Sigmoid activation function, shown here on the left. Notice how the classic S-shaped curve behaves: for any inputs that are very large, either positive or negative, the curve flattens out almost entirely into horizontal lines at the top and bottom.") as tracker:
            self.play(FadeIn(before_title), FadeIn(axes_before))
            self.play(Write(sigmoid_graph))
            self.wait(1.5)

        sig_deriv_graph = axes_before.plot(lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x)))), color=PURPLE)
        deriv_label = Text("Derivative (max 0.25)", font_size=20, color=PURPLE).next_to(sig_deriv_graph, UP)

        with self.voiceover(text="This extreme flatness in the activation function means that the derivative, or the slope of the sigmoid, is very small. In fact, the maximum possible derivative of a sigmoid function is only 0.25, and it quickly drops to zero for most input values. When we apply the chain rule and multiply dozens of these small derivatives together during backpropagation, the overall gradient shrinks exponentially towards absolute zero, causing the vanishing gradient problem we just discussed.") as tracker:
            self.play(Write(sig_deriv_graph), FadeIn(deriv_label))
            self.wait(1.5)

        with self.voiceover(text="To fundamentally solve this architectural flaw, modern deep learning introduced alternative activation functions, the most famous being the Rectified Linear Unit, commonly known as ReLU, shown here on the right. Notice how its shape differs entirely from the sigmoid curve. Instead of squashing values, it simply outputs zero for negative inputs, and outputs the exact original value for any positive inputs.") as tracker:
            self.play(FadeIn(after_title), FadeIn(axes_after))
            self.play(Write(relu_graph))
            self.wait(1.5)

        relu_deriv_graph = axes_after.plot(lambda x: 1 if x > 0 else 0, color=PURPLE)
        relu_deriv_label = Text("Derivative (1 for x>0)", font_size=20, color=PURPLE).next_to(relu_deriv_graph, UP)

        with self.voiceover(text="The true magic of ReLU lies in its derivative. For all positive input values, the slope of the line is a perfect, constant 1. This means the derivative of ReLU is exactly 1 for any positive input. When we backpropagate the error and multiply by 1 repeatedly, the gradient signal is perfectly preserved without shrinking or vanishing. This simple mathematical property is what finally allowed researchers to successfully train incredibly deep networks with hundreds of layers.") as tracker:
            self.play(Write(relu_deriv_graph), FadeIn(relu_deriv_label))
            self.wait(1.5)

        self.play(FadeOut(VGroup(why_title, before_title, after_title, axes_before, axes_after,
                                 sigmoid_graph, relu_graph, sig_deriv_graph, relu_deriv_graph,
                                 deriv_label, relu_deriv_label)))
        self.wait(1.0)

        # ---------------------------------------------------------
        # SECTION 3: USE CASES
        # ---------------------------------------------------------
        usecase_title = Text("Real-World Use Cases", font_size=40, color=BLUE)
        usecase_title.next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="Where exactly are solutions to the vanishing gradient problem applied in the real world? The short answer is: everywhere. Essentially every single modern deep learning system in production today uses specific architectural choices explicitly designed to combat vanishing gradients.") as tracker:
            self.play(Write(usecase_title))
            self.wait(1.5)

        uc1 = Text("1. OpenAI (ChatGPT / GPT-4)", font_size=30, color=BLACK, weight=BOLD)
        uc1_desc = Text("Uses residual connections (ResNets/Transformers) to\nbypass layers and skip gradient multiplication.", font_size=24, color=BLACK)
        uc1_group = VGroup(uc1, uc1_desc).arrange(DOWN, aligned_edge=LEFT)
        uc1_group.move_to(LEFT * 1 + UP * 0.5)

        uc2 = Text("2. Spotify (Audio Recommendations)", font_size=30, color=BLACK, weight=BOLD)
        uc2_desc = Text("Uses LSTMs or GRUs to process sequential audio data,\npreventing gradients from vanishing over long sequences.", font_size=24, color=BLACK)
        uc2_group = VGroup(uc2, uc2_desc).arrange(DOWN, aligned_edge=LEFT)
        uc2_group.next_to(uc1_group, DOWN, buff=1.0, aligned_edge=LEFT)

        with self.voiceover(text="For example, let us look at OpenAI and their massive large language models like ChatGPT or GPT-4. These systems are built on the Transformer architecture. A core component of Transformers are residual connections, or skip connections. These connections literally take the original input and add it directly back to the output of a block of layers, effectively creating a highway that bypasses the complex layers. This allows the gradient to flow completely unimpeded directly backward through the network, solving the vanishing gradient issue for these massive models.") as tracker:
            self.play(FadeIn(uc1))
            self.play(FadeIn(uc1_desc))
            self.wait(1.5)

        with self.voiceover(text="Additionally, consider companies like Spotify or Netflix that process massive amounts of sequential data like audio streams or user listening histories. When processing long sequences, standard Recurrent Neural Networks suffer terribly from vanishing gradients. To fix this, they use advanced architectures like LSTMs or GRUs. These networks introduce specialized mathematical memory gates, such as the forget gate and update gate, which are designed explicitly to trap and protect the error gradient, preventing it from vanishing over long sequences of time-series data.") as tracker:
            self.play(FadeIn(uc2))
            self.play(FadeIn(uc2_desc))
            self.wait(1.5)

        self.play(FadeOut(VGroup(usecase_title, uc1_group, uc2_group)))
        self.wait(1.0)

        # ---------------------------------------------------------
        # SECTION 4: KEY INTERVIEW INSIGHT
        # ---------------------------------------------------------
        insight_title = Text("Key Interview Insight", font_size=40, color=RED)
        insight_title.next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="Finally, let us distill all of this down into the most critical key interview insight you need to remember. When you are sitting in a machine learning interview, the interviewer will likely test if you understand the tradeoffs of the solutions you propose.") as tracker:
            self.play(Write(insight_title))
            self.wait(1.5)

        box = SurroundingRectangle(Text("                                                               \n\n\n\n\n", font_size=28), color=RED, buff=0.5)
        box.next_to(insight_title, DOWN, buff=0.5)

        insight_text_1 = Text("Interviewer Gotcha:", font_size=28, color=RED, weight=BOLD)
        insight_text_2 = Text("Is ReLU the perfect solution?", font_size=28, color=BLACK)

        insight_text_3 = Text("Answer: No. While it solves the vanishing gradient,\nit introduces the 'Dying ReLU' problem.", font_size=26, color=BLACK)

        insight_text_4 = Text("If inputs are negative, the gradient is 0.\nThe neuron dies and never updates again.", font_size=26, color=BLUE)

        insight_group = VGroup(insight_text_1, insight_text_2, insight_text_3, insight_text_4).arrange(DOWN, buff=0.4)
        insight_group.move_to(box.get_center())

        with self.voiceover(text="A very common gotcha question an interviewer might ask you is: 'Since ReLU solves the vanishing gradient problem by providing a constant derivative of 1, is ReLU the absolute perfect, flawless solution?' The answer you must provide is a definitive and confident No.") as tracker:
            self.play(Write(box))
            self.play(Write(insight_text_1), Write(insight_text_2))
            self.wait(1.5)

        with self.voiceover(text="While it is entirely true that ReLU successfully prevents gradients from vanishing for all positive input values, it simultaneously introduces a brand new, dangerous architectural flaw known in the industry as the 'Dying ReLU' problem.") as tracker:
            self.play(FadeIn(insight_text_3))
            self.wait(1.5)

        with self.voiceover(text="This occurs because the derivative of ReLU is exactly zero for all negative inputs. If a large gradient updates a neuron's weights so drastically that the neuron starts only outputting negative values for all possible inputs, its gradient during backpropagation becomes permanently zero. The neuron effectively 'dies'—it will never update its weights again and becomes completely useless to the network. To demonstrate deep expertise to your interviewer, you should immediately suggest fixing this with variants like Leaky ReLU or Parametric ReLU, which allow a tiny, non-zero gradient even for negative values.") as tracker:
            self.play(FadeIn(insight_text_4))
            self.wait(1.5)

        with self.voiceover(text="That concludes our comprehensive deep dive into the Vanishing Gradient problem, why it occurs, how modern architectures solve it, and the critical tradeoffs you need to watch out for. Make sure to review the differences between Sigmoid and ReLU derivatives. Best of luck on your upcoming AI engineering interviews, and keep building!") as tracker:
            self.play(FadeOut(VGroup(title, insight_title, box, insight_group)))
            self.wait(2.0)
