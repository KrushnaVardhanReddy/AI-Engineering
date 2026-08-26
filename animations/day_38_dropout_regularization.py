import numpy as np
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class DropoutRegularization(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # Section 1: What is it?
        title = Tex("Dropout Regularization", color=BLACK).scale(1.5).to_edge(UP)
        with self.voiceover(text="Welcome back to our AI Engineering interview prep series. Today, we are taking a deep dive into Dropout Regularization, one of the most fundamental and widely used techniques for training robust neural networks.") as tracker:
            self.play(Write(title))

        with self.voiceover(text="What exactly is dropout? Dropout is a powerful regularization technique used in deep learning models to prevent a common problem known as overfitting. It does this by randomly disabling, or 'dropping out', a fraction of the neurons in a layer during each forward pass of the training process.") as tracker:
            definition = Tex(
                r"\textbf{Definition: }",
                "Randomly disabling a fraction of neurons\\\\during training to prevent overfitting.",
                color=BLACK
            ).scale(0.8).next_to(title, DOWN, buff=0.5)
            self.play(Write(definition))

        self.wait(1.5)

        # Draw a simple neural net layer
        layer1_dots = VGroup(*[Dot(color=BLACK, radius=0.2) for _ in range(5)]).arrange(DOWN, buff=0.5).shift(LEFT * 3)
        layer2_dots = VGroup(*[Dot(color=BLACK, radius=0.2) for _ in range(5)]).arrange(DOWN, buff=0.5).shift(RIGHT * 3)

        connections = VGroup()
        for d1 in layer1_dots:
            for d2 in layer2_dots:
                connections.add(Line(d1.get_right(), d2.get_left(), color=BLACK, stroke_width=1, stroke_opacity=0.3))

        nn_group = VGroup(layer1_dots, layer2_dots, connections).shift(DOWN * 1)

        with self.voiceover(text="To visualize this, imagine a fully connected neural network where every single neuron in one layer connects to every single neuron in the next layer. During a standard training iteration without dropout, all these pathways are active, and information flows through the entire dense web of connections.") as tracker:
            self.play(FadeIn(nn_group))

        # Dropout animation
        dropped_indices_1 = [1, 3]
        dropped_indices_2 = [0, 4]

        animations = []
        for i in dropped_indices_1:
            animations.append(layer1_dots[i].animate.set_color(RED).set_opacity(0.3))
        for i in dropped_indices_2:
            animations.append(layer2_dots[i].animate.set_color(RED).set_opacity(0.3))

        lines_to_fade = VGroup()
        for line in connections:
            start_dot = line.get_start()
            end_dot = line.get_end()

            # Check if start or end is from a dropped dot
            is_dropped = False
            for i in dropped_indices_1:
                if np.allclose(start_dot, layer1_dots[i].get_right()):
                    is_dropped = True
            for i in dropped_indices_2:
                if np.allclose(end_dot, layer2_dots[i].get_left()):
                    is_dropped = True

            if is_dropped:
                lines_to_fade.add(line)

        for line in lines_to_fade:
            animations.append(line.animate.set_stroke(opacity=0.0))

        with self.voiceover(text="Now, when we apply dropout, we introduce a probability parameter, typically denoted as 'p'. For each training step, every neuron has a probability 'p' of being temporarily removed from the network. By turning them off, represented here in red, all their incoming and outgoing connections are completely severed for that specific forward and backward pass.") as tracker:
            self.play(*animations)

        with self.voiceover(text="This constant, random disruption forces the network to learn redundant and highly robust representations. Because no single neuron can be guaranteed to be present during any given training step, the network cannot rely too heavily on any single feature or specific pathway, which naturally distributes the learning across the entire layer.") as tracker:
            self.wait(1)

        self.wait(1.5)

        self.play(FadeOut(VGroup(definition, nn_group)))

        # Section 2: Why do we need it?
        with self.voiceover(text="So, why do we actually need this seemingly chaotic disruption? The primary reason is to combat the pervasive issue of overfitting, where a model essentially memorizes the training data instead of learning generalizable patterns.") as tracker:
            subtitle = Tex("Why do we need it?", color=BLUE).scale(1.2).next_to(title, DOWN, buff=0.5)
            self.play(Write(subtitle))

        # Draw Overfitting vs Well-generalized curves
        axes_overfit = Axes(
            x_range=[0, 10, 1], y_range=[0, 10, 1],
            x_length=4, y_length=4,
            axis_config={"color": BLACK}
        ).shift(LEFT * 3 + DOWN * 1)

        axes_good = Axes(
            x_range=[0, 10, 1], y_range=[0, 10, 1],
            x_length=4, y_length=4,
            axis_config={"color": BLACK}
        ).shift(RIGHT * 3 + DOWN * 1)

        # Training and validation loss curves (overfit)
        train_curve_overfit = axes_overfit.plot(lambda x: 8 * np.exp(-0.5 * x) + 0.5, color=GREEN)
        val_curve_overfit = axes_overfit.plot(lambda x: 8 * np.exp(-0.5 * x) + 0.5 + 0.1 * x**2, color=RED)

        # Training and validation loss curves (good)
        train_curve_good = axes_good.plot(lambda x: 8 * np.exp(-0.4 * x) + 1.0, color=GREEN)
        val_curve_good = axes_good.plot(lambda x: 8 * np.exp(-0.4 * x) + 1.5, color=BLUE)

        label_overfit = Tex("Without Dropout\\\\(Overfitting)", color=BLACK).scale(0.7).next_to(axes_overfit, DOWN)
        label_good = Tex("With Dropout\\\\(Better Generalization)", color=BLACK).scale(0.7).next_to(axes_good, DOWN)

        legend = VGroup(
            Line(LEFT, RIGHT, color=GREEN).scale(0.3), Tex("Train Loss", color=BLACK).scale(0.5),
            Line(LEFT, RIGHT, color=RED).scale(0.3), Tex("Val Loss (Overfit)", color=BLACK).scale(0.5),
            Line(LEFT, RIGHT, color=BLUE).scale(0.3), Tex("Val Loss (Good)", color=BLACK).scale(0.5)
        ).arrange(RIGHT, buff=0.2).next_to(subtitle, DOWN, buff=0.3)

        with self.voiceover(text="Let's visualize this by looking at standard training and validation loss curves over time. In a network without dropout, as the model continues to train epoch after epoch, it eventually starts to memorize the noise and specifics of the training dataset. You will see the green training loss keep dropping beautifully.") as tracker:
            self.play(Create(axes_overfit), Write(label_overfit), Create(legend))
            self.play(Create(train_curve_overfit))

        with self.voiceover(text="However, the moment of truth comes when we evaluate on unseen validation data. You will observe the red validation loss curve hit a minimum and then begin to climb rapidly upwards. This diverging gap is the classic signature of overfitting; the model performs terribly on any new, unseen real-world data.") as tracker:
            self.play(Create(val_curve_overfit))

        with self.voiceover(text="Now let's look at the exact same model architecture, but this time trained with dropout enabled. Because the model is forced to generalize and cannot build complex co-adaptations between specific neurons, the learning process is slightly harder and noisier initially.") as tracker:
            self.play(Create(axes_good), Write(label_good))

        with self.voiceover(text="But as a direct result, both the training and the blue validation loss curves track each other much more closely. The validation loss stays low and stable, meaning we have successfully trained a model that generalizes beautifully to data it has never seen before.") as tracker:
            self.play(Create(train_curve_good))
            self.play(Create(val_curve_good))

        self.wait(1.5)

        self.play(FadeOut(VGroup(subtitle, axes_overfit, axes_good, train_curve_overfit, val_curve_overfit, train_curve_good, val_curve_good, label_overfit, label_good, legend)))

        # Section 3: Use Cases
        with self.voiceover(text="So where exactly is dropout used in the real world? The short answer is: almost everywhere that deep learning is applied, especially in large, over-parameterized models.") as tracker:
            subtitle_use_cases = Tex("Real-world Use Cases", color=BLUE).scale(1.2).next_to(title, DOWN, buff=0.5)
            self.play(Write(subtitle_use_cases))

        case1 = VGroup(
            Tex(r"\textbf{1. Natural Language Processing (BERT)}", color=BLACK),
            Tex("Google uses dropout extensively in transformer models\\\\like BERT to prevent the massive attention layers\\\\from overfitting on the training corpus.", color=BLACK).scale(0.7)
        ).arrange(DOWN, aligned_edge=LEFT).shift(LEFT * 1 + UP * 0.5)

        case2 = VGroup(
            Tex(r"\textbf{2. Recommender Systems (Spotify)}", color=BLACK),
            Tex("Companies like Spotify use dropout in their deep learning-based\\\\recommendation engines so the model doesn't overfit\\\\to a user's exact past listening history,\\\\allowing for serendipitous song discovery.", color=BLACK).scale(0.7)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(case1, DOWN, buff=1.0)

        use_cases_group = VGroup(case1, case2).move_to(ORIGIN)

        with self.voiceover(text="For a prime example, consider Natural Language Processing. Tech giants like Google rely heavily on dropout inside massive transformer architectures like BERT. These models have hundreds of millions, or even billions, of parameters. Without applying dropout to the dense feed-forward networks and the attention layers, these massive models would simply act as giant lookup tables, memorizing the exact text of their training corpus instead of learning the underlying structure of human language.") as tracker:
            self.play(Write(case1))

        with self.voiceover(text="As a second example, consider recommender systems. Companies like Spotify utilize deep neural networks to drive their recommendation engines. By injecting dropout during training, they actively prevent the model from overfitting strictly to a user's past historical interactions. This forced generalization is actually what allows the algorithm to suggest novel, serendipitous song recommendations that you haven't heard before, rather than just repeating your exact past listening history back to you in a loop.") as tracker:
            self.play(Write(case2))

        self.wait(1.5)

        self.play(FadeOut(VGroup(subtitle_use_cases, use_cases_group)))

        # Section 4: Key Interview Insight
        with self.voiceover(text="Finally, let's cover the absolute most critical thing you need to know to pass your machine learning engineering interview: the key operational difference between how dropout behaves during training versus inference.") as tracker:
            insight_title = Tex("Key Interview Insight", color=PURPLE).scale(1.2).next_to(title, DOWN, buff=0.5)
            self.play(Write(insight_title))

        box = Rectangle(width=10, height=4, color=PURPLE, fill_opacity=0.1)
        box.next_to(insight_title, DOWN, buff=0.5)

        insight_text_1 = Tex(r"\textbf{Training vs. Inference Phase}", color=BLACK)
        insight_text_2 = Tex(r"During \textbf{Training}: Dropout is active. Activations are scaled.", color=BLACK).scale(0.8)
        insight_text_3 = Tex(r"During \textbf{Inference (Test)}: Dropout is \textbf{turned off}.", color=RED).scale(0.8)

        insight_content = VGroup(insight_text_1, insight_text_2, insight_text_3).arrange(DOWN, buff=0.5).move_to(box.get_center())

        with self.voiceover(text="Interviewers love to test your practical engineering knowledge by asking this specific question. It acts as a filter to see who just knows the theory versus who has actually deployed a model to production.") as tracker:
            self.play(Create(box), Write(insight_text_1))

        with self.voiceover(text="During the active training phase, as we discussed, dropout is turned on. Neurons are randomly dropped based on probability 'p', and importantly, the remaining active neurons have their output values mathematically scaled up. This scaling ensures that the overall expected value, or the mathematical sum of the layer's output, remains relatively consistent regardless of which neurons were dropped.") as tracker:
            self.play(Write(insight_text_2))

        with self.voiceover(text="However, when you finish training and move your model into production for inference or testing, dropout must be completely turned off. You want all of your neurons fully active and working together to give you the absolute best, highly deterministic prediction possible. If you make the rookie mistake of leaving dropout enabled in production, your model's predictions will fluctuate wildly and randomly every single time you query it with the exact same input!") as tracker:
            self.play(Write(insight_text_3))

        self.wait(2)

        # Math transition for scaling
        with self.voiceover(text="To really impress your interviewer, let's look at the exact mathematical implementation of this scaling, specifically a technique known as 'inverted dropout', which is the standard implementation used under the hood by modern deep learning frameworks like PyTorch and TensorFlow.") as tracker:
            self.play(FadeOut(VGroup(insight_text_1, insight_text_2, insight_text_3)))

        math_1 = MathTex("x_{train}", "=", "x", "\\cdot", "\\text{mask}", color=BLACK).scale(1.5)
        math_2 = MathTex("x_{train}", "=", "\\frac{x \\cdot \\text{mask}}{1 - p}", color=BLACK).scale(1.5)

        math_1.move_to(box.get_center())
        math_2.move_to(box.get_center())

        with self.voiceover(text="In a naive implementation, you might think we just multiply our input activations, 'x', by a random binary mask consisting of ones and zeros.") as tracker:
            self.play(Write(math_1))

        with self.voiceover(text="But with inverted dropout, to seamlessly keep the expected mathematical value identical between the training and inference phases, we take that masked value and immediately divide it by one minus our dropout probability, 'p', right there during the training step. This clever trick means we don't have to do any messy scaling modifications at inference time; we just turn the mask off and evaluate instantly.") as tracker:
            self.play(TransformMatchingTex(math_1, math_2))

        self.wait(2)

        with self.voiceover(text="That covers everything you need to know about dropout regularization. Understanding both the intuitive geometric reasoning behind it, and the strict engineering rules around inference-time behavior, will absolutely set you apart in a competitive system design or machine learning interview. Thank you so much for watching, keep studying hard, and I will see you in the next video!") as tracker:
            self.play(FadeOut(VGroup(title, insight_title, box, math_2)))

        self.wait(2)
