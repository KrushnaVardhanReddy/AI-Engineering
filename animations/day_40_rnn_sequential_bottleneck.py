from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class RNNSequentialBottleneck(VoiceoverScene):
    def construct(self):
        # Setup aesthetic (Whiteboard Style)
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Title
        title = Text("RNN Sequential Bottleneck", font_size=48, color=BLACK, weight=BOLD)
        title.to_edge(UP)

        with self.voiceover(text="Welcome to Day 40 of AI Engineering Mastery. Today, we are taking a deep dive into the RNN Sequential Bottleneck. This is a fundamental structural limitation in classical sequence models that ultimately forced the AI industry to rethink how we process text and time-series data. Grasping this concept is absolutely essential for understanding why modern architectures were invented in the first place.") as tracker:
            self.play(Write(title))
            self.wait(2.0)

        # 1. What is it?
        what_title = Text("1. What is it?", font_size=36, color=BLUE, weight=BOLD).next_to(title, DOWN, buff=0.5).align_to(title, LEFT)

        definition = Text("Recurrent Neural Networks must process sequences one step at a time.", font_size=24, color=BLACK)
        definition.next_to(what_title, DOWN, buff=0.3).align_to(what_title, LEFT)

        with self.voiceover(text="First, what exactly is the RNN Sequential Bottleneck? Put simply, Recurrent Neural Networks are designed to process sequences incrementally, one step at a time. The fundamental rule of an RNN is that the hidden state calculation at time step t depends strictly on the completed hidden state from time step t minus one. You cannot jump ahead.") as tracker:
            self.play(FadeIn(what_title), Write(definition))
            self.wait(2.0)

        # Visual for What is it? (Unrolled RNN)
        tokens = ["The", "cat", "sat"]
        nodes = VGroup()
        arrows = VGroup()

        for i, token in enumerate(tokens):
            node = VGroup()
            rect = Rectangle(width=1.5, height=1.0, color=BLUE, fill_opacity=0.2)
            label = Text("RNN", font_size=20, color=BLACK)
            node.add(rect, label)

            x_input = Text(f"x_{i}: {token}", font_size=20, color=BLACK)
            x_input.next_to(node, DOWN)
            input_arrow = Arrow(x_input.get_top(), node.get_bottom(), buff=0.1, color=BLACK)

            h_output = MathTex(f"h_{i}", color=BLACK, font_size=28)
            h_output.next_to(node, UP)
            output_arrow = Arrow(node.get_top(), h_output.get_bottom(), buff=0.1, color=BLACK)

            full_node = VGroup(node, x_input, input_arrow, h_output, output_arrow)
            nodes.add(full_node)

        nodes.arrange(RIGHT, buff=1.0).next_to(definition, DOWN, buff=0.8)

        for i in range(len(nodes) - 1):
            arrow = Arrow(nodes[i][0].get_right(), nodes[i+1][0].get_left(), buff=0.1, color=RED)
            arrows.add(arrow)

        with self.voiceover(text="Let's visualize this with an unrolled RNN processing a short sentence: 'The', 'cat', and 'sat'. Notice the red arrows connecting the blocks. The network physically cannot begin processing the word 'cat' until it has entirely finished processing the word 'The' and passed the resulting hidden state forward. This rigid sequence dependency is what creates a severe computational bottleneck, especially for long documents.") as tracker:
            for i, node in enumerate(nodes):
                self.play(FadeIn(node[1]), GrowArrow(node[2])) # x_i and arrow
                self.play(Create(node[0]), run_time=0.5) # RNN block
                self.play(GrowArrow(node[4]), FadeIn(node[3])) # h_i and arrow
                if i < len(arrows):
                    self.play(GrowArrow(arrows[i])) # red h_i to next RNN arrow
            self.wait(2.0)

        # Equation derivation step-by-step
        eq_step1 = MathTex(r"h_t", r"=", r"f(", r"W_{hx}", r"x_t", r")", color=BLACK, font_size=36)
        eq_step1.next_to(nodes, DOWN, buff=0.5)

        with self.voiceover(text="Mathematically, we can see this bottleneck clearly. Initially, you might think a hidden state is just a function of the current input word x t, multiplied by some weight matrix.") as tracker:
            self.play(Write(eq_step1))
            self.wait(1.5)

        eq_step2 = MathTex(r"h_t", r"=", r"f(", r"W_{hh}", r"h_{t-1}", r"+", r"W_{hx}", r"x_t", r")", color=BLACK, font_size=36)
        eq_step2.next_to(nodes, DOWN, buff=0.5)

        with self.voiceover(text="But in an RNN, we must also add the recurrent connection. We take the previous hidden state h t minus one, and multiply it by the recurrent weight matrix. Because of this explicit dependence on h t minus one, the computation for step t is locked until step t minus one finishes. This operation cannot be parallelized across the time dimension.") as tracker:
            self.play(TransformMatchingTex(eq_step1, eq_step2))
            self.wait(2.0)

        self.play(
            FadeOut(what_title), FadeOut(definition), FadeOut(nodes), FadeOut(arrows), FadeOut(eq_step2)
        )

        # 2. Why do we need it (the bottleneck / problem)?
        why_title = Text("2. The Problem: Sequential vs Parallel", font_size=36, color=BLUE, weight=BOLD).next_to(title, DOWN, buff=0.5).align_to(title, LEFT)

        with self.voiceover(text="Why is this such a massive problem in modern AI? The sequential nature leads to exceptionally slow training times on large datasets. Furthermore, it inherently limits the model's ability to learn long-range dependencies, contributing to the notorious vanishing gradient problem where early context is forgotten.") as tracker:
            self.play(FadeIn(why_title))
            self.wait(2.0)

        # Before (Sequential) vs After (Parallel)
        seq_text = Text("Sequential (RNN) - O(N) Time", font_size=24, color=RED, weight=BOLD).next_to(why_title, DOWN, buff=0.5).align_to(why_title, LEFT)
        par_text = Text("Parallel (Transformer) - O(1) Time", font_size=24, color=GREEN, weight=BOLD).next_to(seq_text, DOWN, buff=1.5).align_to(seq_text, LEFT)

        seq_line = Line(LEFT, RIGHT*4, color=RED).next_to(seq_text, RIGHT, buff=0.5)
        dot = Dot(color=BLACK).move_to(seq_line.get_start())

        with self.voiceover(text="In a sequential model like an RNN, training takes Big O of N time, where N is the length of the sequence. If you feed in a document with ten thousand words, the GPU must sit idle, stepping through each token one by one in a strictly linear fashion.") as tracker:
            self.play(Write(seq_text))
            self.play(Create(seq_line))
            self.play(dot.animate.move_to(seq_line.get_end()), run_time=4)
            self.wait(2.0)

        par_blocks = VGroup(*[Rectangle(width=0.8, height=0.5, color=GREEN, fill_opacity=0.3) for _ in range(7)])
        par_blocks.arrange(RIGHT, buff=0.2).next_to(par_text, RIGHT, buff=0.5)

        with self.voiceover(text="Contrast this with modern architectures like Transformers. By completely removing the sequential bottleneck, Transformers process all tokens simultaneously in parallel. This reduces the time complexity over the sequence length to Big O of 1 for parallelizable operations, allowing us to fully utilize modern GPU hardware and train on billions of words.") as tracker:
            self.play(Write(par_text))
            self.play(FadeIn(par_blocks, shift=UP*0.5, lag_ratio=0))
            self.wait(2.0)

        self.play(
            FadeOut(why_title), FadeOut(seq_text), FadeOut(seq_line), FadeOut(dot), FadeOut(par_text), FadeOut(par_blocks)
        )

        # 3. Use Cases
        uses_title = Text("3. Use Cases (Historical Context)", font_size=36, color=BLUE, weight=BOLD).next_to(title, DOWN, buff=0.5).align_to(title, LEFT)

        google_use = Text("• Google Translate: Initially used Seq2Seq RNNs, suffered from slow training.", font_size=24, color=BLACK).next_to(uses_title, DOWN, buff=0.5).align_to(uses_title, LEFT)
        siri_use = Text("• Apple Siri: Relied on LSTMs for speech processing before Transformers.", font_size=24, color=BLACK).next_to(google_use, DOWN, buff=0.5).align_to(google_use, LEFT)

        with self.voiceover(text="To ground this in reality, let's look at some prominent historical use cases. Google Translate initially built its reputation on Sequence-to-Sequence RNNs. However, as translation datasets grew to petabytes of data, the sequential bottleneck made training entirely new models prohibitively slow, forcing them to look for alternatives.") as tracker:
            self.play(FadeIn(uses_title))
            self.play(Write(google_use))
            self.wait(2.0)

        with self.voiceover(text="Similarly, early versions of Apple's Siri relied heavily on LSTMs, which are a specialized form of RNN. While LSTMs help mitigate vanishing gradients using a memory cell, they still suffer from the fundamental sequential bottleneck. This inherent flaw drove the entire industry's rapid shift toward parallelizable self-attention models.") as tracker:
            self.play(Write(siri_use))
            self.wait(2.0)

        self.play(FadeOut(uses_title), FadeOut(google_use), FadeOut(siri_use))

        # 4. Key Interview Insight
        insight_title = Text("4. Key Interview Insight", font_size=36, color=BLUE, weight=BOLD).next_to(title, DOWN, buff=0.5).align_to(title, LEFT)

        box = Rectangle(width=11, height=5, color=PURPLE, fill_opacity=0.1)
        box.next_to(insight_title, DOWN, buff=0.5)

        insight_text_1 = Text("Interviewer Question:", font_size=24, color=PURPLE, weight=BOLD)
        insight_text_2 = Text('"Why did the industry move from RNNs to Transformers?"', font_size=24, color=BLACK, slant=ITALIC)
        insight_text_3 = Text("Answer: Parallelization.", font_size=28, color=RED, weight=BOLD)
        insight_text_4 = Text("RNNs cannot parallelize training over time due to the sequential bottleneck.", font_size=22, color=BLACK)
        insight_text_5 = Text("Transformers use self-attention to process all tokens at once.", font_size=22, color=BLACK)

        insight_group = VGroup(insight_text_1, insight_text_2, insight_text_3, insight_text_4, insight_text_5)
        insight_group.arrange(DOWN, buff=0.4).move_to(box.get_center())

        with self.voiceover(text="Finally, the Key Interview Insight. When interviewing for AI engineering roles, a very common and highly critical question is: 'Why did the industry move away from RNNs and embrace Transformers?'") as tracker:
            self.play(FadeIn(insight_title), Create(box))
            self.play(Write(insight_text_1), Write(insight_text_2))
            self.wait(2.0)

        with self.voiceover(text="Your answer should immediately focus on parallelization, not just accuracy. You must clearly state that RNNs cannot parallelize training over the time dimension because step t relies entirely on the output of step t minus one. Transformers, on the other hand, use self-attention mechanisms to process all tokens simultaneously in a single matrix multiplication, completely eliminating the sequential bottleneck and enabling massive scale.") as tracker:
            self.play(Write(insight_text_3))
            self.play(FadeIn(insight_text_4))
            self.play(FadeIn(insight_text_5))
            self.wait(3.0)

        # Outro
        with self.voiceover(text="Understanding this structural limitation is absolutely crucial for system design, model selection, and architecting modern AI applications. Thank you for watching, and keep building.") as tracker:
            self.play(FadeOut(VGroup(title, insight_title, box, insight_group)))
            self.wait(2.0)
