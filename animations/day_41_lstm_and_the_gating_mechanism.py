from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class LSTMGatingMechanismScene(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # ==========================================
        # SECTION 1: What is it?
        # ==========================================
        title = Text("LSTM and the Gating Mechanism", color=BLACK).scale(1.2).to_edge(UP)
        with self.voiceover(text="Welcome to day 41 of our AI and Machine Learning interview prep series. Today we are taking a deep dive into Long Short-Term Memory networks, commonly known as LSTMs, and their secret weapon: the Gating Mechanism.") as tracker:
            self.play(Write(title))
            self.wait(1.5)

        definition = Text("A specialized Recurrent Neural Network (RNN)", color=BLACK).scale(0.8)
        definition_pt2 = Text("that learns what to keep, and what to forget.", color=BLUE).scale(0.8)
        def_group = VGroup(definition, definition_pt2).arrange(DOWN, buff=0.2).next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="So, what exactly is an LSTM? At a high level, it is a specialized type of Recurrent Neural Network designed to learn long-term dependencies. It does this by explicitly deciding what information from the past is worth keeping, and what information is no longer relevant and should be forgotten.") as tracker:
            self.play(FadeIn(definition, shift=UP))
            self.play(FadeIn(definition_pt2, shift=UP))
            self.wait(1.5)

        # Draw a simple LSTM Cell
        lstm_box = Rectangle(width=4, height=3, color=BLACK, fill_opacity=0.1)
        lstm_label = Text("LSTM Cell", color=BLACK).move_to(lstm_box)
        cell_group = VGroup(lstm_box, lstm_label).move_to(ORIGIN).shift(DOWN*0.5)

        with self.voiceover(text="At the heart of an LSTM is the memory cell. While a standard RNN has a very simple repeating module, like a single tanh layer, the LSTM relies on a much more complex structure featuring three distinct gates to carefully control the flow of information.") as tracker:
            self.play(FadeOut(def_group))
            self.play(Write(lstm_box), Write(lstm_label))
            self.wait(1.5)

        gates = VGroup(
            Text("1. Forget Gate: Removes irrelevant data", color=RED).scale(0.6),
            Text("2. Input Gate: Adds new relevant data", color=GREEN).scale(0.6),
            Text("3. Output Gate: Computes the next hidden state", color=PURPLE).scale(0.6)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).next_to(lstm_box, RIGHT, buff=1)

        with self.voiceover(text="These three gates are the Forget Gate, which acts first to remove irrelevant data; the Input Gate, which decides what new, relevant data should be added to the cell state; and finally, the Output Gate, which decides what portion of the cell state should be passed on to the next hidden state.") as tracker:
            for gate in gates:
                self.play(FadeIn(gate, shift=LEFT))
                self.wait(0.5)
            self.wait(1.5)

        self.play(FadeOut(cell_group), FadeOut(gates))

        # ==========================================
        # SECTION 2: Why do we need it?
        # ==========================================
        why_title = Text("Why do we need LSTMs?", color=BLACK).scale(1).to_edge(UP)
        with self.voiceover(text="This naturally leads to the question: Why do we need LSTMs in the first place? To understand this, let's look at the limitations of a standard Recurrent Neural Network.") as tracker:
            self.play(Transform(title, why_title))
            self.wait(1.5)

        # Before (Standard RNN)
        rnn_title = Text("Standard RNN", color=RED).scale(0.8).to_edge(LEFT).shift(UP*1.5 + RIGHT*1)

        # Draw a sequence of nodes for Vanishing Gradient
        nodes = VGroup(*[Circle(radius=0.4, color=BLACK) for _ in range(5)])
        nodes.arrange(RIGHT, buff=1).shift(UP*0.2)
        arrows = VGroup(*[Arrow(nodes[i].get_right(), nodes[i+1].get_left(), color=BLACK, buff=0.1) for i in range(4)])

        gradients = VGroup(*[
            Text("grad", color=RED).scale(1 - 0.2*i).next_to(arrows[3-i], UP, buff=0.1)
            for i in range(4)
        ])

        with self.voiceover(text="Standard RNNs suffer from a major flaw known as the vanishing gradient problem. When processing long sequences of data, the gradients used to update the neural network's weights shrink exponentially as they propagate back through time.") as tracker:
            self.play(FadeIn(rnn_title))
            self.play(FadeIn(nodes), FadeIn(arrows))
            self.play(FadeIn(gradients))
            self.wait(1.5)

        with self.voiceover(text="Because the gradients become vanishingly small, the network effectively stops learning. It forgets early context. For example, if a long paragraph starts by mentioning someone grew up in 'France', a standard RNN might completely forget that context by the end of the paragraph, failing to predict that the person speaks 'French'.") as tracker:
            cross = Cross(nodes[0])
            self.play(Write(cross))
            self.wait(1.5)

        self.play(FadeOut(rnn_title), FadeOut(nodes), FadeOut(arrows), FadeOut(gradients), FadeOut(cross))

        # After (LSTM)
        lstm_sol_title = Text("With LSTM", color=GREEN).scale(0.8).to_edge(LEFT).shift(UP*1.5 + RIGHT*1)

        # Cell State Highway
        highway = Line(LEFT*4, RIGHT*4, color=BLUE, stroke_width=8).shift(UP*0.2)
        highway_label = Text("Cell State (The Highway)", color=BLUE).scale(0.6).next_to(highway, UP)

        with self.voiceover(text="LSTMs beautifully solve this issue by introducing a separate pathway called the Cell State, often described as an information highway.") as tracker:
            self.play(FadeIn(lstm_sol_title))
            self.play(Write(highway), FadeIn(highway_label))
            self.wait(1.5)

        # Math derivation with TransformMatchingTex
        math_base = MathTex(r"f_t", r"=", r"\sigma(", r"W_f", r"\cdot [h_{t-1}, x_t] + ", r"b_f", r")", color=BLACK).scale(0.8).shift(DOWN*1.5)
        math_base[0].set_color(RED)

        with self.voiceover(text="Information flows straight down this highway with minimal linear interactions. The gates act like physical valves that control this flow. Let's look at the mathematics of just one of these valves: the forget gate.") as tracker:
            self.play(Write(math_base))
            self.wait(1.5)

        math_step2 = MathTex(r"f_t", r"=", r"\sigma(", r"\text{Weights}", r"\cdot \text{Input} + ", r"\text{Bias}", r")", color=BLACK).scale(0.8).shift(DOWN*1.5)
        math_step2[0].set_color(RED)

        with self.voiceover(text="The forget gate takes the previous hidden state and the current input, multiplies them by a weight matrix, adds a bias, and passes the result through a sigmoid activation function.") as tracker:
            self.play(TransformMatchingTex(math_base, math_step2))
            self.wait(1.5)

        math_step3 = MathTex(r"f_t", r"\in", r"(", r"0", r",", r"1", r")", color=BLACK).scale(0.8).shift(DOWN*1.5)
        math_step3[0].set_color(RED)

        with self.voiceover(text="Because it uses a sigmoid function, the output is squashed to a value strictly between 0 and 1 for every number in the cell state.") as tracker:
            self.play(TransformMatchingTex(math_step2, math_step3))
            self.wait(1.5)

        with self.voiceover(text="A zero literally means 'completely forget this piece of information', and a one means 'keep this piece of information entirely'. By relying on addition and pointwise multiplication, the cell state allows gradients to flow back unaltered, effectively preventing the vanishing gradient problem.") as tracker:
            highlight = SurroundingRectangle(math_step3, color=RED, buff=0.1)
            self.play(Write(highlight))
            self.wait(1.5)

        self.play(FadeOut(lstm_sol_title), FadeOut(highway), FadeOut(highway_label), FadeOut(math_step3), FadeOut(highlight))

        # ==========================================
        # SECTION 3: Use Cases
        # ==========================================
        use_case_title = Text("Real-World Use Cases", color=BLACK).scale(1).to_edge(UP)
        with self.voiceover(text="So, where exactly are LSTMs used in the real world? While newer architectures exist today, LSTMs have historically been the backbone of processing sequential data.") as tracker:
            self.play(Transform(title, use_case_title))
            self.wait(1.5)

        uc1 = Text("1. Apple / Siri: Speech Recognition", color=BLACK).scale(0.8).shift(UP*0.5)
        uc1_sub = Text("Processing continuous audio signals into text.", color=BLUE).scale(0.6).next_to(uc1, DOWN)

        with self.voiceover(text="For many years, companies like Apple used LSTMs in virtual assistants like Siri for speech recognition. When translating audio waves into text, understanding the current word often depends heavily on the context of the words that came several seconds before it.") as tracker:
            self.play(FadeIn(uc1, shift=RIGHT))
            self.play(FadeIn(uc1_sub))
            self.wait(1.5)

        uc2 = Text("2. Google Translate: Machine Translation", color=BLACK).scale(0.8).shift(DOWN*1)
        uc2_sub = Text("Mapping sequences of varying lengths (English to Spanish).", color=GREEN).scale(0.6).next_to(uc2, DOWN)

        with self.voiceover(text="Similarly, Google Translate relied massively on LSTM-based sequence-to-sequence models to translate entire sentences. Translation isn't a word-for-word process; grammatical context must be maintained over long distances, which LSTMs handled beautifully through their cell state.") as tracker:
            self.play(FadeIn(uc2, shift=RIGHT))
            self.play(FadeIn(uc2_sub))
            self.wait(1.5)

        self.play(FadeOut(uc1), FadeOut(uc1_sub), FadeOut(uc2), FadeOut(uc2_sub))

        # ==========================================
        # SECTION 4: Key Interview Insight
        # ==========================================
        insight_title = Text("Key Interview Insight", color=BLACK).scale(1).to_edge(UP)
        with self.voiceover(text="Finally, let's discuss a key insight that frequently comes up in machine learning and AI engineering interviews today.") as tracker:
            self.play(Transform(title, insight_title))
            self.wait(1.5)

        box = Rectangle(width=10, height=4, color=PURPLE, fill_opacity=0.05)
        insight_text = Text("Transformers vs. LSTMs", color=BLACK).scale(0.9).next_to(box.get_top(), DOWN, buff=0.3)

        tradeoff1 = Text("LSTMs: Sequential computation (slow to train).", color=RED).scale(0.7)
        tradeoff2 = Text("Transformers: Parallel computation (fast, scalable).", color=GREEN).scale(0.7)

        tradeoffs = VGroup(tradeoff1, tradeoff2).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(insight_text, DOWN, buff=0.5)

        with self.voiceover(text="If you are asked about sequence modeling, interviewers will almost always ask you to compare LSTMs to the modern Transformer architecture.") as tracker:
            self.play(Write(box), Write(insight_text))
            self.wait(1.5)

        with self.voiceover(text="The most critical tradeoff you must mention is the training speed. LSTMs, by their very nature, process data sequentially, one time-step after another. This means they cannot be easily parallelized, making them incredibly slow to train on massive modern datasets using GPUs.") as tracker:
            self.play(FadeIn(tradeoff1, shift=UP))
            self.wait(1.5)

        with self.voiceover(text="Transformers, on the other hand, abandoned recurrence entirely in favor of self-attention mechanisms. They process entire sequences in parallel, which scales vastly better. However, you can score bonus points by noting that LSTMs can sometimes still be more memory-efficient during real-time inference on edge devices, because they don't require storing the entire context window in memory.") as tracker:
            self.play(FadeIn(tradeoff2, shift=UP))
            self.wait(1.5)

        with self.voiceover(text="Remember: while Transformers dominate today's large language models, LSTMs introduced the crucial concept of explicit memory gates, paving the fundamental way for modern sequence modeling. Thanks for watching, and good luck with your interview prep!") as tracker:
            self.play(Write(SurroundingRectangle(tradeoffs, color=BLUE, buff=0.2)))
            self.wait(2)

        self.play(FadeOut(Group(*self.mobjects)))
