from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np

class BeamSearchScene(VoiceoverScene):
    def construct(self):
        # 1. Setup speech service & aesthetic
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # We need approx 5-7 mins of voiceover (~750 - 1050 words).
        # We will narrate 4 distinct sections thoroughly.

        # ----- SECTION 1: WHAT IS IT? -----
        title = Text("Beam Search: Sequence Generation", color=BLACK).scale(0.9).to_edge(UP)
        with self.voiceover(text="Welcome to day fifty-five. Today we are diving deep into Beam Search, a foundational sequence generation strategy used across modern artificial intelligence models. Let's start with a clear definition.") as tracker:
            self.play(Write(title))

        definition = Text(
            "A heuristic search algorithm that explores a graph by expanding\n"
            "the most promising nodes in a limited set, called the beam width.",
            color=BLACK,
            font_size=28
        ).next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="What is it? Simply put, Beam Search is a heuristic search algorithm. At each step of sequence generation, instead of just picking the single most likely next word, it explores a graph by keeping track of the top K most promising sequences, where K is known as the beam width.") as tracker:
            self.play(FadeIn(definition, shift=UP))
            self.wait(1.5)

        with self.voiceover(text="By evaluating multiple pathways simultaneously, it avoids getting trapped in local optima while remaining far more computationally efficient than an exhaustive search.") as tracker:
            self.wait(1)

        self.play(FadeOut(definition))

        # Visualizing the Tree
        with self.voiceover(text="Let us visualize this search process. Imagine our model is trying to generate a sentence. It starts at a root node, and at the first time step, it generates several possible tokens.") as tracker:
            root_node = Circle(radius=0.4, color=BLACK, fill_color=BLUE, fill_opacity=0.3).shift(UP*2)
            root_label = Text("Start", color=BLACK, font_size=24).move_to(root_node)
            self.play(FadeIn(root_node), Write(root_label))

            t1_n1 = Circle(radius=0.3, color=BLACK, fill_color=GREEN, fill_opacity=0.3).move_to(LEFT*2.5 + UP*0.5)
            t1_n2 = Circle(radius=0.3, color=BLACK, fill_color=GREEN, fill_opacity=0.3).move_to(LEFT*0 + UP*0.5)
            t1_n3 = Circle(radius=0.3, color=BLACK, fill_color=RED, fill_opacity=0.3).move_to(RIGHT*2.5 + UP*0.5)

            l1 = Line(root_node.get_bottom(), t1_n1.get_top(), color=BLACK)
            l2 = Line(root_node.get_bottom(), t1_n2.get_top(), color=BLACK)
            l3 = Line(root_node.get_bottom(), t1_n3.get_top(), color=BLACK)

            self.play(
                Write(l1), Write(l2), Write(l3),
                FadeIn(t1_n1), FadeIn(t1_n2), FadeIn(t1_n3)
            )

        with self.voiceover(text="If we use a beam width of 2, the algorithm evaluates the probabilities and only keeps the top 2 most promising nodes, discarding the rest. In this case, the green nodes are kept, and the red node is pruned.") as tracker:
            cross = Cross(t1_n3, stroke_color=RED, stroke_width=6)
            self.play(Write(cross))
            self.wait(1.5)

        with self.voiceover(text="In the next time step, the model expands only the two preserved nodes. It generates all possible next tokens for these two pathways, calculates their cumulative sequence probabilities, and again, strictly keeps only the top 2 sequences overall.") as tracker:
            t2_n1 = Circle(radius=0.3, color=BLACK, fill_color=GREEN, fill_opacity=0.3).move_to(LEFT*3.5 + DOWN*1)
            t2_n2 = Circle(radius=0.3, color=BLACK, fill_color=RED, fill_opacity=0.3).move_to(LEFT*1.5 + DOWN*1)

            t2_n3 = Circle(radius=0.3, color=BLACK, fill_color=GREEN, fill_opacity=0.3).move_to(RIGHT*0.5 + DOWN*1)
            t2_n4 = Circle(radius=0.3, color=BLACK, fill_color=RED, fill_opacity=0.3).move_to(RIGHT*2 + DOWN*1)

            l4 = Line(t1_n1.get_bottom(), t2_n1.get_top(), color=BLACK)
            l5 = Line(t1_n1.get_bottom(), t2_n2.get_top(), color=BLACK)
            l6 = Line(t1_n2.get_bottom(), t2_n3.get_top(), color=BLACK)
            l7 = Line(t1_n2.get_bottom(), t2_n4.get_top(), color=BLACK)

            self.play(
                Write(l4), Write(l5), Write(l6), Write(l7),
                FadeIn(t2_n1), FadeIn(t2_n2), FadeIn(t2_n3), FadeIn(t2_n4)
            )

            cross2 = Cross(t2_n2, stroke_color=RED, stroke_width=6)
            cross3 = Cross(t2_n4, stroke_color=RED, stroke_width=6)
            self.play(Write(cross2), Write(cross3))
            self.wait(1.5)

        with self.voiceover(text="This process repeats until an end-of-sequence token is generated. Beam Search acts as a sliding window of possibilities, offering a balanced middle ground between the narrowness of greedy decoding and the impossible cost of exploring every single combination.") as tracker:
            self.wait(1)
            self.play(FadeOut(VGroup(root_node, root_label, t1_n1, t1_n2, t1_n3, l1, l2, l3, cross, t2_n1, t2_n2, t2_n3, t2_n4, l4, l5, l6, l7, cross2, cross3)))


        # ----- SECTION 2: WHY DO WE NEED IT? -----
        subtitle = Text("Why do we need it?", color=BLUE).scale(0.8).next_to(title, DOWN, buff=0.5)
        with self.voiceover(text="So, why do we need Beam Search in the first place? To understand this, we have to look at the limitations of the simpler alternative: Greedy Decoding.") as tracker:
            self.play(Write(subtitle))

        greedy_title = Text("1. Greedy Decoding (Before)", color=BLACK, font_size=28).move_to(UP*1.5 + LEFT*3)
        with self.voiceover(text="In Greedy Decoding, the model generates the sequence by taking the `argmax` at each step. It simply picks the highest probability word right now, completely ignoring the future context.") as tracker:
            self.play(Write(greedy_title))
            eq_greedy = MathTex(
                r"y_t = \arg\max_{y} P(y \mid y_1, \dots, y_{t-1}, x)", color=BLACK
            ).scale(0.8).next_to(greedy_title, DOWN, buff=0.5)
            self.play(FadeIn(eq_greedy, shift=DOWN))
            self.wait(1.5)

        with self.voiceover(text="Imagine the model has to choose between starting a sentence with 'The' which has a 40 percent probability, or 'A' which has a 30 percent probability. Greedy decoding immediately locks in 'The'.") as tracker:
            greedy_ex = MathTex(r"P(\text{The}) = 0.40 \quad > \quad P(\text{A}) = 0.30", color=BLACK).scale(0.7).next_to(eq_greedy, DOWN, buff=0.5)
            self.play(Write(greedy_ex))
            self.wait(1)

        with self.voiceover(text="But what if the path starting with 'A' actually leads to a much better, more highly probable complete sentence down the line? Greedy decoding will never know, because it permanently discarded 'A' at step 1. It is short-sighted and frequently generates suboptimal, disjointed sequences.") as tracker:
            self.wait(1.5)

        beam_title = Text("2. Beam Search (After)", color=BLACK, font_size=28).move_to(UP*1.5 + RIGHT*3)
        with self.voiceover(text="This is exactly the problem Beam Search solves. By keeping the top K paths, we calculate the joint probability of the entire sequence.") as tracker:
            self.play(Write(beam_title))
            eq_beam = MathTex(
                r"\max", r"\prod_{t=1}^{T}", r"P(y_t \mid y_{<t}, x)", color=BLACK
            ).scale(0.8).next_to(beam_title, DOWN, buff=0.5)
            self.play(Write(eq_beam[0]))
            self.play(Write(eq_beam[1]))
            self.play(Write(eq_beam[2]))
            self.wait(1.5)

        with self.voiceover(text="Often, we compute this using log probabilities to prevent underflow, changing the multiplication into a summation of log probabilities.") as tracker:
            eq_beam_log = MathTex(
                r"\max", r"\sum_{t=1}^{T}", r"\log", r"P(y_t \mid y_{<t}, x)", color=BLACK
            ).scale(0.8).next_to(beam_title, DOWN, buff=0.5)
            self.play(TransformMatchingTex(eq_beam, eq_beam_log))
            self.wait(1.5)

        with self.voiceover(text="Because Beam Search maintains multiple candidates, if the path starting with 'A' eventually becomes mathematically stronger by step 3 or 4, the algorithm can pivot to it. It trades a small amount of extra compute for a massive improvement in overall sequence quality.") as tracker:
            self.wait(1.5)
            self.play(FadeOut(VGroup(subtitle, greedy_title, eq_greedy, greedy_ex, beam_title, eq_beam_log)))


        # ----- SECTION 3: USE CASES -----
        subtitle_cases = Text("Real-World Use Cases", color=GREEN).scale(0.8).next_to(title, DOWN, buff=0.5)
        with self.voiceover(text="Where is Beam Search used in the real world? It is the backbone of almost any application requiring highly accurate sequence generation, especially before the era of stochastic LLM sampling.") as tracker:
            self.play(Write(subtitle_cases))

        # Use Case 1
        uc1 = Text("1. Google Translate (Machine Translation)", color=BLACK, font_size=28).move_to(UP*1 + LEFT*2)
        with self.voiceover(text="First, consider Machine Translation systems like Google Translate. When translating a sentence from French to English, grammatical structure changes significantly.") as tracker:
            self.play(FadeIn(uc1, shift=RIGHT))

        uc1_desc = Text(
            "Maintains multiple translation hypotheses\n"
            "to ensure the final sentence is grammatically sound.",
            color=DARK_GRAY,
            font_size=22
        ).next_to(uc1, DOWN, aligned_edge=LEFT)
        with self.voiceover(text="Google Translate uses Beam Search to maintain multiple translation hypotheses at once. This ensures that the system doesn't commit to a poor grammatical structure early on, resulting in a final English sentence that is both accurate and fluent.") as tracker:
            self.play(Write(uc1_desc))
            self.wait(1.5)

        # Use Case 2
        uc2 = Text("2. OpenAI Whisper (Speech Recognition)", color=BLACK, font_size=28).move_to(DOWN*1.5 + LEFT*2)
        with self.voiceover(text="Second, consider Automatic Speech Recognition systems, such as OpenAI's Whisper. Audio data is often noisy, and homophones—words that sound identical but mean different things—are common.") as tracker:
            self.play(FadeIn(uc2, shift=RIGHT))

        uc2_desc = Text(
            "Disambiguates phonemes and homophones by\n"
            "evaluating the broader linguistic context.",
            color=DARK_GRAY,
            font_size=22
        ).next_to(uc2, DOWN, aligned_edge=LEFT)
        with self.voiceover(text="Whisper uses Beam Search to transcribe the audio. By keeping multiple transcript candidates alive, the model can look at the broader context of the sentence to correctly disambiguate words, dramatically reducing transcription errors.") as tracker:
            self.play(Write(uc2_desc))
            self.wait(1.5)

        self.play(FadeOut(VGroup(subtitle_cases, uc1, uc1_desc, uc2, uc2_desc)))


        # ----- SECTION 4: KEY INTERVIEW INSIGHT -----
        subtitle_insight = Text("Key Interview Insight", color=PURPLE).scale(0.8).next_to(title, DOWN, buff=0.5)
        with self.voiceover(text="Now, let us discuss the most critical part: the key interview insight. If you are asked about Beam Search in an AI engineering interview, they will almost certainly test your knowledge on its major flaw: Length Penalty.") as tracker:
            self.play(Write(subtitle_insight))

        box = Rectangle(width=10, height=4, color=PURPLE, fill_color=PURPLE_A, fill_opacity=0.1).shift(DOWN*0.5)
        insight_title = Text("The Length Bias Tradeoff", color=BLACK, font_size=32, weight=BOLD).move_to(box.get_top() + DOWN*0.5)

        with self.voiceover(text="Because Beam Search calculates the joint probability by multiplying probabilities together—or adding negative log probabilities—longer sequences will inherently have lower total probabilities.") as tracker:
            self.play(FadeIn(box), Write(insight_title))

            math_prob = MathTex(r"P < 1 \implies \text{More terms } = \text{ Smaller product}", color=BLACK).scale(0.8).next_to(insight_title, DOWN, buff=0.5)
            self.play(Write(math_prob))
            self.wait(1.5)

        insight_text = Text(
            "Without correction, Beam Search heavily biases toward short, truncated outputs.",
            color=BLACK,
            font_size=24
        ).next_to(math_prob, DOWN, buff=0.5)

        with self.voiceover(text="Since every probability is a fraction less than 1, multiplying more of them together naturally results in a smaller number. As a result, standard Beam Search exhibits a strong, unfair bias towards very short, truncated sequences, simply because they have fewer terms to multiply.") as tracker:
            self.play(FadeIn(insight_text, shift=UP))
            self.wait(1.5)

        with self.voiceover(text="To fix this, we apply a Length Penalty during the scoring phase. We divide the cumulative log probability by the sequence length raised to a penalty exponent alpha, typically around 0.7. This normalizes the scores and allows longer, detailed sentences to compete fairly against shorter ones.") as tracker:
            penalty_math = MathTex(
                r"\text{Score} = \frac{\sum \log P(y_t)}{L^\alpha}", color=BLUE
            ).scale(0.9).next_to(insight_text, DOWN, buff=0.5)
            self.play(Write(penalty_math))
            self.wait(1.5)

        with self.voiceover(text="Additionally, interviewers may ask about Beam Width versus Cost. A larger Beam Width, say 5 or 10, yields better quality sequences but linearly increases the memory and compute cost. However, pushing it too high, like to 50, actually hurts performance, causing models to output safe but incredibly generic text.") as tracker:
            self.wait(2)

        with self.voiceover(text="That concludes our deep dive into Beam Search. You now understand its mechanics, why it replaces greedy decoding, its real world applications, and the critical length penalty tradeoff. Keep building, and see you tomorrow.") as tracker:
            self.wait(2)
            self.play(FadeOut(VGroup(title, subtitle_insight, box, insight_title, math_prob, insight_text, penalty_math)))
            self.wait(1)
