from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class DotProductVisualization(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Section 1: What is it?
        title = Text("What is the Dot Product?", color=BLACK, font_size=48).to_edge(UP)
        self.play(Write(title))

        with self.voiceover(text="Welcome to our interview prep series! Today we are looking at the dot product. So, what exactly is the dot product? In simple terms, it is a fundamental mathematical operation that takes two vectors and returns a single, scalar number. This number elegantly represents how much the two vectors point in the same direction. It is a core concept in linear algebra, and is absolutely essential for understanding how machine learning algorithms operate under the hood, especially in natural language processing and recommendation systems.") as tracker:

            vec1 = Arrow(start=ORIGIN, end=RIGHT*2 + UP*1, color=BLUE, buff=0)
            vec1_label = MathTex("\\vec{A}", color=BLUE).next_to(vec1.get_end(), UP)

            vec2 = Arrow(start=ORIGIN, end=RIGHT*3, color=RED, buff=0)
            vec2_label = MathTex("\\vec{B}", color=RED).next_to(vec2.get_end(), RIGHT)

            axes = Axes(
                x_range=[-1, 4, 1],
                y_range=[-1, 3, 1],
                axis_config={"color": BLACK},
                x_length=5,
                y_length=4
            ).shift(LEFT * 2)

            vec_group = VGroup(axes, vec1, vec1_label, vec2, vec2_label)

            self.play(Create(axes), Create(vec1), Write(vec1_label), Create(vec2), Write(vec2_label))

            projection_line = DashedLine(
                start=vec1.get_end(),
                end=RIGHT*2,
                color=BLACK
            )
            projection_vec = Arrow(start=ORIGIN, end=RIGHT*2, color=PURPLE, buff=0)
            projection_label = Tex("Projection", color=PURPLE, font_size=24).next_to(projection_vec, DOWN)

            self.play(Create(projection_line))
            self.play(Create(projection_vec), Write(projection_label))

            eq1 = MathTex("\\vec{A} \\cdot \\vec{B}", color=BLACK).to_edge(RIGHT).shift(UP)
            eq2 = MathTex("=", "|\\vec{A}|", "|\\vec{B}|", "\\cos(\\theta)", color=BLACK).next_to(eq1, DOWN)
            eq_combined = MathTex("\\vec{A} \\cdot \\vec{B}", "=", "|\\vec{A}|", "|\\vec{B}|", "\\cos(\\theta)", color=BLACK).to_edge(RIGHT)

            self.play(Write(eq1))
            self.wait(0.5)
            self.play(Write(eq2))
            self.wait(0.5)
            self.play(TransformMatchingTex(VGroup(eq1, eq2), eq_combined))

            self.wait(1.5)

        self.play(FadeOut(vec_group), FadeOut(projection_line), FadeOut(projection_vec), FadeOut(projection_label), FadeOut(eq_combined), FadeOut(title))

        # Section 2: Why do we need it?
        title2 = Text("Why Do We Need It?", color=BLACK, font_size=48).to_edge(UP)
        self.play(Write(title2))

        with self.voiceover(text="So why do we need it? Why is it so crucial for artificial intelligence? Imagine trying to find similar items, like matching a search query to a document, without the dot product. You would just have a list of arbitrary coordinates in space, with no efficient mathematical way to measure their alignment or similarity. You would be lost in high-dimensional space without a compass to guide you.") as tracker:

            no_dot_text = Text("Without Dot Product: Just points in space", color=RED, font_size=32).shift(UP*2)
            self.play(Write(no_dot_text))

            points = VGroup(*[Dot(point=[np.random.uniform(-3, 3), np.random.uniform(-1, 1), 0], color=BLACK) for _ in range(5)])
            self.play(Create(points))
            self.wait(1.5)

        with self.voiceover(text="But with the dot product, we can instantly and efficiently calculate similarity. It gives us a concrete, reliable metric: the higher the dot product, the more aligned and similar the vectors are. This seemingly simple operation is the mathematical foundation of semantic search, allowing us to find meaning and relationships across vast amounts of complex data.") as tracker:
            self.play(FadeOut(no_dot_text), FadeOut(points))

            with_dot_text = Text("With Dot Product: Measuring Similarity", color=GREEN, font_size=32).shift(UP*2)
            self.play(Write(with_dot_text))

            vec_a = Arrow(start=ORIGIN, end=UP*2 + RIGHT, color=BLUE, buff=0)
            vec_b1 = Arrow(start=ORIGIN, end=UP*2 + RIGHT*1.5, color=GREEN, buff=0)
            vec_b2 = Arrow(start=ORIGIN, end=DOWN*2 + RIGHT, color=RED, buff=0)

            label_a = Text("Query", color=BLUE, font_size=24).next_to(vec_a.get_end(), UP)
            label_b1 = Text("Match", color=GREEN, font_size=24).next_to(vec_b1.get_end(), UP)
            label_b2 = Text("Unrelated", color=RED, font_size=24).next_to(vec_b2.get_end(), DOWN)

            self.play(Create(vec_a), Write(label_a))
            self.play(Create(vec_b1), Write(label_b1), Create(vec_b2), Write(label_b2))

            dot_high = MathTex("\\vec{A} \\cdot \\vec{B}_{match} \\gg 0", color=GREEN).next_to(vec_b1, RIGHT, buff=1)
            dot_low = MathTex("\\vec{A} \\cdot \\vec{B}_{unrelated} < 0", color=RED).next_to(vec_b2, RIGHT, buff=1)

            self.play(Write(dot_high), Write(dot_low))
            self.wait(1.5)

        self.play(FadeOut(with_dot_text), FadeOut(vec_a), FadeOut(vec_b1), FadeOut(vec_b2), FadeOut(label_a), FadeOut(label_b1), FadeOut(label_b2), FadeOut(dot_high), FadeOut(dot_low), FadeOut(title2))

        # Section 3: Use Cases
        title3 = Text("Real-World Use Cases", color=BLACK, font_size=48).to_edge(UP)
        self.play(Write(title3))

        with self.voiceover(text="Where is this used in practice? The answer is everywhere in modern AI. For example, consider large language models like ChatGPT. The core of their architecture is the Transformer attention mechanism. Within this mechanism, the dot product is calculated between Query and Key vectors to determine exactly which words in a sentence are most relevant to each other, allowing the model to understand context and nuance.") as tracker:

            case1_title = Text("1. ChatGPT (Transformers)", color=BLUE, font_size=36).shift(UP*1.5)
            self.play(Write(case1_title))

            attention_eq = MathTex("\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q \\cdot K^T}{\\sqrt{d_k}}\\right) V", color=BLACK)
            self.play(Write(attention_eq))
            self.wait(1.5)
            self.play(FadeOut(case1_title), FadeOut(attention_eq))

        with self.voiceover(text="Another prominent example is Spotify. They heavily utilize the dot product in their sophisticated recommendation systems. They match your specific user preference vector with a song's unique feature vector. A high dot product between these two vectors means the song aligns well with your tastes, and therefore you will probably like it! This is how they curate those incredibly accurate personalized playlists.") as tracker:

            case2_title = Text("2. Spotify (Recommendations)", color=GREEN, font_size=36).shift(UP*1.5)
            self.play(Write(case2_title))

            user_vec = MathTex("\\text{User Vector } \\vec{U}", color=BLUE).shift(LEFT*2)
            song_vec = MathTex("\\text{Song Vector } \\vec{S}", color=RED).shift(RIGHT*2)
            dot_op = MathTex("\\cdot", color=BLACK).move_to(ORIGIN)

            result = Text("High Score = Good Recommendation", color=BLACK, font_size=28).shift(DOWN*1.5)

            self.play(Write(user_vec), Write(song_vec))
            self.play(Write(dot_op))
            self.play(Write(result))
            self.wait(1.5)

        self.play(FadeOut(title3), FadeOut(case2_title), FadeOut(user_vec), FadeOut(song_vec), FadeOut(dot_op), FadeOut(result))

        # Section 4: Key Interview Insight
        title4 = Text("Key Interview Insight", color=BLACK, font_size=48).to_edge(UP)
        self.play(Write(title4))

        with self.voiceover(text="Now, here is a key insight for your technical interviews. Interviewers will frequently test your understanding of the subtle difference between the Dot Product and Cosine Similarity. You must know that the dot product is highly sensitive to the magnitude, or length, of the vectors. In contrast, cosine similarity normalizes the vectors, meaning it strictly only looks at the angle between them, completely ignoring their length.") as tracker:

            insight_box = Rectangle(width=10, height=4, color=PURPLE, fill_opacity=0.1)
            insight_text = Text("Dot Product vs. Cosine Similarity", color=BLACK, font_size=36).next_to(insight_box, UP, buff=0.5)

            dot_eq = MathTex("\\text{Dot Product} = |A||B|\\cos(\\theta)", color=BLACK).shift(UP*0.5)
            cos_eq = MathTex("\\text{Cosine Sim.} = \\cos(\\theta) = \\frac{A \\cdot B}{|A||B|}", color=BLACK).shift(DOWN*0.5)

            self.play(Create(insight_box), Write(insight_text))
            self.play(Write(dot_eq))
            self.play(Write(cos_eq))
            self.wait(1.5)

        with self.voiceover(text="This distinction is critical. If the magnitude of your vectors represents something meaningful, like importance, frequency of a word, or confidence, you should definitely use the dot product. However, if you only care about the direction or the pure semantic meaning of the data, regardless of its length, you should use cosine similarity. Knowing precisely when to use which metric is a very common, and highly distinguishing, interview question that you must master.") as tracker:

            insight1 = Text("Magnitudes matter? -> Dot Product", color=RED, font_size=28).shift(DOWN*2)
            insight2 = Text("Only angle matters? -> Cosine Similarity", color=BLUE, font_size=28).shift(DOWN*2.5)

            self.play(Write(insight1))
            self.play(Write(insight2))
            self.wait(2)

        self.play(FadeOut(title4), FadeOut(insight_box), FadeOut(insight_text), FadeOut(dot_eq), FadeOut(cos_eq), FadeOut(insight1), FadeOut(insight2))

        # Outro
        with self.voiceover(text="And that's the dot product! Thanks for watching.") as tracker:
            outro_text = Text("Master the basics. Build the future.", color=BLACK, font_size=40)
            self.play(Write(outro_text))
            self.wait(2)
            self.play(FadeOut(outro_text))
