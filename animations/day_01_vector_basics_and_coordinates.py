from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class VectorBasicsAndCoordinates(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # ==========================================
        # SECTION 1: What is it?
        # ==========================================
        title_1 = Text("What is a Vector?", color=BLACK).scale(1.2).to_edge(UP)

        with self.voiceover(text="Welcome to day one of our AI engineering interview prep. Today we're covering vector basics and coordinates.") as tracker:
            self.play(Write(title_1))

        definition = Text("A vector is an array of numbers that represents\na point in space or a specific feature set.", color=BLACK).scale(0.7)

        with self.voiceover(text="At its core, a vector is simply an array of numbers. In AI, these numbers represent a point in a high-dimensional space or a specific set of features.") as tracker:
            self.play(FadeIn(definition))

        self.wait(1.5)
        self.play(FadeOut(definition))

        # Show axes and a vector
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": BLACK, "include_numbers": True},
        ).scale(0.8)

        # Change numbers color
        for num in axes.x_axis.numbers:
            num.set_color(BLACK)
        for num in axes.y_axis.numbers:
            num.set_color(BLACK)

        vector_arrow = Arrow(axes.c2p(0, 0), axes.c2p(3, 4), buff=0, color=BLUE)
        point = Dot(axes.c2p(3, 4), color=BLUE)
        coords = MathTex(r"\begin{bmatrix} 3 \\ 4 \end{bmatrix}", color=BLACK).next_to(point, RIGHT)

        with self.voiceover(text="Visually, we can think of a vector as an arrow pointing from the origin to a specific coordinate, like x equals 3, and y equals 4.") as tracker:
            self.play(Create(axes))
            self.play(GrowArrow(vector_arrow))
            self.play(FadeIn(point), Write(coords))

        self.wait(1.5)
        self.play(FadeOut(axes), FadeOut(vector_arrow), FadeOut(point), FadeOut(coords), FadeOut(title_1))

        # ==========================================
        # SECTION 2: Why do we need it?
        # ==========================================
        title_2 = Text("Why do we need vectors?", color=BLACK).scale(1.2).to_edge(UP)

        with self.voiceover(text="Why do we need vectors in machine learning? Let's look at the problem they solve.") as tracker:
            self.play(Write(title_2))

        # Before: Unstructured Data
        before_text = Text("Without Vectors: Raw Text", color=BLACK).scale(0.8).shift(UP*1.5)
        raw_text = Text('"I love this movie!"', color=RED).scale(0.8)

        with self.voiceover(text="Imagine we have raw, unstructured text like the sentence, 'I love this movie!' Computers cannot natively understand or compare strings of text easily.") as tracker:
            self.play(FadeIn(before_text))
            self.play(Write(raw_text))

        self.wait(1.5)

        # After: Vector Representation
        after_text = Text("With Vectors: Computable Data", color=BLACK).scale(0.8).shift(UP*1.5)
        vector_math = MathTex(r"v = [0.12, -0.45, 0.89, \dots]", color=BLUE).scale(1.2)

        with self.voiceover(text="By converting this text into a vector using an embedding model, we transform it into computable data. A list of floating-point numbers.") as tracker:
            self.play(Transform(before_text, after_text), Transform(raw_text, vector_math))

        with self.voiceover(text="Now, we can perform math on these numbers to measure semantic similarity or train neural networks.") as tracker:
            self.wait(1.5)

        self.play(FadeOut(before_text), FadeOut(raw_text), FadeOut(after_text), FadeOut(vector_math), FadeOut(title_2))


        # ==========================================
        # SECTION 3: Use Cases
        # ==========================================
        title_3 = Text("Real-World Use Cases", color=BLACK).scale(1.2).to_edge(UP)

        with self.voiceover(text="Where are vectors actually used in production?") as tracker:
            self.play(Write(title_3))

        # Case 1
        case1_title = Text("1. OpenAI (ChatGPT)", color=BLUE).scale(0.9).shift(UP * 1 + LEFT * 2)
        case1_desc = Text("Converts prompts into dense vectors\nto understand context and meaning.", color=BLACK).scale(0.6).next_to(case1_title, DOWN, aligned_edge=LEFT)

        with self.voiceover(text="First, OpenAI uses vectors in ChatGPT. Every prompt you type is converted into dense vectors so the model can understand context and meaning.") as tracker:
            self.play(FadeIn(case1_title))
            self.play(Write(case1_desc))

        self.wait(1.5)

        # Case 2
        case2_title = Text("2. Spotify", color=GREEN).scale(0.9).shift(DOWN * 1 + LEFT * 2)
        case2_desc = Text("Maps users and songs into vector space\nfor personalized recommendations.", color=BLACK).scale(0.6).next_to(case2_title, DOWN, aligned_edge=LEFT)

        with self.voiceover(text="Second, Spotify uses vector embeddings. They map both users and songs into a shared vector space to calculate distance and generate personalized recommendations.") as tracker:
            self.play(FadeIn(case2_title))
            self.play(Write(case2_desc))

        self.wait(1.5)
        self.play(FadeOut(title_3), FadeOut(case1_title), FadeOut(case1_desc), FadeOut(case2_title), FadeOut(case2_desc))


        # ==========================================
        # SECTION 4: Key Interview Insight
        # ==========================================
        title_4 = Text("Key Interview Insight", color=BLACK).scale(1.2).to_edge(UP)

        with self.voiceover(text="Let's look at the most common gotcha that interviewers will test you on regarding vectors.") as tracker:
            self.play(Write(title_4))

        # Callout box
        box = Rectangle(width=10, height=4, color=RED, fill_opacity=0.1)
        box.set_stroke(width=4)

        insight_title = Text("Curse of Dimensionality", color=RED).scale(1.0).next_to(box.get_top(), DOWN, buff=0.3)
        insight_desc = Text(
            "As vector dimensions increase (e.g. 1536d):\n"
            "• Computational cost scales linearly or quadratically.\n"
            "• Distances between points become less meaningful.",
            color=BLACK,
            t2c={"1536d": BLUE, "cost scales": PURPLE, "less meaningful": PURPLE}
        ).scale(0.7).next_to(insight_title, DOWN, buff=0.5)

        with self.voiceover(text="The biggest tradeoff is the Curse of Dimensionality. Modern embedding models like OpenAI's output vectors with over a thousand dimensions.") as tracker:
            self.play(Create(box))
            self.play(FadeIn(insight_title))

        with self.voiceover(text="As dimensions increase, the computational cost to calculate distances scales up massively. More importantly, in extremely high dimensions, the distances between any two random points start to look the same, making simple distance metrics less meaningful.") as tracker:
            self.play(Write(insight_desc))

        self.wait(1.5)

        with self.voiceover(text="Interviewers want to see that you understand the tradeoff between having enough dimensions to capture complex semantic meaning, and having too many dimensions that cause performance issues and distance degradation.") as tracker:
            self.wait(1.5)

        with self.voiceover(text="That wraps up our day one breakdown of vector basics. Happy studying!") as tracker:
            self.play(FadeOut(insight_title), FadeOut(insight_desc), FadeOut(box), FadeOut(title_4))
            self.wait(1)
