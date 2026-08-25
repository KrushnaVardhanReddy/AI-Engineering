from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class CosineSimilarity3D(VoiceoverScene):
    def construct(self):
        # Aesthetic: Whiteboard Style
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # =========================================================================
        # Section 1: What is it?
        # =========================================================================
        title = Text("Cosine Similarity in 3D", color=BLACK, font_size=48).to_edge(UP)
        self.play(Write(title))
        self.wait(2)

        with self.voiceover(text="Welcome to Day 3 of AI Engineering Mastery. Today, we are going to dive deep into a foundational concept for any vector search system: Cosine Similarity, specifically visualizing it in a three-dimensional space.") as tracker:
            self.wait(1.5)

        with self.voiceover(text="So, what exactly is cosine similarity? Put simply, it is a mathematical metric that measures the cosine of the angle between two multi-dimensional vectors, determining how similar their directions are, completely independent of their lengths or magnitudes.") as tracker:
            definition = Text(
                "Metric measuring the cosine of the angle\nbetween two vectors in a multi-dimensional space.",
                color=BLACK, font_size=32, t2c={"cosine of the angle": BLUE}
            ).next_to(title, DOWN, buff=0.5)
            self.play(FadeIn(definition, shift=UP))
            self.wait(2)

        with self.voiceover(text="Let's build a three-dimensional space to visualize this. Here we have our X, Y, and Z axes forming our embedding space.") as tracker:
            axes = ThreeDAxes(
                x_range=[-1, 5, 1],
                y_range=[-1, 5, 1],
                z_range=[-1, 5, 1],
                x_length=5,
                y_length=5,
                z_length=5,
                axis_config={"color": BLACK}
            ).scale(0.7).next_to(definition, DOWN, buff=0.5)

            self.play(Create(axes), run_time=2)
            self.wait(1.5)

        with self.voiceover(text="Now, let's plot a vector representing a concept, say, Vector A, pointing to coordinates 4, 2, and 3.") as tracker:
            vec_a_coords = np.array([4, 2, 3])
            vec_b_coords = np.array([2, 4, 1])

            vec_a = Arrow3D(axes.c2p(0, 0, 0), axes.c2p(*vec_a_coords), color=BLUE, thickness=0.02)
            label_a = MathTex(r"\vec{A}", color=BLUE).next_to(axes.c2p(*vec_a_coords), RIGHT)

            self.play(Create(vec_a), Write(label_a))
            self.wait(1.5)

        with self.voiceover(text="And here is Vector B, pointing to coordinates 2, 4, and 1. Think of these vectors as the mathematical representations of two sentences produced by an embedding model.") as tracker:
            vec_b = Arrow3D(axes.c2p(0, 0, 0), axes.c2p(*vec_b_coords), color=RED, thickness=0.02)
            label_b = MathTex(r"\vec{B}", color=RED).next_to(axes.c2p(*vec_b_coords), UP)

            self.play(Create(vec_b), Write(label_b))
            self.wait(1.5)

        with self.voiceover(text="Notice the angle forming between them at the origin. That angle, theta, is what cosine similarity measures. If the vectors point in the exact same direction, the angle is 0, and the cosine is 1, indicating perfect similarity.") as tracker:
            angle_arc = ArcBetweenPoints(
                axes.c2p(*(vec_a_coords * 0.4)),
                axes.c2p(*(vec_b_coords * 0.4)),
                color=PURPLE, stroke_width=4
            )
            theta_label = MathTex(r"\theta", color=PURPLE).next_to(angle_arc, OUT)
            self.play(Create(angle_arc), Write(theta_label))
            self.wait(2)

        with self.voiceover(text="Let's look at the mathematical derivation of this metric step by step. We start from the geometric definition of the dot product.") as tracker:
            self.play(
                FadeOut(axes), FadeOut(vec_a), FadeOut(vec_b),
                FadeOut(label_a), FadeOut(label_b), FadeOut(angle_arc), FadeOut(theta_label), FadeOut(definition)
            )

            eq1 = MathTex(r"\vec{A} \cdot \vec{B} = \|\vec{A}\| \|\vec{B}\| \cos(\theta)", color=BLACK).move_to(ORIGIN)
            self.play(Write(eq1))
            self.wait(2)

        with self.voiceover(text="To isolate the cosine of theta, we simply divide both sides by the product of the magnitudes of vector A and vector B.") as tracker:
            eq2 = MathTex(r"\cos(\theta)", r"=", r"\frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}", color=BLACK).move_to(ORIGIN)
            self.play(TransformMatchingTex(eq1, eq2))
            self.wait(2)

        with self.voiceover(text="And there we have it. The Cosine Similarity formula.") as tracker:
            eq3 = MathTex(r"\text{Cosine Similarity} = ", r"\cos(\theta)", r"=", r"\frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}", color=BLACK).move_to(ORIGIN)
            self.play(TransformMatchingTex(eq2, eq3))
            self.wait(3)

        self.play(FadeOut(eq3))

        # =========================================================================
        # Section 2: Why do we need it?
        # =========================================================================
        with self.voiceover(text="Now you might be asking: Why do we need this? Why can't we just use the traditional Euclidean distance, measuring the straight-line distance between the endpoints of the vectors?") as tracker:
            why_title = Text("Why do we need it?", color=PURPLE, font_size=40).next_to(title, DOWN, buff=0.5)
            self.play(Write(why_title))
            self.wait(2)

        with self.voiceover(text="Imagine comparing two documents by plotting the frequency of the word 'AI' on the X-axis and 'Machine Learning' on the Y-axis.") as tracker:
            axes2 = Axes(
                x_range=[0, 10, 1],
                y_range=[0, 10, 1],
                x_length=8,
                y_length=4,
                axis_config={"color": BLACK}
            ).next_to(why_title, DOWN, buff=0.5)
            x_label = axes2.get_x_axis_label("AI Freq", edge=RIGHT, direction=DOWN).set_color(BLACK)
            y_label = axes2.get_y_axis_label("ML Freq", edge=UP, direction=LEFT).set_color(BLACK)

            self.play(Create(axes2), Write(x_label), Write(y_label))
            self.wait(1.5)

        with self.voiceover(text="Here is a short article about AI. It uses the words just a few times.") as tracker:
            doc1_coords = np.array([2, 1, 0])
            doc1 = Arrow(axes2.c2p(0, 0), axes2.c2p(doc1_coords[0], doc1_coords[1]), buff=0, color=BLUE)
            doc1_label = Text("Short Article", color=BLUE, font_size=24).next_to(doc1.get_end(), RIGHT)
            self.play(GrowArrow(doc1), Write(doc1_label))
            self.wait(1.5)

        with self.voiceover(text="And here is a massive 500-page textbook on the exact same topic. It uses those words hundreds of times, so its vector stretches far out.") as tracker:
            doc2_coords = np.array([8, 4, 0])
            doc2 = Arrow(axes2.c2p(0, 0), axes2.c2p(doc2_coords[0], doc2_coords[1]), buff=0, color=RED)
            doc2_label = Text("Long Book", color=RED, font_size=24).next_to(doc2.get_end(), UP)
            self.play(GrowArrow(doc2), Write(doc2_label))
            self.wait(2)

        with self.voiceover(text="If we calculate the Euclidean distance between their endpoints, the distance is enormous. A system using Euclidean distance might conclude these documents are completely unrelated.") as tracker:
            euclidean_line = DashedLine(
                axes2.c2p(doc1_coords[0], doc1_coords[1]),
                axes2.c2p(doc2_coords[0], doc2_coords[1]),
                color=GREEN
            )
            euclidean_label = Text("Huge Euclidean Distance", color=GREEN, font_size=24).next_to(euclidean_line.get_center(), UP, buff=0.2)
            self.play(Create(euclidean_line), Write(euclidean_label))
            self.wait(2.5)

        with self.voiceover(text="However, because they discuss the exact same ratio of topics, their vectors point in the exact same direction. Cosine similarity focuses purely on this angle, completely ignoring the length.") as tracker:
            self.play(FadeOut(euclidean_line), FadeOut(euclidean_label))

            angle_zero_text = MathTex(r"\theta = 0 \Rightarrow \cos(0) = 1.0", color=PURPLE).to_edge(DOWN)
            self.play(Write(angle_zero_text))
            self.wait(2.5)

        with self.voiceover(text="This makes Cosine Similarity incredibly robust for Natural Language Processing, where document lengths vary wildly but semantic meaning is what truly matters.") as tracker:
            self.wait(2)

        self.play(
            FadeOut(why_title), FadeOut(axes2), FadeOut(x_label), FadeOut(y_label),
            FadeOut(doc1), FadeOut(doc1_label), FadeOut(doc2), FadeOut(doc2_label), FadeOut(angle_zero_text)
        )

        # =========================================================================
        # Section 3: Use Cases
        # =========================================================================
        with self.voiceover(text="So, where is this actually used in the industry? Let's look at two major real-world use cases.") as tracker:
            usecase_title = Text("Real-World Use Cases", color=GREEN, font_size=40).next_to(title, DOWN, buff=0.5)
            self.play(Write(usecase_title))
            self.wait(2)

        with self.voiceover(text="First, consider ChatGPT and OpenAI. When building Retrieval-Augmented Generation systems, they use Cosine Similarity to perform semantic search. It compares the embedding vector of your user query against millions of document chunk vectors in a database like Qdrant to retrieve the most contextually relevant information.") as tracker:
            chatgpt_usecase = VGroup(
                Text("1. ChatGPT (OpenAI)", color=BLACK, font_size=32, weight=BOLD),
                Text("Semantic Search & RAG (Retrieval-Augmented Generation)", color=BLUE, font_size=28)
            ).arrange(DOWN, aligned_edge=LEFT).next_to(usecase_title, DOWN, buff=1.0).to_edge(LEFT, buff=1.0)

            self.play(FadeIn(chatgpt_usecase, shift=RIGHT))
            self.wait(3)

        with self.voiceover(text="Second, consider Spotify's recommendation engine. They create a preference vector based on your listening history, and compare it against the feature vectors of millions of songs. Cosine similarity helps find the songs that align closest with your taste direction, regardless of how many total songs you listen to.") as tracker:
            spotify_usecase = VGroup(
                Text("2. Spotify", color=BLACK, font_size=32, weight=BOLD),
                Text("Recommender Systems (Matching user vectors to songs)", color=RED, font_size=28)
            ).arrange(DOWN, aligned_edge=LEFT).next_to(chatgpt_usecase, DOWN, buff=1.0).to_edge(LEFT, buff=1.0)

            self.play(FadeIn(spotify_usecase, shift=RIGHT))
            self.wait(3)

        self.play(FadeOut(usecase_title), FadeOut(chatgpt_usecase), FadeOut(spotify_usecase))

        # =========================================================================
        # Section 4: Key Interview Insight
        # =========================================================================
        with self.voiceover(text="Finally, here is the key interview insight you absolutely must know for machine learning system design interviews.") as tracker:
            insight_title = Text("Key Interview Insight", color=RED, font_size=40).next_to(title, DOWN, buff=0.5)
            self.play(Write(insight_title))
            self.wait(2)

        with self.voiceover(text="Interviewers will often ask you about the computational trade-off between Cosine Similarity and Dot Product. The gotcha is this:") as tracker:
            box = Rectangle(width=12, height=5, color=RED, fill_color=WHITE, fill_opacity=1.0).next_to(insight_title, DOWN, buff=0.5)

            insight_text_1 = Text("Gotcha: Cosine Similarity vs. Dot Product", color=BLACK, font_size=32, weight=BOLD)

            insight_group = VGroup(insight_text_1).arrange(DOWN, buff=0.4).move_to(box.get_center()).shift(UP*1.5)

            self.play(Create(box))
            self.play(Write(insight_text_1))
            self.wait(2.5)

        with self.voiceover(text="Calculating the magnitudes for the denominator of the Cosine Similarity formula is computationally expensive. However, if you pre-normalize your vectors so that their magnitude is exactly 1...") as tracker:
            insight_text_2 = MathTex(r"\text{If } \|\vec{A}\| = 1 \text{ and } \|\vec{B}\| = 1:", color=BLACK).next_to(insight_text_1, DOWN, buff=0.5)
            self.play(Write(insight_text_2))
            self.wait(2.5)

        with self.voiceover(text="Then the denominator becomes 1 times 1. This means the Cosine Similarity is mathematically identical to the Dot Product.") as tracker:
            insight_text_3 = MathTex(r"\text{Cosine Similarity} = \frac{\vec{A} \cdot \vec{B}}{1 \cdot 1} = \vec{A} \cdot \vec{B}", color=BLUE).next_to(insight_text_2, DOWN, buff=0.5)
            self.play(Write(insight_text_3))
            self.wait(2.5)

        with self.voiceover(text="Therefore, in production vector databases, always L2-normalize your vectors at ingestion time, and use the much faster Dot Product metric for your search queries to massively speed up retrieval times and save compute.") as tracker:
            insight_text_4 = Text("Use Dot Product for faster computation!", color=GREEN, font_size=28).next_to(insight_text_3, DOWN, buff=0.5)
            self.play(Write(insight_text_4))
            self.play(Indicate(insight_text_4, color=RED, scale_factor=1.1))
            self.wait(3.5)

        with self.voiceover(text="That concludes Day 3. You now deeply understand Cosine Similarity. See you in the next module.") as tracker:
            self.wait(2)

        self.play(FadeOut(Group(*self.mobjects)))
