from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import math
import numpy as np

class VectorDatabaseExplained(VoiceoverScene):
    def construct(self):
        # Setup audio service and background
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # We will structure the script to hit the 5-7 minute target length.
        # This means long pauses between ideas and comprehensive spoken explanations.
        # Using slow animations and breaking down concepts visually.

        self.section_what_is_it()
        self.section_why_do_we_need_it()
        self.section_use_cases()
        self.section_interview_insight()

    def section_what_is_it(self):
        # --- WHAT IS IT? ---
        title = Text("What is a Vector Database?", color=BLACK, font_size=48, weight=BOLD)

        with self.voiceover(text="Welcome back. Today we are tackling a critical component of modern AI architecture: the Vector Database. But what exactly is a Vector Database?") as tracker:
            self.play(Write(title))
        self.wait(1.5)

        with self.voiceover(text="In simple terms, a Vector Database is a specialized storage system designed specifically to store, index, and query high-dimensional vector embeddings.") as tracker:
            self.play(title.animate.to_edge(UP))
            definition = Text("A specialized storage system for high-dimensional vector embeddings.", color=BLACK, font_size=32).next_to(title, DOWN, buff=0.5)
            self.play(FadeIn(definition))
        self.wait(1.5)

        with self.voiceover(text="To really understand this, we need to visualize what an embedding is. An embedding represents complex data, like text or an image, as an array of numbers, or coordinates in space.") as tracker:
            self.play(FadeOut(definition))

            # Show a sentence converting to a vector
            sentence = Text('"The cat sat on the mat"', color=BLUE, font_size=36)
            self.play(FadeIn(sentence))

        self.wait(1.5)

        with self.voiceover(text="A machine learning model, such as an LLM, processes this sentence and outputs a dense vector. This vector is just a long list of floating-point numbers.") as tracker:
            arrow = Arrow(start=UP, end=DOWN, color=BLACK).next_to(sentence, DOWN)
            vector_text = MathTex(
                r"\begin{bmatrix} 0.12 \\ -0.84 \\ 0.55 \\ \vdots \\ 0.09 \end{bmatrix}",
                color=BLACK
            ).next_to(arrow, DOWN)

            self.play(GrowArrow(arrow))
            self.play(Write(vector_text))

        self.wait(1.5)

        with self.voiceover(text="Now imagine millions or billions of these vectors. Traditional relational databases, like PostgreSQL, organize data in rows and columns. They are not built to efficiently search through billions of high-dimensional lists of numbers.") as tracker:
            self.play(FadeOut(sentence), FadeOut(arrow), FadeOut(vector_text))

            # Show a database symbol
            db_cylinder = Cylinder(radius=1.5, height=2, color=BLACK, fill_color=BLUE, fill_opacity=0.1)
            db_cylinder.rotate(PI/2, axis=RIGHT)
            db_cylinder.rotate(PI/8, axis=UP)

            db_label = Text("Vector Database", color=BLACK, font_size=36).next_to(db_cylinder, DOWN)

            self.play(Create(db_cylinder), Write(db_label))

        self.wait(1.5)

        with self.voiceover(text="A vector database is purpose-built to organize these numerical vectors in a way that allows us to find similar items extremely fast, using spatial proximity rather than exact keyword matches.") as tracker:
            # Animate some points going into the DB
            points = VGroup(*[Dot(color=RED, radius=0.1) for _ in range(5)])
            points.arrange(RIGHT, buff=0.5).next_to(db_cylinder, UP, buff=1)

            self.play(FadeIn(points))
            self.play(
                *[point.animate.move_to(db_cylinder.get_center() + np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0])) for point in points]
            )
            self.play(FadeOut(points))

        self.wait(1.5)
        self.play(FadeOut(db_cylinder), FadeOut(db_label), FadeOut(title))

    def section_why_do_we_need_it(self):
        # --- WHY DO WE NEED IT? ---
        title = Text("Why do we need it?", color=BLACK, font_size=48, weight=BOLD).to_edge(UP)

        with self.voiceover(text="So, why do we need a dedicated Vector Database? Why can't we just use what we already have?") as tracker:
            self.play(Write(title))
        self.wait(1.5)

        with self.voiceover(text="Let's look at a 'before and after' scenario. Before vector databases, search was heavily reliant on lexical or keyword search.") as tracker:
            subtitle_before = Text("Before: Keyword Search (Lexical)", color=RED, font_size=36).next_to(title, DOWN, buff=0.5)
            self.play(FadeIn(subtitle_before))

        self.wait(1.5)

        with self.voiceover(text="Imagine you search a database for the query 'fast car'. A traditional database looks for exact matches of the words 'fast' and 'car'.") as tracker:
            query_box = Rectangle(width=4, height=1, color=BLACK).move_to(LEFT * 3)
            query_text = Text('"fast car"', color=BLACK, font_size=32).move_to(query_box.get_center())
            self.play(Create(query_box), Write(query_text))

        self.wait(1.5)

        with self.voiceover(text="If your database contains a document that says 'quick automobile', the traditional database will not return a match, because the exact keywords 'fast' and 'car' are missing. It has zero understanding of the semantic meaning.") as tracker:
            doc_box = Rectangle(width=4, height=1, color=BLACK).move_to(RIGHT * 3)
            doc_text = Text('"quick automobile"', color=BLACK, font_size=32).move_to(doc_box.get_center())

            cross = Cross(doc_box, stroke_color=RED)

            self.play(Create(doc_box), Write(doc_text))
            self.play(Create(cross))

        self.wait(1.5)

        with self.voiceover(text="Now, let's look at the 'after' scenario: Semantic Search using a Vector Database. Here, we don't care about the exact words. We care about the underlying meaning.") as tracker:
            self.play(
                FadeOut(subtitle_before),
                FadeOut(query_box), FadeOut(query_text),
                FadeOut(doc_box), FadeOut(doc_text), FadeOut(cross)
            )

            subtitle_after = Text("After: Semantic Search (Vector DB)", color=GREEN, font_size=36).next_to(title, DOWN, buff=0.5)
            self.play(FadeIn(subtitle_after))

        self.wait(1.5)

        with self.voiceover(text="Both the search query 'fast car' and the document 'quick automobile' are converted into vectors by an embedding model. Because their meanings are so similar, their resulting vectors are placed very close together in the high-dimensional space.") as tracker:
            # Draw an axes system
            axes = Axes(
                x_range=[0, 10, 1],
                y_range=[0, 10, 1],
                x_length=6,
                y_length=4,
                axis_config={"color": BLACK},
            ).shift(DOWN * 0.5)

            self.play(Create(axes))

            dot1 = Dot(axes.c2p(7, 8), color=BLUE, radius=0.1)
            label1 = Text('"fast car"', color=BLUE, font_size=24).next_to(dot1, UP)

            dot2 = Dot(axes.c2p(7.5, 7.5), color=GREEN, radius=0.1)
            label2 = Text('"quick automobile"', color=GREEN, font_size=24).next_to(dot2, RIGHT)

            self.play(FadeIn(dot1), Write(label1))
            self.play(FadeIn(dot2), Write(label2))

        self.wait(1.5)

        with self.voiceover(text="When we query the Vector Database, it uses mathematical distance metrics, like Cosine Similarity or Euclidean distance, to find the nearest neighbors to the query vector. It mathematically calculates that 'quick automobile' is the closest match to 'fast car', and returns it as a highly relevant result.") as tracker:
            # Draw a dashed line between them
            distance_line = DashedLine(dot1.get_center(), dot2.get_center(), color=RED)
            self.play(Create(distance_line))

            # Show formula appearing step by step using TransformMatchingTex
            math_formula_part1 = MathTex(
                r"\text{Cosine Similarity} =",
                color=BLACK
            ).scale(0.8).next_to(axes, DOWN).shift(LEFT * 1.5)

            math_formula_part2 = MathTex(
                r"\text{Cosine Similarity} =", r"\frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}",
                color=BLACK
            ).scale(0.8).next_to(axes, DOWN)

            self.play(Write(math_formula_part1))
            self.wait(0.5)
            self.play(TransformMatchingTex(math_formula_part1, math_formula_part2))

        self.wait(2.0)

        self.play(
            FadeOut(title), FadeOut(subtitle_after),
            FadeOut(axes), FadeOut(dot1), FadeOut(label1),
            FadeOut(dot2), FadeOut(label2), FadeOut(distance_line),
            FadeOut(math_formula_part2)
        )

    def section_use_cases(self):
        # --- USE CASES ---
        title = Text("Real-World Use Cases", color=BLACK, font_size=48, weight=BOLD).to_edge(UP)

        with self.voiceover(text="Now that we know how it works, let's explore how major tech companies actually use Vector Databases in production today.") as tracker:
            self.play(Write(title))
        self.wait(1.5)

        with self.voiceover(text="Our first use case is ChatGPT, created by OpenAI. Specifically, Vector Databases are used heavily for Retrieval-Augmented Generation, commonly known as RAG.") as tracker:
            case1_title = Text("1. OpenAI (ChatGPT) - RAG", color=BLUE, font_size=36).next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=1)
            self.play(FadeIn(case1_title))

        self.wait(1.5)

        with self.voiceover(text="When you upload a massive PDF document to ChatGPT, the model cannot read the entire thing at once due to context window limits. Instead, the document is broken into chunks, embedded, and stored in a Vector Database.") as tracker:
            doc_icon = Rectangle(width=1, height=1.5, color=BLACK).next_to(case1_title, DOWN, buff=0.5).align_to(case1_title, LEFT)
            lines = VGroup(*[Line(LEFT*0.3, RIGHT*0.3, color=BLACK) for _ in range(4)]).arrange(DOWN, buff=0.2).move_to(doc_icon.get_center())
            doc_group = VGroup(doc_icon, lines)
            self.play(Create(doc_group))

            # Show chunking
            chunks = VGroup(*[Rectangle(width=0.8, height=0.3, color=RED) for _ in range(3)]).arrange(DOWN, buff=0.1).next_to(doc_icon, RIGHT, buff=1)
            arrow1 = Arrow(start=doc_icon.get_right(), end=chunks.get_left(), color=BLACK)
            self.play(GrowArrow(arrow1), Create(chunks))

        self.wait(1.5)

        with self.voiceover(text="When you ask a question, your question is also embedded. The Vector Database finds the most relevant chunks of text to provide the LLM with exactly the context it needs to answer accurately. This powers enterprise search and custom chatbots worldwide.") as tracker:
            db_box = Square(side_length=1.5, color=PURPLE).next_to(chunks, RIGHT, buff=1)
            db_text = Text("Vector\nDB", color=PURPLE, font_size=20).move_to(db_box.get_center())
            db_group = VGroup(db_box, db_text)

            arrow2 = Arrow(start=chunks.get_right(), end=db_box.get_left(), color=BLACK)
            self.play(GrowArrow(arrow2), Create(db_group))

        self.wait(2.0)

        with self.voiceover(text="Our second major use case is recommendation systems, such as the one used by Spotify to recommend music and podcasts.") as tracker:
            case2_title = Text("2. Spotify - Recommendation Engines", color=GREEN, font_size=36).next_to(db_group, DOWN, buff=1).to_edge(LEFT, buff=1)
            self.play(FadeIn(case2_title))

        self.wait(1.5)

        with self.voiceover(text="Spotify creates embeddings not just for songs, based on audio features and metadata, but also for users, based on their listening history. Both users and songs exist as vectors in the same multi-dimensional space.") as tracker:
            user_icon = Circle(radius=0.3, color=BLACK).next_to(case2_title, DOWN, buff=0.5).align_to(case2_title, LEFT)
            user_label = Text("User Vector", color=BLACK, font_size=24).next_to(user_icon, RIGHT)

            song_icon = Square(side_length=0.6, color=BLACK).next_to(user_icon, DOWN, buff=0.5)
            song_label = Text("Song Vector", color=BLACK, font_size=24).next_to(song_icon, RIGHT)

            self.play(Create(user_icon), Write(user_label))
            self.play(Create(song_icon), Write(song_label))

        self.wait(1.5)

        with self.voiceover(text="To suggest a new track for you to listen to, Spotify simply queries the Vector Database for the song vectors that are mathematically closest to your personal user vector. This powers highly personalized, real-time recommendations at scale.") as tracker:
            # show a quick proximity visual
            box = Rectangle(width=4, height=2, color=BLACK).next_to(song_label, RIGHT, buff=1)
            u_dot = Dot(box.get_center() + LEFT*1 + UP*0.5, color=BLUE)
            s_dot = Dot(box.get_center() + LEFT*0.5 + UP*0.2, color=GREEN)
            s_dot2 = Dot(box.get_center() + RIGHT*1 + DOWN*0.5, color=RED)

            self.play(Create(box))
            self.play(FadeIn(u_dot), FadeIn(s_dot), FadeIn(s_dot2))

            match_circle = Circle(radius=0.6, color=GREEN).move_to(u_dot.get_center() + RIGHT*0.25 + DOWN*0.15)
            self.play(Create(match_circle))

        self.wait(2.0)
        self.play(
            *[FadeOut(m) for m in self.mobjects]
        )

    def section_interview_insight(self):
        # --- KEY INTERVIEW INSIGHT ---
        title = Text("Key Interview Insight", color=BLACK, font_size=48, weight=BOLD).to_edge(UP)

        with self.voiceover(text="Finally, let's discuss a key interview insight. If you are interviewing for an AI or Machine Learning Engineering role, you will almost certainly be asked about the tradeoffs of using a Vector Database.") as tracker:
            self.play(Write(title))

        self.wait(1.5)

        with self.voiceover(text="The most common 'gotcha' question interviewers ask is: 'Why not just compute the similarity between the query and every single item in the database? Why do we need a specialized index?'") as tracker:
            question = Text("Q: Why not compute similarity against ALL items?", color=RED, font_size=32).next_to(title, DOWN, buff=1)
            self.play(FadeIn(question))

        self.wait(1.5)

        with self.voiceover(text="The answer comes down to algorithm efficiency. Computing exact nearest neighbors for millions of vectors requires an exhaustive search, known as K-Nearest Neighbors, or K-N-N. The time complexity scales linearly, O of N, which is far too slow for real-time applications.") as tracker:
            knn_text = Text("Exact Search (k-NN) = Exhaustive, Slow O(N)", color=BLACK, font_size=28).next_to(question, DOWN, buff=0.5)
            self.play(Write(knn_text))

        self.wait(1.5)

        with self.voiceover(text="To solve this, Vector Databases implement A-N-N, or Approximate Nearest Neighbors. Algorithms like H-N-S-W (Hierarchical Navigable Small World) sacrifice a tiny amount of accuracy in exchange for massive speed gains.") as tracker:
            callout_box = Rectangle(width=10, height=3, color=BLUE, fill_color=BLUE, fill_opacity=0.1).next_to(knn_text, DOWN, buff=1)

            insight_title = Text("The Tradeoff: A-N-N", color=BLUE, font_size=36, weight=BOLD).move_to(callout_box.get_top() + DOWN*0.5)
            insight_desc = Text("We trade perfect accuracy for immense speed\nusing Approximate Nearest Neighbors (e.g., HNSW).", color=BLACK, font_size=28, text_align="center").next_to(insight_title, DOWN, buff=0.5)

            self.play(Create(callout_box))
            self.play(Write(insight_title))
            self.play(FadeIn(insight_desc))

        self.wait(1.5)

        with self.voiceover(text="By navigating a graph structure, HNSW achieves sub-linear search time. So remember the tradeoff: Vector Databases trade perfect accuracy for real-time speed. Mentioning this tradeoff explicitly will show senior-level understanding during your system design interviews.") as tracker:
            tradeoff_eq = MathTex(
                r"\text{Accuracy} \downarrow \quad \text{Speed} \uparrow\uparrow",
                color=BLACK
            ).next_to(insight_desc, DOWN, buff=0.5)

            self.play(Write(tradeoff_eq))

        self.wait(3.0)

        with self.voiceover(text="That wraps up our deep dive into Vector Databases. See you in the next lesson!") as tracker:
            self.play(*[FadeOut(m) for m in self.mobjects])

        self.wait(1.0)
