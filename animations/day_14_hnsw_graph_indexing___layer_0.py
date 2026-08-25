from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import random
import numpy as np

class HNSWLayer0(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # --- Title ---
        title = Tex("HNSW Graph Indexing: Layer 0", color=BLACK, font_size=48)
        title.to_edge(UP)

        with self.voiceover(text="Welcome to Day 14 of the A I Engineering Mastery series. Today, we are exploring one of the most important concepts in vector databases and semantic search: HNSW Graph Indexing, focusing specifically on Layer 0. If you have ever wondered how modern A I applications can search through millions, or even billions, of high-dimensional vectors in mere milliseconds, the answer almost always involves HNSW, which stands for Hierarchical Navigable Small World. Let us dive into the details.") as tracker:
            self.play(Write(title))

        self.wait(1.5)

        # --- Section 1: What is it? ---
        def_title = Tex("What is it?", color=BLUE, font_size=40).next_to(title, DOWN, buff=0.5)
        definition = Tex(
            "Layer 0 is the foundational base layer of the HNSW index\\\\",
            "where every single data point is stored in a densely connected graph.",
            color=BLACK, font_size=32
        ).next_to(def_title, DOWN, buff=0.5)

        with self.voiceover(text="What exactly is HNSW, and more specifically, what is Layer 0? HNSW is a state-of-the-art algorithm used for approximate nearest neighbor search. Layer 0 is the foundational base layer of the HNSW index structure. It is the only layer where every single data point in our entire dataset is actually stored.") as tracker:
            self.play(FadeIn(def_title, shift=UP))
            self.play(Write(definition))

        self.wait(1.5)

        # Animate a densely connected graph
        random.seed(42)
        # Create nodes
        dots = VGroup(*[Dot(point=[random.uniform(-4, 4), random.uniform(-2.5, 0.5), 0], color=BLUE, radius=0.1) for _ in range(25)])

        edges = VGroup()
        for i in range(len(dots)):
            distances = [(j, np.linalg.norm(dots[i].get_center() - dots[j].get_center())) for j in range(len(dots)) if i != j]
            distances.sort(key=lambda x: x[1])
            for j, dist in distances[:3]:
                if i < j: # Avoid double drawing
                    line = Line(dots[i].get_center(), dots[j].get_center(), color=BLACK, stroke_width=1.5, stroke_opacity=0.6)
                    edges.add(line)

        graph_group = VGroup(edges, dots).shift(DOWN * 0.5)

        with self.voiceover(text="Visually, you can think of Layer 0 as a densely connected, sprawling graph on a two-dimensional plane. Each node or point in this graph represents a vector—such as a text embedding or an image embedding. And the edges connecting these nodes represent proximity; each node is connected to its closest neighbors in the high-dimensional space. Layer 0 contains the absolute ground truth of our dataset's topology. While higher layers act as fast highways for search, Layer 0 is the street-level map where the final, precise search happens.") as tracker:
            self.play(FadeIn(dots, lag_ratio=0.1), run_time=2)
            self.play(Create(edges, lag_ratio=0.1), run_time=4)

        self.wait(1.5)
        self.play(FadeOut(def_title), FadeOut(definition), FadeOut(graph_group))


        # --- Section 2: Why do we need it? ---
        why_title = Tex("Why do we need it?", color=RED, font_size=40).next_to(title, DOWN, buff=0.5)

        # Before: Flat Index (exhaustive search)
        before_text = Tex("Before: Flat Index (Brute-Force Search)", color=BLACK, font_size=32).next_to(why_title, DOWN, buff=0.5)

        flat_dots = VGroup(*[Dot(point=[random.uniform(-5, -0.5), random.uniform(-2, 1), 0], color=BLACK, radius=0.1) for _ in range(15)])
        query_dot = Dot(point=[-3, -0.5, 0], color=RED, radius=0.12)
        query_label = Tex("Query", color=RED, font_size=24).next_to(query_dot, UP, buff=0.1)

        flat_search_lines = VGroup(*[Line(query_dot.get_center(), d.get_center(), color=RED, stroke_width=2, stroke_opacity=0.5) for d in flat_dots])

        with self.voiceover(text="To truly appreciate HNSW, we need to ask: why do we need it? Let us contrast it with the naive approach: the Flat Index. Imagine you have a massive database of embeddings. When a new query comes in, a Flat Index performs a brute-force, exhaustive search. This means it calculates the exact distance—whether that is Euclidean distance or Cosine Similarity—between the query vector and every single vector in the database.") as tracker:
            self.play(FadeIn(why_title, shift=UP))
            self.play(Write(before_text))
            self.play(FadeIn(flat_dots), FadeIn(query_dot), FadeIn(query_label))
            self.play(Create(flat_search_lines, lag_ratio=0.1), run_time=3)

        self.wait(1.5)

        eq_flat = MathTex("\\text{Time} = ", "O(", "N", "\\times D)").set_color(BLACK).next_to(before_text, DOWN, buff=0.5).shift(RIGHT * 3)

        with self.voiceover(text="Let us look at the time complexity of this brute-force approach. The time taken is proportional to Big O of N times D, where N is the number of vectors, and D is the dimensionality of each vector. If you have a hundred million vectors of fifteen hundred dimensions, this calculation becomes incredibly slow. It scales terribly in production.") as tracker:
            self.play(Write(eq_flat))

        self.wait(1.5)

        # After: HNSW Graph Traversal
        after_text = Tex("After: HNSW Graph Traversal", color=BLACK, font_size=32).next_to(why_title, DOWN, buff=0.5)

        hnsw_dots = VGroup(*[Dot(point=[random.uniform(0.5, 5), random.uniform(-2, 1), 0], color=BLUE, radius=0.1) for _ in range(15)])
        hnsw_edges = VGroup()
        for i in range(len(hnsw_dots)):
            distances = [(j, np.linalg.norm(hnsw_dots[i].get_center() - hnsw_dots[j].get_center())) for j in range(len(hnsw_dots)) if i != j]
            distances.sort(key=lambda x: x[1])
            for j, dist in distances[:2]:
                if i < j:
                    line = Line(hnsw_dots[i].get_center(), hnsw_dots[j].get_center(), color=BLACK, stroke_width=1.5, stroke_opacity=0.6)
                    hnsw_edges.add(line)

        hnsw_query_dot = Dot(point=[3, -0.5, 0], color=GREEN, radius=0.12)
        hnsw_query_label = Tex("Query", color=GREEN, font_size=24).next_to(hnsw_query_dot, UP, buff=0.1)

        # entry point
        entry_dot = hnsw_dots[0]
        entry_label = Tex("Entry Point", color=BLACK, font_size=24).next_to(entry_dot, UP, buff=0.1)

        # Path simulation
        path_nodes = [hnsw_dots[0], hnsw_dots[4], hnsw_dots[7]]  # arbitrary path
        path_lines = VGroup()
        for i in range(len(path_nodes)-1):
            path_lines.add(Line(path_nodes[i].get_center(), path_nodes[i+1].get_center(), color=GREEN, stroke_width=3))
        path_lines.add(Line(path_nodes[-1].get_center(), hnsw_query_dot.get_center(), color=GREEN, stroke_width=3))

        with self.voiceover(text="Now, let us observe what happens when we use HNSW Graph Traversal. Instead of comparing the query to every point, we navigate our pre-built graph. We start at a designated entry point and compute the distance only to its immediate neighbors. We then greedily move to the neighbor that is closest to our query. We repeat this process, hopping from node to node, rapidly zeroing in on the target.") as tracker:
            self.play(FadeOut(before_text), FadeOut(flat_search_lines), FadeOut(flat_dots), FadeOut(query_dot), FadeOut(query_label))
            self.play(Write(after_text))
            self.play(FadeIn(hnsw_dots), Create(hnsw_edges), FadeIn(hnsw_query_dot), FadeIn(hnsw_query_label))
            self.play(FadeIn(entry_dot), FadeIn(entry_label))
            self.play(Create(path_lines, lag_ratio=1), run_time=3)

        self.wait(1.5)

        eq_hnsw = MathTex("\\text{Time} = ", "O(", "\\log(N)", "\\times D)").set_color(BLACK).move_to(eq_flat.get_center())

        with self.voiceover(text="By relying on the small world property of the graph, our search path is extremely short. Let us transform our time complexity equation. The linear N becomes logarithmic. The time complexity drops drastically to Big O of log N times D. This leap from linear to logarithmic time is what makes searches blazingly fast, reducing query times from seconds to mere milliseconds.") as tracker:
            self.play(TransformMatchingTex(eq_flat, eq_hnsw))

        self.wait(1.5)
        self.play(FadeOut(why_title), FadeOut(after_text), FadeOut(hnsw_dots), FadeOut(hnsw_edges), FadeOut(hnsw_query_dot), FadeOut(hnsw_query_label), FadeOut(entry_label), FadeOut(path_lines), FadeOut(eq_hnsw))


        # --- Section 3: Use Cases ---
        use_cases_title = Tex("Real-World Use Cases", color=GREEN, font_size=40).next_to(title, DOWN, buff=0.5)

        uc1 = Tex("1. ", "Spotify:", " Song recommendations via audio embeddings", color=BLACK, font_size=32)
        uc1[1].set_color(GREEN)
        uc1.next_to(use_cases_title, DOWN, buff=1.0).align_to(title, LEFT).shift(RIGHT * 2)

        uc2 = Tex("2. ", "ChatGPT:", " Retrieving relevant context from memory", color=BLACK, font_size=32)
        uc2[1].set_color(BLUE)
        uc2.next_to(uc1, DOWN, buff=0.5).align_to(uc1, LEFT)

        with self.voiceover(text="Because of its unparalleled efficiency, HNSW powers some of the most massive real-world systems you use every day. Let us explore two prominent use cases.") as tracker:
            self.play(FadeIn(use_cases_title, shift=UP))

        with self.voiceover(text="First, consider Spotify. When you listen to a song, Spotify represents that audio track as a dense vector embedding. To generate your custom, real-time recommendations, Spotify uses an HNSW-based system to rapidly search through millions of tracks, finding songs with similar audio and metadata embeddings almost instantly.") as tracker:
            self.play(Write(uc1))

        self.wait(1.5)

        with self.voiceover(text="Second, think about Large Language Models like ChatGPT. When dealing with long, complex conversations or Retrieval-Augmented Generation architectures, the application needs to retrieve relevant context from memory. ChatGPT and similar systems use vector databases powered by HNSW under the hood. When you ask a question, your query is embedded, and the system quickly traverses the HNSW graph to find the most semantically relevant past messages or documents, maintaining accurate context seamlessly.") as tracker:
            self.play(Write(uc2))

        self.wait(1.5)
        self.play(FadeOut(use_cases_title), FadeOut(uc1), FadeOut(uc2))


        # --- Section 4: Key Interview Insight ---
        insight_title = Tex("Key Interview Insight", color=PURPLE, font_size=40).next_to(title, DOWN, buff=0.5)

        box = Rectangle(width=11, height=3, color=PURPLE, fill_color=PURPLE, fill_opacity=0.1)
        box.next_to(insight_title, DOWN, buff=0.5)

        insight_text = Tex(
            "Tradeoff: Search Speed vs. Memory Consumption\\\\",
            "Storing all graph edges for millions of vectors requires\\\\",
            "massive RAM compared to a simple Flat Index.",
            color=BLACK, font_size=32
        )
        insight_text[0].set_color(RED)
        insight_text.move_to(box.get_center())

        with self.voiceover(text="Finally, let us discuss the key interview insight you absolutely need to know. When interviewing for A I Engineering or Machine Learning Infrastructure roles, interviewers love to test your understanding of system tradeoffs.") as tracker:
            self.play(FadeIn(insight_title, shift=UP))
            self.play(Create(box))

        with self.voiceover(text="The biggest tradeoff with HNSW, and specifically at Layer 0, is its immense memory consumption. Storing all those structural graph edges—meaning the pointers connecting neighbors—for millions of vectors requires a massive amount of Random Access Memory compared to a simple flat index. Every node needs to keep a list of its connections, and in a densely connected Layer 0, that adds up very quickly.") as tracker:
            self.play(Write(insight_text))

        self.wait(1.5)

        with self.voiceover(text="You are explicitly trading memory space for blazing fast search speed. Furthermore, inserting new vectors into an HNSW index is slower because the graph must be dynamically updated and edges rewired. Understanding this space-time tradeoff—knowing when to use HNSW versus a simpler index—is the hallmark of a mature A I Engineer.") as tracker:
            self.play(insight_text[0].animate.scale(1.1))
            self.wait(1.0)
            self.play(insight_text[0].animate.scale(1.0/1.1))

        self.wait(1.5)

        with self.voiceover(text="Thank you for joining this deep dive into HNSW Layer 0. Keep practicing, and good luck with your A I engineering journey.") as tracker:
            self.play(FadeOut(insight_title), FadeOut(box), FadeOut(insight_text), FadeOut(title))

        self.wait(1.5)
