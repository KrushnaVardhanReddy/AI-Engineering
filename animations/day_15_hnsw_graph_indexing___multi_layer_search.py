from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class HNSWMultiLayerSearch(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # 1. What is it?
        title = Text("HNSW Graph Indexing", color=BLACK).scale(1.2).to_edge(UP)
        subtitle = Text("Hierarchical Navigable Small World", color=BLUE).scale(0.8).next_to(title, DOWN)

        with self.voiceover(text="Today we are exploring HNSW, which stands for Hierarchical Navigable Small World. It is a graph-based data structure used for extremely fast Approximate Nearest Neighbor search.") as tracker:
            self.play(Write(title))
            self.play(FadeIn(subtitle))
            self.wait(1.5)

        # Drawing a simplified graph
        layer_texts = VGroup(
            Text("Layer 2 (Few nodes, long links)", color=BLACK).scale(0.5),
            Text("Layer 1 (More nodes, medium links)", color=BLACK).scale(0.5),
            Text("Layer 0 (All nodes, short links)", color=BLACK).scale(0.5)
        ).arrange(DOWN, buff=2).shift(LEFT * 3)

        dots_layer2 = VGroup(Dot(color=RED).shift(UP * 1.5 + RIGHT), Dot(color=RED).shift(UP * 2.5 + RIGHT * 4))
        dots_layer1 = VGroup(Dot(color=GREEN).shift(UP * 0.5 + RIGHT * 1.5), Dot(color=GREEN).shift(UP * 0.2 + RIGHT * 3.5), Dot(color=GREEN).shift(DOWN * 0.2 + RIGHT * 2))
        dots_layer0 = VGroup(
            Dot(color=BLUE).shift(DOWN * 1.5 + RIGHT * 0.5), Dot(color=BLUE).shift(DOWN * 2 + RIGHT * 2),
            Dot(color=BLUE).shift(DOWN * 1.8 + RIGHT * 3.5), Dot(color=BLUE).shift(DOWN * 2.5 + RIGHT * 1.5),
            Dot(color=BLUE).shift(DOWN * 1.2 + RIGHT * 2.5), Dot(color=BLUE).shift(DOWN * 2.2 + RIGHT * 4.5)
        )

        with self.voiceover(text="HNSW builds a multi-layered graph. The top layer has very few nodes with long connections, allowing us to quickly skip across the dataset.") as tracker:
            self.play(FadeIn(layer_texts[0]), FadeIn(dots_layer2))
            line_l2 = Line(dots_layer2[0].get_center(), dots_layer2[1].get_center(), color=BLACK)
            self.play(Create(line_l2))
            self.wait(1.5)

        with self.voiceover(text="As we move down the layers, there are more nodes and shorter connections, allowing us to fine-tune our search until we reach the bottom layer which contains all the data points.") as tracker:
            self.play(FadeIn(layer_texts[1]), FadeIn(dots_layer1), FadeIn(layer_texts[2]), FadeIn(dots_layer0))
            self.wait(1.5)

        self.play(FadeOut(Group(title, subtitle, layer_texts, dots_layer2, dots_layer1, dots_layer0, line_l2)))

        # 2. Why do we need it?
        title_why = Text("Why do we need it?", color=BLACK).scale(1.2).to_edge(UP)
        with self.voiceover(text="Why do we need this multi-layered approach? Let's compare exhaustive search with HNSW.") as tracker:
            self.play(Write(title_why))
            self.wait(1.5)

        flat_title = Text("Exhaustive (Flat) Search", color=RED).scale(0.8).shift(UP * 2 + LEFT * 3)
        hnsw_title = Text("HNSW Search", color=GREEN).scale(0.8).shift(UP * 2 + RIGHT * 3)

        flat_nodes = VGroup(*[Dot(color=BLUE).move_to(LEFT * 4 + RIGHT * (i%4) * 0.5 + DOWN * (i//4) * 0.5) for i in range(16)])
        query_node = Dot(color=RED).move_to(LEFT * 2.5 + DOWN * 0.5)

        with self.voiceover(text="In an exhaustive search, or flat search, finding the nearest neighbor means calculating the distance to every single point in the database. This is extremely slow for large datasets.") as tracker:
            self.play(FadeIn(flat_title), FadeIn(flat_nodes), FadeIn(query_node))
            lines = VGroup(*[Line(query_node.get_center(), n.get_center(), color=BLACK, stroke_width=1) for n in flat_nodes])
            self.play(Create(lines), run_time=2)
            self.wait(1.5)

        hnsw_nodes = VGroup(
            Dot(color=BLUE).move_to(RIGHT * 2 + UP * 0.5),
            Dot(color=BLUE).move_to(RIGHT * 4 + UP * 1),
            Dot(color=BLUE).move_to(RIGHT * 3 + DOWN * 1)
        )
        hnsw_query = Dot(color=RED).move_to(RIGHT * 4.5 + DOWN * 0.5)

        with self.voiceover(text="With HNSW, we enter at the top layer, make a few large hops to get close to the query, and then drop down to finer layers. We only evaluate a tiny fraction of the dataset, making the search blazing fast.") as tracker:
            self.play(FadeIn(hnsw_title), FadeIn(hnsw_nodes), FadeIn(hnsw_query))
            arrow1 = Arrow(hnsw_nodes[0].get_center(), hnsw_nodes[1].get_center(), color=BLACK)
            arrow2 = Arrow(hnsw_nodes[1].get_center(), hnsw_nodes[2].get_center(), color=BLACK)
            arrow3 = Arrow(hnsw_nodes[2].get_center(), hnsw_query.get_center(), color=RED)
            self.play(Create(arrow1))
            self.play(Create(arrow2))
            self.play(Create(arrow3))
            self.wait(1.5)

        self.play(FadeOut(Group(title_why, flat_title, hnsw_title, flat_nodes, query_node, lines, hnsw_nodes, hnsw_query, arrow1, arrow2, arrow3)))

        # 3. Use Cases
        title_uses = Text("Real-World Use Cases", color=BLACK).scale(1.2).to_edge(UP)

        with self.voiceover(text="Because of its speed and efficiency, HNSW is the default indexing algorithm for many industry applications.") as tracker:
            self.play(Write(title_uses))
            self.wait(1.5)

        use_case1 = Text("1. Spotify: Finding similar songs instantly", color=BLUE).scale(0.7).shift(UP * 1)
        use_case2 = Text("2. Qdrant / Milvus: Default vector database index", color=PURPLE).scale(0.7).shift(DOWN * 1)

        with self.voiceover(text="For example, Spotify uses HNSW to instantly find and recommend songs similar to the one you are listening to among millions of tracks.") as tracker:
            self.play(FadeIn(use_case1))
            self.wait(1.5)

        with self.voiceover(text="Additionally, modern vector databases like Qdrant and Milvus use HNSW as their default indexing structure for retrieving document embeddings in retrieval-augmented generation pipelines.") as tracker:
            self.play(FadeIn(use_case2))
            self.wait(1.5)

        self.play(FadeOut(Group(title_uses, use_case1, use_case2)))

        # 4. Key Interview Insight
        title_insight = Text("Key Interview Insight", color=RED).scale(1.2).to_edge(UP)
        box = Rectangle(width=10, height=4, color=RED, fill_color=WHITE, fill_opacity=1).shift(DOWN * 0.5)

        insight_text = VGroup(
            Text("Tradeoff: Search Speed vs. Memory/Build Time", color=BLACK).scale(0.8),
            Text("M: Number of bi-directional links (Memory)", color=BLUE).scale(0.6),
            Text("ef_construction: Size of dynamic list during build (Time)", color=PURPLE).scale(0.6)
        ).arrange(DOWN, buff=0.5).move_to(box.get_center())

        with self.voiceover(text="If you are asked about HNSW in an interview, the most important insight to mention is the tradeoff between search speed, memory usage, and index build time.") as tracker:
            self.play(Write(title_insight))
            self.play(Create(box))
            self.play(Write(insight_text[0]))
            self.wait(1.5)

        with self.voiceover(text="Interviewers want to see if you know how to tune it. The parameter 'M' controls the number of links per node. Higher 'M' means better search quality but significantly higher memory usage.") as tracker:
            self.play(FadeIn(insight_text[1]))
            self.wait(1.5)

        with self.voiceover(text="The parameter 'ef_construction' controls the depth of the search during index building. A higher value means a more accurate graph, but it will take much longer to build. Understanding these knobs shows you are ready for production.") as tracker:
            self.play(FadeIn(insight_text[2]))
            self.wait(2)

        self.play(FadeOut(Group(title_insight, box, insight_text)))
