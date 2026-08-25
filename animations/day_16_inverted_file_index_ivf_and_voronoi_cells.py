from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class Day16IVFAndVoronoi(VoiceoverScene):
    def construct(self):
        # 1. Setup speech service and background
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # Helper colors based on aesthetic rules
        # Use BLACK for primary text and outlines, vibrant colors for highlights
        c_text = BLACK
        c_hl1 = BLUE
        c_hl2 = RED
        c_hl3 = GREEN
        c_hl4 = PURPLE

        # ---------------------------------------------------------
        # Part 1: What is it?
        # ---------------------------------------------------------
        title = Tex("Inverted File Index (IVF) and Voronoi Cells", color=c_text).scale(1.2)
        title.to_edge(UP)

        with self.voiceover(text="Welcome back to the AI Engineering prep series. Today, we're talking about the Inverted File Index, commonly known as I V F, and Voronoi Cells. This is a foundational concept for anyone building production vector databases or working on retrieval augmented generation systems. If you've ever wondered how databases can search through billions of vectors in milliseconds, this is the algorithmic secret sauce behind it.") as tracker:
            self.play(Write(title))
            self.wait(1.5)

        def_text = Tex(
            "What is it? \\\\",
            "An approximate nearest neighbor search strategy that partitions \\\\ vector space into regions using clustering.",
            color=c_text
        ).scale(0.8)
        def_text.next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="First, what exactly is it? The Inverted File Index, or IVF, is an approximate nearest neighbor search strategy. Its primary goal is to drastically speed up vector retrieval by intelligently partitioning the high dimensional vector space into smaller, manageable regions. These regions are known as Voronoi Cells. The partitioning is typically achieved using clustering techniques like the K-Means clustering algorithm during the index building phase.") as tracker:
            self.play(FadeIn(def_text, shift=DOWN))
            self.wait(1.5)

        # Draw a basic Voronoi diagram visually
        voronoi_group = VGroup()

        # Let's create some centroids
        centroids = [
            [-3, -1, 0],
            [0, 1, 0],
            [3, -2, 0]
        ]
        dots = VGroup(*[Dot(point, color=c_hl2, radius=0.1) for point in centroids])

        # Let's create some lines dividing the space (Voronoi edges)
        l1 = Line([-1.5, 3, 0], [-1.5, -0.5, 0], color=c_text)
        l2 = Line([-1.5, -0.5, 0], [0, -4, 0], color=c_text)
        l3 = Line([-1.5, -0.5, 0], [6, 1, 0], color=c_text)

        regions = VGroup(l1, l2, l3, dots)

        # Some points inside
        pts_left = VGroup(*[Dot([-3 + x*0.5, -1 + y*0.5, 0], color=c_hl1, radius=0.05) for x,y in [(1,0), (0,1), (-1,-1), (0,-1)]])
        pts_top = VGroup(*[Dot([0 + x*0.5, 1 + y*0.5, 0], color=c_hl3, radius=0.05) for x,y in [(1,1), (-1,0), (0,-1), (1,-1)]])
        pts_right = VGroup(*[Dot([3 + x*0.5, -2 + y*0.5, 0], color=c_hl4, radius=0.05) for x,y in [(0,0), (-1,1), (1,-1), (0.5,0.5)]])

        all_pts = VGroup(pts_left, pts_top, pts_right)
        voronoi_group.add(regions, all_pts)
        voronoi_group.scale(0.7).move_to(DOWN * 1)

        with self.voiceover(text="Imagine a vast two dimensional plane filled with millions of vector embeddings representing your text or image data. A Voronoi diagram mathematically divides this continuous plane into discrete regions based on the distance to a specific, predefined set of anchor points, which we call centroids. Every point in space gets assigned to the centroid it is closest to.") as tracker:
            self.play(FadeIn(dots))
            self.play(Write(VGroup(l1, l2, l3)))
            self.wait(1.5)

        with self.voiceover(text="Every resulting region, or Voronoi cell, represents a discrete cluster of data. All vectors that fall into a specific cell are mathematically closer to that cell's centroid than to any other centroid in the entire space. In the Inverted File Index architecture, we maintain a dictionary or hash map where each key is a centroid, and the value is the list of vectors residing within its specific Voronoi cell.") as tracker:
            self.play(FadeIn(all_pts))
            self.wait(1.5)

        with self.voiceover(text="So, instead of maintaining a giant, flat list of vectors, we maintain an organized index of centroids. These centroids act as signposts, pointing the database directly to the smaller, localized buckets of vectors where a query is most likely to find its nearest neighbors.") as tracker:
            self.play(all_pts.animate.scale(1.5))
            self.play(all_pts.animate.scale(1/1.5))
            self.wait(1.5)

        self.play(FadeOut(def_text), FadeOut(voronoi_group))

        # ---------------------------------------------------------
        # Part 2: Why do we need it?
        # ---------------------------------------------------------
        why_title = Tex("Why do we need it?", color=c_text).scale(0.9)
        why_title.next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="Why do we need this concept at all? Let's take a step back and examine the core problem it is trying to solve. Consider the traditional, brute force approach to searching vectors, often referred to as Flat Search, or Exhaustive K Nearest Neighbors search.") as tracker:
            self.play(Write(why_title))
            self.wait(1.5)

        # Before: Flat Search
        query_dot = Dot([-4, 0, 0], color=c_hl2, radius=0.15)
        query_label = Tex("Query", color=c_text).scale(0.6).next_to(query_dot, UP)

        db_dots = VGroup(*[Dot([-2 + i*0.8, -1 + (i%3)*0.8, 0], color=c_text, radius=0.08) for i in range(10)])
        db_dots.move_to(RIGHT * 2)

        before_group = VGroup(query_dot, query_label, db_dots)

        with self.voiceover(text="In a standard flat search paradigm, whenever a user provides a query vector, the system is forced to compute the mathematical distance between that single query and absolutely every other vector stored in the entire database.") as tracker:
            self.play(FadeIn(before_group))
            self.wait(1.5)

        lines_before = VGroup(*[Line(query_dot.get_center(), d.get_center(), color=c_hl2, stroke_width=2) for d in db_dots])

        with self.voiceover(text="As you can visually see on screen, the query node has to draw a line and check itself against everything. If your system contains 10 million document embeddings, that means performing 10 million expensive cosine similarity calculations for just one user query. This yields a linear time complexity of Big O of N. While this guarantees perfect accuracy, it simply does not scale for modern production applications.") as tracker:
            self.play(Write(lines_before, run_time=2))
            self.wait(1.5)

        self.play(FadeOut(lines_before), FadeOut(before_group))

        # After: IVF Search
        with self.voiceover(text="Now, let's contrast that and see how the Inverted File Index dramatically optimizes this process through smart space pruning.") as tracker:
            self.wait(1.0)

        after_query_dot = Dot([-4, 0, 0], color=c_hl2, radius=0.15)
        after_query_label = Tex("Query", color=c_text).scale(0.6).next_to(after_query_dot, UP)

        # 3 centroids
        c1 = Dot([0, 2, 0], color=c_hl1, radius=0.1)
        c2 = Dot([2, -1, 0], color=c_hl3, radius=0.1)
        c3 = Dot([1, 0.5, 0], color=c_hl4, radius=0.1)

        # Vectors around c2
        db_dots_c2 = VGroup(*[Dot([2 + (i-2)*0.4, -1 + (i%2)*0.4, 0], color=c_text, radius=0.08) for i in range(5)])

        after_group = VGroup(after_query_dot, after_query_label, c1, c2, c3, db_dots_c2)

        with self.voiceover(text="With IVF implemented, when a new query comes in, our first step is to calculate the distance from the query to only our limited set of predefined centroids. Because the number of centroids is very small, usually a parameter known as k, this is an extremely fast operation.") as tracker:
            self.play(FadeIn(VGroup(after_query_dot, after_query_label, c1, c2, c3)))
            c_lines = VGroup(
                Line(after_query_dot.get_center(), c1.get_center(), color=c_hl2, stroke_width=2),
                Line(after_query_dot.get_center(), c2.get_center(), color=c_hl2, stroke_width=2),
                Line(after_query_dot.get_center(), c3.get_center(), color=c_hl2, stroke_width=2)
            )
            self.play(Write(c_lines))
            self.wait(1.5)

        with self.voiceover(text="Once we have those distances, we simply identify the closest centroid. This immediately tells us which specific Voronoi cell our query most likely belongs to, and effectively gives us a localized neighborhood to focus on.") as tracker:
            self.play(c2.animate.scale(1.5))
            self.play(FadeOut(c_lines))
            self.wait(1.5)

        with self.voiceover(text="Then, and this is the crucial part, we only calculate distances between the query and the handful of vectors located within that specific cell. We completely ignore all other regions and clusters of the vector space, saving enormous amounts of compute time.") as tracker:
            self.play(FadeIn(db_dots_c2))
            cell_lines = VGroup(*[Line(after_query_dot.get_center(), d.get_center(), color=c_hl3, stroke_width=2) for d in db_dots_c2])
            self.play(Write(cell_lines))
            self.wait(1.5)

        with self.voiceover(text="By intelligently pruning the search space and skipping the vast majority of the vectors, IVF massively reduces the overall search scope. This shifts our search complexity down to sub-linear time, typically approximated as Big O of N divided by the number of clusters. The end result is a blazing fast retrieval process capable of handling billions of records.") as tracker:
            self.play(FadeOut(cell_lines), FadeOut(after_group), FadeOut(why_title))
            self.wait(1.5)

        # ---------------------------------------------------------
        # Part 3: Use Cases
        # ---------------------------------------------------------
        usecase_title = Tex("Real-World Use Cases", color=c_text).scale(0.9)
        usecase_title.next_to(title, DOWN, buff=0.5)

        uc1 = Tex("1. \\textbf{Spotify}: Similar Song Recommendations (FAISS)", color=c_text).scale(0.7)
        uc2 = Tex("2. \\textbf{Pinterest}: Visual Search \& Similar Pins", color=c_text).scale(0.7)

        uc_group = VGroup(uc1, uc2).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        uc_group.next_to(usecase_title, DOWN, buff=1.0)

        with self.voiceover(text="To ground this theory, let's look at a couple of massive scale real-world use cases where the Inverted File Index pattern is an absolute architectural necessity.") as tracker:
            self.play(Write(usecase_title))
            self.wait(1.5)

        with self.voiceover(text="First, consider Spotify's recommendation engine. When you are listening to a specific song, Spotify wants to instantly recommend the next similar track. Their embedding space contains hundreds of millions of songs, constantly updating. By using IVF under the hood, often inside powerful C++ libraries like Facebook AI Similarity Search, or FAISS, they can instantly locate and recommend songs that sit in the exact same Voronoi cell as your current track, bypassing the rest of the library entirely.") as tracker:
            self.play(FadeIn(uc1, shift=RIGHT))
            self.wait(1.5)

        with self.voiceover(text="Second, look at Pinterest. For their visual search product, where users upload an image to find visually similar pins, Pinterest relies heavily on partitioned indexes. IVF clusters millions of images based on similar visual features extracted by deep learning models. This architectural choice allows their search engine to retrieve highly relevant pins in a fraction of a second, without choking their database servers.") as tracker:
            self.play(FadeIn(uc2, shift=RIGHT))
            self.wait(1.5)

        self.play(FadeOut(uc_group), FadeOut(usecase_title))

        # ---------------------------------------------------------
        # Part 4: Key Interview Insight
        # ---------------------------------------------------------
        insight_title = Tex("Key Interview Insight", color=c_text).scale(0.9)
        insight_title.next_to(title, DOWN, buff=0.5)

        # Callout box
        box = Rectangle(width=10, height=4, color=c_hl2, fill_color=c_hl2, fill_opacity=0.1)
        box.next_to(insight_title, DOWN, buff=0.5)

        insight_text_1 = Tex("The Tradeoff: ", "Speed ", "vs. ", "Recall", color=c_text).scale(0.8)
        insight_text_1.set_color_by_tex("Speed", c_hl1)
        insight_text_1.set_color_by_tex("Recall", c_hl2)
        insight_text_1.move_to(box.get_center() + UP*1)

        insight_text_2 = Tex("nlist", " : Number of clusters (Voronoi cells)", color=c_text).scale(0.7)
        insight_text_3 = Tex("nprobe", " : Number of clusters to search", color=c_text).scale(0.7)
        insight_text_2.set_color_by_tex("nlist", c_hl3)
        insight_text_3.set_color_by_tex("nprobe", c_hl3)

        params_group = VGroup(insight_text_2, insight_text_3).arrange(DOWN, aligned_edge=LEFT)
        params_group.next_to(insight_text_1, DOWN, buff=0.5)

        with self.voiceover(text="Finally, let's discuss the most important part of this module: The Key Interview Insight. When a senior engineering interviewer asks you about IVF, they are not just checking if you know what it is. They are rigorously testing if you understand its fundamental tradeoff, and how to tune a production database.") as tracker:
            self.play(Write(insight_title))
            self.play(Write(box))
            self.wait(1.5)

        with self.voiceover(text="The core tradeoff of the Inverted File Index is Speed versus Recall. Recall is a metric that measures how often the true nearest neighbor is actually successfully retrieved by your query. Because IVF intentionally only searches a small subset of the total space, the true nearest neighbor might occasionally be missed, especially if it lies just barely across the boundary of an adjacent Voronoi cell that you didn't check.") as tracker:
            self.play(FadeIn(insight_text_1))
            self.wait(1.5)

        with self.voiceover(text="To manage this edge effect problem, interviewers absolutely expect you to know two critical hyper parameters: n-list, and n-probe.") as tracker:
            self.play(FadeIn(params_group))
            self.wait(1.5)

        with self.voiceover(text="N-list dictates the total number of Voronoi cells you partition the entire vector space into during index creation. N-probe dictates the number of those neighboring cells you choose to actively search at query time. Increasing the n-probe parameter tells the database to search more adjacent cells. This directly improves your recall and search accuracy, but it proportionally decreases your search speed and increases compute costs. Tuning these two variables against your business requirements is the literal essence of vector database optimization.") as tracker:
            self.play(insight_text_3.animate.scale(1.1))
            self.play(insight_text_3.animate.scale(1/1.1))
            self.wait(1.5)

        # Math text derivation for equation using TransformMatchingTex
        math_1_part1 = MathTex("Recall", color=c_text).scale(0.9)
        math_1_part1.next_to(params_group, DOWN, buff=0.5).shift(LEFT*2)
        math_1_full = MathTex("Recall", "\\propto", "nprobe", color=c_text).scale(0.9)
        math_1_full.move_to(math_1_part1.get_center())

        math_2_part1 = MathTex("Speed", color=c_text).scale(0.9)
        math_2_part1.next_to(params_group, DOWN, buff=0.5).shift(RIGHT*2)
        math_2_full = MathTex("Speed", "\\propto", "\\frac{1}{nprobe}", color=c_text).scale(0.9)
        math_2_full.move_to(math_2_part1.get_center())

        with self.voiceover(text="Mathematically speaking, we can formulate this tradeoff directly. We can say that the recall percentage of your query is directly proportional to the n-probe parameter.") as tracker:
            self.play(Write(math_1_part1))
            self.wait(0.5)
            self.play(TransformMatchingTex(math_1_part1, math_1_full))
            self.wait(1.5)

        with self.voiceover(text="Conversely, your overall query execution speed is inversely proportional to the n-probe parameter.") as tracker:
            self.play(Write(math_2_part1))
            self.wait(0.5)
            self.play(TransformMatchingTex(math_2_part1, math_2_full))
            self.wait(1.5)

        with self.voiceover(text="Mastering this tradeoff will confidently set you apart in System Design interviews. Understanding the knobs and dials of your infrastructure is what makes you an AI Engineer, not just a developer. That wraps up Day 16 on the Inverted File Index and Voronoi Cells. Take time to study this tradeoff. Keep building, and I will see you in the next lesson!") as tracker:
            self.play(FadeOut(VGroup(box, insight_text_1, params_group, math_1_full, math_2_full, insight_title, title)))
            self.wait(1.5)

        # End of scene
        self.wait(2)
