from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np

class KMeansClusteringScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        self.section_what_is_it()
        self.section_why_do_we_need_it()
        self.section_use_cases()
        self.section_key_insight()

    def section_what_is_it(self):
        title = Tex("What is K-Means Clustering?", color=BLACK, font_size=48)
        title.to_edge(UP)

        with self.voiceover(text="Welcome back to our machine learning interview prep series. Today, we are going to dive deep into a foundational unsupervised learning algorithm: K-Means Clustering. Let's start by defining exactly what K-Means Clustering is and how it operates fundamentally. In machine learning, K-Means is a popular unsupervised algorithm designed to automatically group unlabeled data into K distinct, non-overlapping clusters based on their inherent similarity.") as tracker:
            self.play(Write(title))
            self.wait(2.0)

            definition = Tex(
                "An ", "\\textbf{unsupervised learning}", " algorithm\\\\",
                "that groups ", "\\textbf{unlabeled data}", " into ", "\\textbf{K clusters}\\\\",
                "based on ", "\\textbf{similarity}."
            ).set_color(BLACK)
            definition.set_color_by_tex("\\textbf{unsupervised learning}", BLUE)
            definition.set_color_by_tex("\\textbf{unlabeled data}", RED)
            definition.set_color_by_tex("\\textbf{K clusters}", GREEN)
            definition.set_color_by_tex("\\textbf{similarity}", PURPLE)

            self.play(FadeIn(definition, shift=UP))
            self.wait(3.0)

        with self.voiceover(text="When we say unsupervised, it means our dataset has no predefined target variables or known outcomes. The algorithm has to find the hidden structures entirely on its own. The primary objective is to partition the data in such a way that the variance within each cluster is minimized. Let's break down the mathematical objective function step by step to see how it achieves this.") as tracker:
            self.wait(1.5)

        self.play(FadeOut(definition))

        with self.voiceover(text="At its core, K-Means seeks to find a set of clusters S, that minimizes a specific cost function. Let's write that out. First, we denote our goal: to find the argument that minimizes the total variance, across all possible cluster assignments S.") as tracker:
            eq1 = MathTex("\\arg\\min_S", color=BLACK)
            eq1.next_to(title, DOWN, buff=1)
            self.play(Write(eq1))
            self.wait(2.0)

        with self.voiceover(text="We are summing over all K clusters. So we introduce the outer sum from i equals 1 to K. This iterates through each of our distinct groups.") as tracker:
            eq2 = MathTex("\\arg\\min_S \\sum_{i=1}^K", color=BLACK)
            eq2.move_to(eq1, aligned_edge=LEFT)
            self.play(TransformMatchingTex(eq1, eq2))
            self.wait(2.0)

        with self.voiceover(text="Next, for each cluster i, we look at every single data point x that belongs to that specific cluster, denoted as S sub i.") as tracker:
            eq3 = MathTex("\\arg\\min_S \\sum_{i=1}^K \\sum_{x \\in S_i}", color=BLACK)
            eq3.move_to(eq2, aligned_edge=LEFT)
            self.play(TransformMatchingTex(eq2, eq3))
            self.wait(2.0)

        with self.voiceover(text="Finally, we measure the squared Euclidean distance between each data point x and the centroid or mean of that cluster, mu sub i. By minimizing this squared distance, K-Means tightly packs the data points around their respective centroids.") as tracker:
            eq4 = MathTex("\\arg\\min_S \\sum_{i=1}^K \\sum_{x \\in S_i} \\| x - \\mu_i \\|^2", color=BLACK)
            eq4.move_to(eq3, aligned_edge=LEFT)
            self.play(TransformMatchingTex(eq3, eq4))
            self.wait(3.0)

        self.play(FadeOut(eq4))

        with self.voiceover(text="To understand this visually, let's look at a 2D plot. Imagine we have a set of scattered data points spread across our feature space. K-Means will iteratively assign these points to clusters and adjust the central points to best represent the local groupings.") as tracker:
            axes = Axes(
                x_range=[0, 10, 1],
                y_range=[0, 10, 1],
                x_length=6,
                y_length=4,
                axis_config={"color": BLACK},
            ).shift(DOWN * 1)

            points_cluster1 = [axes.c2p(x, y) for x, y in [(2, 2), (2.5, 3), (3, 2.5), (1.5, 2.5), (3, 3)]]
            points_cluster2 = [axes.c2p(x, y) for x, y in [(7, 7), (8, 6), (7.5, 8), (6.5, 7.5), (8, 8)]]
            points_cluster3 = [axes.c2p(x, y) for x, y in [(7, 2), (8, 2.5), (7.5, 3), (6.5, 2.5), (8, 2)]]

            all_dots_list = [Dot(p, color=BLACK) for p in points_cluster1 + points_cluster2 + points_cluster3]
            all_dots = VGroup(*all_dots_list)

            self.play(Create(axes), FadeIn(all_dots))
            self.wait(2.0)

        with self.voiceover(text="The algorithm starts by randomly placing K initial centroids. Then it measures the distance from every point to these centroids. It discovers the natural groupings by assigning each data point to the nearest centroid, and then recalculating the centroid's position until it converges.") as tracker:
            c1 = Dot(axes.c2p(2.4, 2.6), color=BLUE, radius=0.15)
            c2 = Dot(axes.c2p(7.4, 7.3), color=RED, radius=0.15)
            c3 = Dot(axes.c2p(7.4, 2.4), color=GREEN, radius=0.15)
            self.play(FadeIn(VGroup(c1, c2, c3)))
            self.wait(1.5)

            colored_dots = VGroup()
            for i, p in enumerate(points_cluster1 + points_cluster2 + points_cluster3):
                if i < 5:
                    colored_dots.add(Dot(p, color=BLUE))
                elif i < 10:
                    colored_dots.add(Dot(p, color=RED))
                else:
                    colored_dots.add(Dot(p, color=GREEN))

            # Use 1-to-1 transform for smooth visuals
            self.play(*[Transform(all_dots[i], colored_dots[i]) for i in range(len(all_dots))])
            self.wait(3.0)

        self.play(FadeOut(VGroup(title, axes, all_dots, c1, c2, c3, colored_dots)))

    def section_why_do_we_need_it(self):
        title = Tex("Why do we need it?", color=BLACK, font_size=48)
        title.to_edge(UP)

        with self.voiceover(text="Now that we know what K-Means is, why do we actually need it? What specific problem does it solve in the real world? Let's consider a scenario where you are analyzing a massive dataset of customer purchases. Initially, you have absolutely no labels. You don't know who your premium customers are, who the bargain hunters are, or who the seasonal shoppers are.") as tracker:
            self.play(Write(title))

            np.random.seed(42)
            messy_data_points = [
                [np.random.uniform(-4, 4), np.random.uniform(-2, 2), 0]
                for _ in range(40)
            ]
            messy_data = VGroup(*[Dot(p, color=BLACK) for p in messy_data_points])
            self.play(FadeIn(messy_data))
            self.wait(2.0)

        with self.voiceover(text="Without an algorithm like K-Means, manually identifying patterns in high-dimensional data across thousands or millions of records is virtually impossible. To a human, this data just looks like a giant, unorganized cloud of noise.") as tracker:
            question_mark = Tex("?", color=RED, font_size=144).move_to(messy_data)
            self.play(FadeIn(question_mark))
            self.wait(2.0)
            self.play(FadeOut(question_mark))

        with self.voiceover(text="By applying K-Means clustering, we can automatically discover the underlying structures hidden within the noise. We can group similar items together, making the complex data actionable. For example, the algorithm might isolate three distinct customer segments based on their spending behavior and frequency.") as tracker:
            organized_data = VGroup()
            for dot in messy_data:
                x = dot.get_center()[0]
                if x < -1.5:
                    organized_data.add(Dot(dot.get_center(), color=BLUE))
                elif x > 1.5:
                    organized_data.add(Dot(dot.get_center(), color=GREEN))
                else:
                    organized_data.add(Dot(dot.get_center(), color=PURPLE))

            self.play(*[Transform(messy_data[i], organized_data[i]) for i in range(len(messy_data))])
            self.wait(2.0)

            circles = VGroup(
                Circle(color=BLUE, radius=1.5).move_to(LEFT * 2.5),
                Circle(color=PURPLE, radius=1.5).move_to(ORIGIN),
                Circle(color=GREEN, radius=1.5).move_to(RIGHT * 2.5)
            )
            self.play(Create(circles))
            self.wait(2.0)

        with self.voiceover(text="Now, instead of treating all customers identically, your business can craft tailored marketing strategies for the blue segment, the purple segment, and the green segment. This turns raw data into a strategic asset.") as tracker:
            self.wait(3.0)

        self.play(FadeOut(VGroup(title, messy_data, circles, organized_data)))

    def section_use_cases(self):
        title = Tex("Real-World Use Cases", color=BLACK, font_size=48)
        title.to_edge(UP)

        with self.voiceover(text="To solidify this concept, let's look at a couple of concrete real-world use cases where K-Means clustering is deployed in production environments.") as tracker:
            self.play(Write(title))
            self.wait(2.0)

        with self.voiceover(text="First, consider the music streaming giant, Spotify. They utilize various clustering algorithms, including K-Means, to analyze the audio features of millions of songs. Features like tempo, acousticness, and danceability. By clustering these songs, they can generate highly personalized recommendation playlists, like Discover Weekly, matching your taste with clusters of songs you haven't heard yet.") as tracker:
            spotify_box = SurroundingRectangle(Text("Spotify", color=BLACK), color=GREEN, fill_color=GREEN, fill_opacity=0.2, buff=0.5)
            spotify_text = Text("Spotify", color=BLACK).move_to(spotify_box)
            spotify_desc = Tex("Groups songs by audio features\\\\for personalized playlists", color=BLACK).next_to(spotify_box, DOWN)
            spotify_group = VGroup(spotify_box, spotify_text, spotify_desc).move_to(LEFT * 3)

            self.play(FadeIn(spotify_group))
            self.wait(3.0)

        with self.voiceover(text="Another major example is e-commerce companies like Amazon. They use clustering for Customer Segmentation. By grouping users based on their purchase history, browsing behavior, and cart abandonment rates, Amazon can run highly targeted email marketing campaigns and recommend products that users in similar clusters have bought.") as tracker:
            amazon_box = SurroundingRectangle(Text("Amazon", color=BLACK), color=BLUE, fill_color=BLUE, fill_opacity=0.2, buff=0.5)
            amazon_text = Text("Amazon", color=BLACK).move_to(amazon_box)
            amazon_desc = Tex("Customer Segmentation\\\\based on behavior", color=BLACK).next_to(amazon_box, DOWN)
            amazon_group = VGroup(amazon_box, amazon_text, amazon_desc).move_to(RIGHT * 3)

            self.play(FadeIn(amazon_group))
            self.wait(3.0)

        self.play(FadeOut(VGroup(title, spotify_group, amazon_group)))

    def section_key_insight(self):
        title = Tex("Key Interview Insight", color=BLACK, font_size=48)
        title.to_edge(UP)

        with self.voiceover(text="Now we arrive at the most important part of this video: the key interview insight. When an interviewer asks you about K-Means, what is the biggest gotcha they are secretly testing you on? It almost always comes down to the algorithm's initialization.") as tracker:
            self.play(Write(title))
            self.wait(2.0)

        with self.voiceover(text="The main issue to communicate is that standard K-Means is highly sensitive to the initial placement of the centroids. Because the starting positions are entirely random, it's possible to have a very poor start.") as tracker:
            insight_box = Rectangle(width=10, height=4, color=RED)
            insight_title = Tex("\\textbf{The Initialization Trap}", color=RED).move_to(insight_box.get_top() + DOWN * 0.5)
            insight_text1 = Tex("K-Means is highly sensitive to", color=BLACK).next_to(insight_title, DOWN, buff=0.5)
            insight_text2 = Tex("\\textbf{initial centroid placement}.", color=RED).next_to(insight_text1, DOWN, buff=0.2)

            insight_group = VGroup(insight_box, insight_title, insight_text1, insight_text2)
            self.play(FadeIn(insight_group))
            self.wait(3.0)

        with self.voiceover(text="If the random initialization starts in a bad spot, perhaps with multiple centroids clustered tightly together, the algorithm can easily get stuck in a local minimum. This means it converges, but it completely fails to find the true, globally optimal clusters.") as tracker:
            insight_text3 = Tex("Bad initialization $\\rightarrow$ Local Minimum", color=BLACK).next_to(insight_text2, DOWN, buff=0.5)
            self.play(Write(insight_text3))
            self.wait(3.0)

        with self.voiceover(text="So, what is the expected solution? When this comes up in an interview, you should immediately mention K-Means Plus Plus. K-Means Plus Plus is an improved initialization strategy that intelligently places the initial centroids far apart from each other.") as tracker:
            solution_text = Tex("Solution: Use \\textbf{K-Means++}", color=BLUE).next_to(insight_text3, DOWN, buff=0.5)
            self.play(Write(solution_text))
            self.wait(3.0)

        with self.voiceover(text="This drastically reduces the likelihood of converging to a poor local minimum. Remembering to bring up K-Means Plus Plus will show your interviewer that you don't just know the theory, but you understand the practical drawbacks of the standard algorithm and exactly how to fix them in production code.") as tracker:
            self.play(Indicate(solution_text, color=BLUE))
            self.wait(4.0)

        self.play(FadeOut(VGroup(title, insight_group, insight_text3, solution_text)))
