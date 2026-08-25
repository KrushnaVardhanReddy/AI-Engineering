from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class TSNEVisualization(VoiceoverScene):
    def construct(self):
        # Set up voiceover service and aesthetics
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # We define consistent colors based on instructions
        TEXT_COLOR = BLACK
        H_COLOR1 = BLUE
        H_COLOR2 = RED
        H_COLOR3 = GREEN
        H_COLOR4 = PURPLE

        ###################################################################
        # Section 1: What is it?
        ###################################################################

        title = Tex("t-SNE", color=TEXT_COLOR, font_size=64)
        subtitle = Tex("t-Distributed Stochastic Neighbor Embedding", color=TEXT_COLOR, font_size=36)
        VGroup(title, subtitle).arrange(DOWN, buff=0.5).to_edge(UP)

        with self.voiceover(text="Welcome to Day 29 of our AI interview prep series. Today we are diving deep into t-SNE, which stands for t-Distributed Stochastic Neighbor Embedding. This is a very common topic when discussing unsupervised learning and data visualization techniques.") as tracker:
            self.play(Write(title))
            self.play(FadeIn(subtitle, shift=UP))
            self.wait(1.5)

        def_text = Tex(
            "A non-linear dimensionality reduction technique\\\\",
            "specifically designed to visualize\\\\",
            "high-dimensional data in 2D or 3D.",
            color=TEXT_COLOR, font_size=40
        )
        def_text.next_to(subtitle, DOWN, buff=1)

        with self.voiceover(text="At its core, it is a powerful non-linear dimensionality reduction technique specifically designed to help us visualize high-dimensional data, typically by projecting it down into a 2D or 3D space where we can easily inspect it.") as tracker:
            self.play(Write(def_text))
            self.wait(1.5)

        self.play(FadeOut(def_text))

        # Visual diagram of reduction
        high_dim_box = Rectangle(width=2, height=3, color=H_COLOR1)
        high_dim_text = Tex("High-Dimensional\\\\Vectors\\\\(e.g. 768-D)", color=TEXT_COLOR, font_size=32).move_to(high_dim_box)
        high_dim_group = VGroup(high_dim_box, high_dim_text).shift(LEFT * 4)

        low_dim_box = Rectangle(width=3, height=3, color=H_COLOR3)
        low_dim_text = Tex("2D Scatter\\\\Plot", color=TEXT_COLOR).move_to(low_dim_box)
        low_dim_group = VGroup(low_dim_box, low_dim_text).shift(RIGHT * 4)

        arrow = Arrow(high_dim_box.get_right(), low_dim_box.get_left(), color=TEXT_COLOR, buff=0.5)
        arrow_text = Tex("t-SNE", color=H_COLOR2).next_to(arrow, UP)

        diagram = VGroup(high_dim_group, arrow, arrow_text, low_dim_group)
        diagram.shift(DOWN * 1)

        with self.voiceover(text="Imagine you have a dataset where each data point is represented by hundreds or even thousands of features, like a 768-dimensional text embedding from a language model. Humans cannot see in 768 dimensions. t-SNE acts as a smart mathematical translator, mapping those massive, incomprehensible vectors down into a simple two-dimensional scatter plot that our brains can actually process and understand.") as tracker:
            self.play(FadeIn(high_dim_group))
            self.play(FadeIn(arrow), Write(arrow_text))
            self.play(FadeIn(low_dim_group))
            self.wait(2.5)

        self.play(FadeOut(VGroup(title, subtitle, diagram)))

        ###################################################################
        # Section 2: Why do we need it?
        ###################################################################

        sec2_title = Tex("Why do we need it?", color=TEXT_COLOR, font_size=56).to_edge(UP)

        with self.voiceover(text="You might be wondering: why do we need a complex algorithm like t-SNE when we already have simpler, faster dimensionality reduction techniques like Principal Component Analysis, or PCA? Let's take a look at a common problem when dealing with real-world, non-linear data.") as tracker:
            self.play(Write(sec2_title))
            self.wait(1.5)

        # Draw axes for a bad overlap scenario
        axes1 = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=5, y_length=5,
            axis_config={"color": TEXT_COLOR, "include_numbers": False}
        ).shift(LEFT * 3 + DOWN * 0.5)

        axes1_label = Tex("Without t-SNE (e.g. PCA)", color=TEXT_COLOR, font_size=32).next_to(axes1, UP)

        import random
        random.seed(42)

        # Create a messy overlapping cluster visualization
        dots_bad = VGroup()
        for _ in range(40):
            # Overlapping random blobs
            x1, y1 = random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)
            x2, y2 = random.uniform(-1, 2), random.uniform(-1, 2)
            dots_bad.add(Dot(axes1.c2p(x1, y1), color=H_COLOR1, radius=0.08))
            dots_bad.add(Dot(axes1.c2p(x2, y2), color=H_COLOR2, radius=0.08))

        with self.voiceover(text="If we use linear methods like PCA on highly complex, non-linear data manifolds, the algorithm often fails to capture the underlying structure. As a result, the different classes or clusters end up collapsing on top of each other, creating a messy, indistinguishable overlapping blob where you can't tell the blue category from the red category.") as tracker:
            self.play(FadeIn(axes1), Write(axes1_label))
            self.play(FadeIn(dots_bad))
            self.wait(2.5)

        axes2 = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=5, y_length=5,
            axis_config={"color": TEXT_COLOR, "include_numbers": False}
        ).shift(RIGHT * 3 + DOWN * 0.5)

        axes2_label = Tex("With t-SNE", color=TEXT_COLOR, font_size=32).next_to(axes2, UP)

        # Create well-separated clusters
        dots_good = VGroup()
        for _ in range(40):
            x1, y1 = random.gauss(-1.5, 0.4), random.gauss(1.5, 0.4)
            x2, y2 = random.gauss(1.5, 0.4), random.gauss(-1.5, 0.4)
            dots_good.add(Dot(axes2.c2p(x1, y1), color=H_COLOR1, radius=0.08))
            dots_good.add(Dot(axes2.c2p(x2, y2), color=H_COLOR2, radius=0.08))

        with self.voiceover(text="However, t-SNE takes a completely different approach. It models the probabilities of points being neighbors in high-dimensional space, and then tries to recreate those exact same probabilities in low-dimensional space. Over many iterations, it effectively untangles the data by pulling similar points close together and actively pushing dissimilar points apart. This non-linear mapping reveals the hidden, distinct clusters that PCA simply cannot see.") as tracker:
            self.play(FadeIn(axes2), Write(axes2_label))

            # To show the transformation, let's copy dots_bad to the right and transform them to dots_good
            dots_trans = dots_bad.copy().move_to(axes2.c2p(0,0))
            self.play(FadeIn(dots_trans))
            self.play(Transform(dots_trans, dots_good), run_time=3)
            self.wait(2.5)

        self.play(FadeOut(VGroup(sec2_title, axes1, axes1_label, dots_bad, axes2, axes2_label, dots_trans, dots_good)))

        ###################################################################
        # Section 3: Use Cases
        ###################################################################

        sec3_title = Tex("Real-World Use Cases", color=TEXT_COLOR, font_size=56).to_edge(UP)

        with self.voiceover(text="So, where is t-SNE actually used in the real world by large technology companies? Because it is computationally expensive, it's almost exclusively used for data exploration and debugging, rather than running live in production inference pipelines.") as tracker:
            self.play(Write(sec3_title))
            self.wait(1.5)

        uc1_box = RoundedRectangle(width=8, height=2.5, corner_radius=0.2, color=H_COLOR1, fill_opacity=0.1)
        uc1_text = Tex(r"\textbf{Google (TensorBoard):}\\Visualizing word embeddings (like Word2Vec or BERT)\\to discover semantic clusters and biases.", color=TEXT_COLOR, font_size=36)
        uc1_text.move_to(uc1_box)
        uc1_group = VGroup(uc1_box, uc1_text).shift(UP * 1)

        with self.voiceover(text="For example, Google uses t-SNE extensively, and it is a built-in feature inside their TensorBoard visualization toolkit. Machine learning researchers use t-SNE to project massive, complex word embeddings from models like Word-To-Vec or BERT into 2D space. This allows them to visually inspect if the model has learned appropriate semantic relationships, and to audit the model for unintended clusters of bias before deployment.") as tracker:
            self.play(FadeIn(uc1_box), Write(uc1_text))
            self.wait(2.5)

        uc2_box = RoundedRectangle(width=8, height=2.5, corner_radius=0.2, color=H_COLOR3, fill_opacity=0.1)
        uc2_text = Tex(r"\textbf{Spotify:}\\Clustering users and tracks based on high-dimensional\\listening behavior to refine recommendation engines.", color=TEXT_COLOR, font_size=36)
        uc2_text.move_to(uc2_box)
        uc2_group = VGroup(uc2_box, uc2_text).next_to(uc1_group, DOWN, buff=0.5)

        with self.voiceover(text="Similarly, companies like Spotify rely on t-SNE during their exploratory data analysis phase to cluster users and music tracks. By reducing thousands of dimensions of user listening behavior into a simple visual map, data scientists can identify distinct, subtle micro-genres and mathematically verify that their personalized recommendation engines are grouping similar songs together correctly.") as tracker:
            self.play(FadeIn(uc2_box), Write(uc2_text))
            self.wait(2.5)

        self.play(FadeOut(VGroup(sec3_title, uc1_group, uc2_group)))

        ###################################################################
        # Section 4: Key Interview Insight
        ###################################################################

        sec4_title = Tex("Key Interview Insight", color=H_COLOR2, font_size=64).to_edge(UP)

        with self.voiceover(text="Now for the most important part of today's lesson: the key interview insight. Machine Learning interviewers love to ask about t-SNE specifically to test if you actually understand its mathematical limitations, or if you just treat it as a black box.") as tracker:
            self.play(Write(sec4_title))
            self.wait(1.5)

        insight_box = RoundedRectangle(width=10, height=4, corner_radius=0.2, color=H_COLOR4, fill_opacity=0.05)

        insight_pt1 = Tex(r"\textbf{The Gotcha:}", color=H_COLOR2, font_size=44)
        insight_pt2 = Tex("t-SNE preserves ", "LOCAL", " structure, \\\\not ", "GLOBAL", " distance.", color=TEXT_COLOR, font_size=40)
        insight_pt2.set_color_by_tex("LOCAL", H_COLOR3)
        insight_pt2.set_color_by_tex("GLOBAL", H_COLOR2)

        insight_group = VGroup(insight_pt1, insight_pt2).arrange(DOWN, buff=0.5)
        insight_box.move_to(insight_group)

        with self.voiceover(text="The major gotcha that trips up many candidates is this fundamental rule: t-SNE is designed to strictly preserve local structure, not global distance.") as tracker:
            self.play(FadeIn(insight_box))
            self.play(Write(insight_pt1))
            self.play(FadeIn(insight_pt2, shift=UP))
            self.wait(2.5)

        sub_insight = Tex("The distance between two completely different clusters\\\\on the plot is mathematically meaningless.", color=TEXT_COLOR, font_size=32)
        sub_insight.next_to(insight_box, DOWN, buff=0.5)

        with self.voiceover(text="This means that if you look at a t-SNE plot, the points within a single tight cluster are indeed very similar to each other. However, if you measure the empty space between two completely different clusters on opposite sides of the plot, that distance is practically meaningless. You cannot claim that two far-apart clusters are highly dissimilar just because they are far apart on the 2D graph. The global geometry is distorted.") as tracker:
            self.play(Write(sub_insight))
            self.wait(3.5)

        # Math component line by line to show KL Divergence minimization
        math_title = Tex("It minimizes Kullback-Leibler (KL) Divergence:", color=TEXT_COLOR, font_size=32).to_edge(DOWN).shift(UP*1.5)

        math_eq1 = MathTex("Cost", "=", r"\sum_i", "KL(", "P_i", "||", "Q_i", ")", color=TEXT_COLOR, font_size=36).next_to(math_title, DOWN, buff=0.3)
        math_eq1.set_color_by_tex("P_i", H_COLOR1)
        math_eq1.set_color_by_tex("Q_i", H_COLOR2)

        math_eq2 = MathTex("Cost", "=", r"\sum_i", r"\sum_j", "p_{j|i}", r"\log", r"\frac{p_{j|i}}{q_{j|i}}", color=TEXT_COLOR, font_size=36).next_to(math_title, DOWN, buff=0.3)
        math_eq2.set_color_by_tex("p_{j|i}", H_COLOR1)
        math_eq2.set_color_by_tex("q_{j|i}", H_COLOR2)

        with self.voiceover(text="To understand why this happens, we have to look at the loss function. Under the hood, t-SNE minimizes the Kullback-Leibler divergence between the high-dimensional probability distribution P, representing the true data, and the low-dimensional probability distribution Q, representing the scatter plot.") as tracker:
            self.play(FadeIn(math_title))
            self.play(Write(math_eq1))
            self.wait(2.5)

        with self.voiceover(text="If we expand this out, we can see the asymmetric nature of KL divergence. Because it uses a Student's t-distribution with heavy tails for the low-dimensional space, the algorithm heavily penalizes you for placing true neighbors far apart, but it is extremely lenient if you place distant points somewhat close together. This mathematical asymmetry is exactly why local clusters are perfect, but the global macro-structure falls apart.") as tracker:
            self.play(TransformMatchingTex(math_eq1, math_eq2))
            self.wait(4)

        # Final conclusion
        self.play(FadeOut(VGroup(sec4_title, insight_box, insight_group, sub_insight, math_title, math_eq2)))

        final_text = Tex("Mastering t-SNE allows you to debug embeddings and\\\\showcase deep understanding in interviews.", color=TEXT_COLOR, font_size=40)
        with self.voiceover(text="Mastering the mechanics of t-SNE, rather than just calling the scikit-learn function, allows you to visually debug embedding models and showcase a deep, nuanced understanding of data visualization trade-offs during your senior engineering interviews. Keep practicing, and I will see you in the next lesson!") as tracker:
            self.play(Write(final_text))
            self.wait(3)

        self.play(FadeOut(final_text))
        self.wait(1)
