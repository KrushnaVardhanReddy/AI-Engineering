from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np

class SVMMacroScene(VoiceoverScene):
    def construct(self):
        # Initial settings
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Title Sequence
        title = Tex(
            "Support Vector Machines (SVM)\\\\and the Kernel Trick",
            color=BLACK,
            font_size=56
        )

        with self.voiceover(text="Welcome back to our AI Engineering mastery series! Today, we are exploring Day twenty-four: Support Vector Machines and the Kernel Trick.") as tracker:
            self.play(Write(title))
            self.wait(1.5)

        with self.voiceover(text="Let's dive into the core concepts, discover why we need them, explore some use cases, and wrap up with a key interview insight.") as tracker:
            self.play(title.animate.to_edge(UP).scale(0.6))
            self.wait(1.5)

        # ---------------------------------------------------------
        # Section 1: What is it?
        # ---------------------------------------------------------
        section_1_title = Tex("1. What is it?", color=BLACK, font_size=40).to_edge(UP).align_to(title, LEFT)

        definition_text = Tex(
            "A powerful algorithm that finds the optimal hyperplane\\\\to separate classes with the maximum margin.",
            color=BLACK,
            font_size=36
        ).next_to(section_1_title, DOWN, buff=0.5)

        with self.voiceover(text="First, what is a Support Vector Machine? Simply put, it is a powerful algorithm that finds the optimal hyperplane to separate data classes with the maximum possible margin.") as tracker:
            self.play(
                Transform(title, section_1_title),
                FadeIn(definition_text, shift=DOWN)
            )
            self.wait(1.5)

        # Draw a 2D SVM Diagram
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": BLACK, "include_numbers": False}
        ).next_to(definition_text, DOWN, buff=0.5)

        # Create points
        np.random.seed(42)
        blue_points = [axes.c2p(x, y) for x, y in zip(np.random.normal(-1.5, 0.5, 15), np.random.normal(1.5, 0.5, 15))]
        red_points = [axes.c2p(x, y) for x, y in zip(np.random.normal(1.5, 0.5, 15), np.random.normal(-1.5, 0.5, 15))]

        blue_dots = VGroup(*[Dot(p, color=BLUE) for p in blue_points])
        red_dots = VGroup(*[Dot(p, color=RED) for p in red_points])

        with self.voiceover(text="Imagine we have two groups of data points in a two-dimensional space. Our goal is to draw a line that separates the blue points from the red points.") as tracker:
            self.play(FadeIn(axes))
            self.play(FadeIn(blue_dots), FadeIn(red_dots))
            self.wait(1.5)

        hyperplane = Line(axes.c2p(-3, 0), axes.c2p(3, 0), color=BLACK, stroke_width=4)
        hyperplane.rotate(PI/4)

        margin_top = Line(axes.c2p(-3, 0.8), axes.c2p(3, 0.8), color=BLACK, stroke_width=2).set_opacity(0.5)
        margin_top.rotate(PI/4)

        margin_bottom = Line(axes.c2p(-3, -0.8), axes.c2p(3, -0.8), color=BLACK, stroke_width=2).set_opacity(0.5)
        margin_bottom.rotate(PI/4)

        with self.voiceover(text="The algorithm draws many possible lines, but it chooses the one that maximizes the distance—called the margin—between the line and the closest points of each class. This optimal line is our decision boundary, or hyperplane.") as tracker:
            self.play(Write(hyperplane))
            self.play(Write(margin_top), Write(margin_bottom))
            self.wait(1.5)

        sv1 = Circle(radius=0.15, color=GREEN).move_to(axes.c2p(-0.35, 1.25))
        sv2 = Circle(radius=0.15, color=GREEN).move_to(axes.c2p(0.8, -0.4))
        sv_label = Tex("Support Vectors", color=GREEN, font_size=28).next_to(sv1, UP)

        with self.voiceover(text="These closest data points are critical. They are literally the vectors that support the hyperplane. We call them Support Vectors. If you moved any other points, the boundary would not change, but moving these support vectors would shift everything.") as tracker:
            self.play(FadeIn(sv1), FadeIn(sv2), Write(sv_label))
            self.wait(2.0)

        self.play(FadeOut(VGroup(axes, blue_dots, red_dots, hyperplane, margin_top, margin_bottom, sv1, sv2, sv_label, definition_text)))

        # ---------------------------------------------------------
        # Section 2: Why do we need it? (The Kernel Trick)
        # ---------------------------------------------------------
        section_2_title = Tex("2. Why do we need it? (The Kernel Trick)", color=BLACK, font_size=40).to_edge(UP).align_to(title, LEFT)

        with self.voiceover(text="Now, why do we need this? And more importantly, what happens when data isn't perfectly separable by a straight line? This brings us to the famous Kernel Trick.") as tracker:
            self.play(Transform(title, section_2_title))
            self.wait(1.5)

        # 1D Non-separable data
        ax_1d = NumberLine(x_range=[-3, 3, 1], length=8, color=BLACK, include_numbers=False)

        # Inner group red, outer group blue
        red_1d_vals = [-0.5, 0, 0.5]
        blue_1d_vals = [-2, -1.5, 1.5, 2]

        red_1d_dots = VGroup(*[Dot(ax_1d.n2p(x), color=RED) for x in red_1d_vals])
        blue_1d_dots = VGroup(*[Dot(ax_1d.n2p(x), color=BLUE) for x in blue_1d_vals])

        with self.voiceover(text="Let's look at a problem where we only have one dimension. We have red points clustered in the middle, and blue points on the outside. Try as you might, you cannot draw a single point or line in one dimension to separate the red from the blue.") as tracker:
            self.play(FadeIn(ax_1d))
            self.play(FadeIn(red_1d_dots), FadeIn(blue_1d_dots))
            self.wait(1.5)

        # Show projection to 2D
        ax_2d = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 9, 2],
            x_length=8,
            y_length=4,
            axis_config={"color": BLACK, "include_numbers": False}
        ).shift(UP*0.5)

        kernel_eq = MathTex("y = x^2", color=BLACK).next_to(ax_2d, UP)

        with self.voiceover(text="This is where the Kernel trick shines. Instead of struggling in one dimension, what if we mathematically project our data into a higher dimension? Let's apply a transformation, say, y equals x squared.") as tracker:
            self.play(
                Transform(ax_1d, ax_2d.x_axis),
                FadeIn(ax_2d.y_axis),
                Write(kernel_eq)
            )
            self.wait(1.5)

        # Move points to y = x^2
        red_2d_dots = VGroup(*[Dot(ax_2d.c2p(x, x**2), color=RED) for x in red_1d_vals])
        blue_2d_dots = VGroup(*[Dot(ax_2d.c2p(x, x**2), color=BLUE) for x in blue_1d_vals])

        with self.voiceover(text="As we square the x values to become y values, our data is lifted into two dimensions. The red points, which had small x values, stay low. The blue points, with large x values, shoot up high.") as tracker:
            self.play(
                Transform(red_1d_dots, red_2d_dots),
                Transform(blue_1d_dots, blue_2d_dots)
            )
            self.wait(1.5)

        # Draw 2D hyperplane
        hyperplane_2d = Line(ax_2d.c2p(-3, 1), ax_2d.c2p(3, 1), color=GREEN, stroke_width=4)

        with self.voiceover(text="Now, the data is linearly separable! We can easily draw a straight horizontal line to perfectly separate the red and blue classes. This projection into higher dimensions is the essence of the Kernel trick. It computes these relationships computationally cheaply without actually transforming the data.") as tracker:
            self.play(Write(hyperplane_2d))
            self.wait(2.0)

        self.play(FadeOut(VGroup(ax_1d, ax_2d.y_axis, red_1d_dots, blue_1d_dots, kernel_eq, hyperplane_2d)))

        # ---------------------------------------------------------
        # Section 3: Use Cases
        # ---------------------------------------------------------
        section_3_title = Tex("3. Real-World Use Cases", color=BLACK, font_size=40).to_edge(UP).align_to(title, LEFT)

        with self.voiceover(text="So, where are Support Vector Machines actually used in the real world?") as tracker:
            self.play(Transform(title, section_3_title))
            self.wait(1.5)

        case_1 = Tex(
            "\\textbf{1. Bioinformatics (e.g., Illumina):}\\\\Gene expression classification.",
            color=BLACK,
            font_size=36
        ).shift(UP*1.5)

        case_2 = Tex(
            "\\textbf{2. Text Categorization (e.g., Reuters):}\\\\High-dimensional document sorting.",
            color=BLACK,
            font_size=36
        ).next_to(case_1, DOWN, buff=1.0)

        with self.voiceover(text="The first major use case is in bioinformatics. Companies like Illumina use Support Vector Machines to classify gene expression data. Because genetic data has thousands of dimensions, SVMs excel at finding boundaries where other algorithms fail.") as tracker:
            self.play(FadeIn(case_1, shift=UP))
            self.wait(1.5)

        with self.voiceover(text="A second classic use case is text categorization. News organizations like Reuters historically used SVMs to sort documents into topics. Even before deep learning, SVMs were the gold standard for high-dimensional text data, easily separating sparse bag-of-words vectors.") as tracker:
            self.play(FadeIn(case_2, shift=UP))
            self.wait(2.0)

        self.play(FadeOut(VGroup(case_1, case_2)))

        # ---------------------------------------------------------
        # Section 4: Key Interview Insight
        # ---------------------------------------------------------
        section_4_title = Tex("4. Key Interview Insight", color=BLACK, font_size=40).to_edge(UP).align_to(title, LEFT)

        with self.voiceover(text="Finally, let's talk about the key insight that interviewers will expect you to know when discussing SVMs.") as tracker:
            self.play(Transform(title, section_4_title))
            self.wait(1.5)

        box = Rectangle(width=10, height=4, color=PURPLE, stroke_width=4, fill_color=PURPLE, fill_opacity=0.1)

        insight_title = Tex("\\textbf{The Bias-Variance Tradeoff (C Parameter)}", color=PURPLE, font_size=36).next_to(box.get_top(), DOWN, buff=0.3)
        insight_desc = Tex(
            "Low C: Soft margin, allows misclassifications (High Bias, Low Variance).\\\\",
            "High C: Hard margin, tries to perfectly separate (Low Bias, High Variance).",
            color=BLACK,
            font_size=30
        ).next_to(insight_title, DOWN, buff=0.5)

        with self.voiceover(text="In a machine learning interview, you will almost certainly be asked about the 'C' parameter in Support Vector Machines.") as tracker:
            self.play(Write(box), FadeIn(insight_title))
            self.wait(1.5)

        with self.voiceover(text="The C parameter controls the penalty for misclassification. A low C creates a 'soft margin' that allows some errors, which prevents overfitting. This results in high bias but low variance.") as tracker:
            self.play(Write(insight_desc[0]))
            self.wait(1.5)

        with self.voiceover(text="Conversely, a high C creates a 'hard margin' that strictly punishes errors. It will wiggle the boundary aggressively to classify everything perfectly, leading to low bias but high variance, which means it might overfit to the training data.") as tracker:
            self.play(Write(insight_desc[1]))
            self.wait(1.5)

        with self.voiceover(text="Understanding this tradeoff, and how to tune the kernel trick alongside the C parameter, proves you truly understand the mechanics of the algorithm.") as tracker:
            self.wait(2.0)

        # Conclusion
        with self.voiceover(text="That concludes our deep dive into Support Vector Machines and the Kernel trick. Excellent work making it to day twenty four. See you in the next lesson!") as tracker:
            self.play(FadeOut(VGroup(title, box, insight_title, insight_desc)))
            self.wait(1.5)
