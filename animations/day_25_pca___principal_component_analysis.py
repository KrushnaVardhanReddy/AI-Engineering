import numpy as np
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class PCAAnimation(VoiceoverScene):
    def construct(self):
        # Setting up the whiteboard aesthetic
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # We need a 5-7 minute animation. This requires approx 750-1050 words of voiceover.
        # Let's break it down into 4 detailed sections.

        self.intro_section()
        self.what_is_it_section()
        self.why_do_we_need_it_section()
        self.use_cases_section()
        self.key_interview_insight_section()
        self.outro_section()

    def intro_section(self):
        title = Tex("Principal Component Analysis (PCA)", color=BLACK, font_size=60)
        subtitle = Tex("Day 25 - AI/ML Interview Prep", color=BLUE, font_size=40)

        group = VGroup(title, subtitle).arrange(DOWN, buff=0.5)

        with self.voiceover(text="Welcome back to Day 25 of our AI and Machine Learning Interview Prep series. Today, we are going to dive deep into a foundational algorithm in unsupervised learning and dimensionality reduction: Principal Component Analysis, or PCA for short. PCA is an extremely popular topic in technical interviews for data science and machine learning roles. It tests your geometric intuition, your understanding of linear algebra, and your grasp of how data variance impacts model performance. Over the next few minutes, we will break down exactly what PCA is, why we desperately need it when dealing with modern, high-dimensional datasets, some real-world use cases at major tech companies, and finally, the most common 'gotcha' questions that interviewers will throw at you. Let's get started.") as tracker:
            self.play(Write(title), run_time=1.5)
            self.play(FadeIn(subtitle), run_time=1)


        self.play(FadeOut(group))

    def what_is_it_section(self):
        section_title = Tex("What is PCA?", color=BLACK, font_size=50)
        section_title.to_edge(UP)

        definition = Tex(
            "An unsupervised technique that transforms high-dimensional data\\\\",
            "into a lower-dimensional space while preserving as much\\\\",
            "variance (or information) as possible.",
            color=BLACK, font_size=36
        )

        with self.voiceover(text="So, what exactly is Principal Component Analysis? At its core, PCA is an unsupervised dimensionality reduction technique. Its main goal is to take a dataset with a large number of features, or high-dimensional data, and transform it into a new, lower-dimensional space. However, it does this very carefully: it aims to preserve as much of the original data's variance, or information, as possible. By finding new axes, called principal components, PCA gives us a condensed summary of the data.") as tracker:
            self.play(Write(section_title))
            self.play(FadeIn(definition, shift=UP))


        self.play(FadeOut(definition))

        # Visualizing 2D to 1D PCA
        axes = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=6, y_length=6,
            axis_config={"color": BLACK}
        )

        # Generate some correlated points

        np.random.seed(42)
        x_vals = np.random.normal(0, 1, 30)
        y_vals = 0.8 * x_vals + np.random.normal(0, 0.4, 30)

        points = VGroup(*[Dot(axes.c2p(x, y), color=BLUE, radius=0.08) for x, y in zip(x_vals, y_vals)])

        with self.voiceover(text="Let's visualize this with a simple two-dimensional example. Imagine we have a dataset with two features, represented here by the X and Y axes. As you can see, the data points are highly correlated, forming an elongated shape. In two dimensions, this might not seem like a big deal, but imagine if this was thousands of dimensions.") as tracker:
            self.play(Create(axes))
            self.play(FadeIn(points))
            self.wait(1.5)

        pc1_line = Line(
            axes.c2p(-3, -2.4), axes.c2p(3, 2.4),
            color=RED, stroke_width=4
        )
        pc1_label = Tex("Principal Component 1", color=RED, font_size=24).next_to(pc1_line.get_end(), DOWN+RIGHT, buff=0.1)

        with self.voiceover(text="PCA analyzes this data and finds the direction where the data varies the most. This direction of maximum variance is our first Principal Component. We can draw a line through the data that best fits the spread. This red line represents Principal Component 1. It captures the vast majority of the information in our dataset.") as tracker:
            self.play(Create(pc1_line))
            self.play(FadeIn(pc1_label))
            self.wait(1.5)

        # Projections
        projections = VGroup()
        for x, y in zip(x_vals, y_vals):
            # The line is y = 0.8x.
            # Projection of (x,y) onto direction vector (1, 0.8)
            dot_prod = (x * 1 + y * 0.8)
            mag_sq = 1**2 + 0.8**2
            proj_x = dot_prod / mag_sq
            proj_y = 0.8 * proj_x

            proj_line = DashedLine(axes.c2p(x, y), axes.c2p(proj_x, proj_y), color=GRAY, stroke_width=2)
            projections.add(proj_line)

        with self.voiceover(text="To reduce our dimensions from two down to one, PCA projects all of our original data points orthogonally onto this new principal component axis. We are essentially squashing the data flat onto this red line. Notice how we lose a little bit of information—the small distances perpendicular to the line—but we retain the overall shape and spread of the data.") as tracker:
            self.play(Create(projections))


        self.play(FadeOut(VGroup(axes, points, pc1_line, pc1_label, projections, section_title)))

    def why_do_we_need_it_section(self):
        section_title = Tex("Why do we need PCA?", color=BLACK, font_size=50)
        section_title.to_edge(UP)

        curse_text = Tex("The Curse of Dimensionality", color=RED, font_size=40)

        with self.voiceover(text="So, why do we actually need PCA in practice? The primary reason is a phenomenon known in machine learning as the Curse of Dimensionality. As the number of features in our dataset grows, the amount of data we need to generalize accurately grows exponentially. High dimensional space is incredibly sparse, which makes distance-based algorithms, like K-Nearest Neighbors, perform very poorly.") as tracker:
            self.play(Write(section_title))
            self.play(FadeIn(curse_text))


        self.play(curse_text.animate.shift(UP*1.5))

        before_text = Tex("Before PCA: 10,000 Features", color=BLACK, font_size=30).shift(LEFT*3 + UP*0.5)
        after_text = Tex("After PCA: 50 Features", color=BLACK, font_size=30).shift(RIGHT*3 + UP*0.5)

        before_box = Rectangle(width=4, height=3, color=GRAY, fill_opacity=0.2).next_to(before_text, DOWN)
        after_box = Rectangle(width=2, height=3, color=BLUE, fill_opacity=0.2).next_to(after_text, DOWN)

        arrow = Arrow(before_box.get_right(), after_box.get_left(), color=BLACK, buff=0.2)

        noise_text = Tex("Noisy, Slow, Overfitting", color=RED, font_size=24).move_to(before_box)
        signal_text = Tex("Fast, Generalizes", color=GREEN, font_size=24).move_to(after_box)

        with self.voiceover(text="Consider a scenario where you are working with high-resolution image data or text embeddings, resulting in datasets with 10,000 or more features. Training a model on this directly is computationally expensive, prone to overfitting, and filled with noise.") as tracker:
            self.play(FadeIn(before_text), Create(before_box), Write(noise_text))
            self.wait(1.5)

        with self.voiceover(text="By applying PCA, we can extract just the top principal components—say, 50 features—that explain 95 percent of the variance. We strip away the noise and redundant features, leaving a dense, highly informative representation. This makes our models train vastly faster, require less memory, and often, generalize better to unseen data.") as tracker:
            self.play(Create(arrow))
            self.play(FadeIn(after_text), Create(after_box), Write(signal_text))


        self.play(FadeOut(VGroup(section_title, curse_text, before_text, after_text, before_box, after_box, arrow, noise_text, signal_text)))

    def use_cases_section(self):
        section_title = Tex("Real-World Use Cases", color=BLACK, font_size=50)
        section_title.to_edge(UP)

        uc1 = Tex("1. Finance (e.g., Bloomberg, Citadel)", color=BLUE, font_size=36)
        uc1_desc = Tex("Identifying overarching market trends from thousands of stock tickers.", color=BLACK, font_size=28)

        uc2 = Tex("2. Bioinformatics (e.g., 23andMe)", color=PURPLE, font_size=36)
        uc2_desc = Tex("Compressing genomic sequencing data to cluster genetic variations.", color=BLACK, font_size=28)

        uc1_group = VGroup(uc1, uc1_desc).arrange(DOWN, aligned_edge=LEFT).shift(UP*1 + LEFT*2)
        uc2_group = VGroup(uc2, uc2_desc).arrange(DOWN, aligned_edge=LEFT).shift(DOWN*1.5 + LEFT*2)

        with self.voiceover(text="Let's look at a couple of real-world use cases where PCA is a critical part of the data pipeline.") as tracker:
            self.play(Write(section_title))
            self.wait(0.5)

        with self.voiceover(text="First, in quantitative finance, firms like Bloomberg or Citadel use PCA for risk management and portfolio optimization. They might have daily returns for thousands of individual stock tickers. PCA helps them extract a few underlying 'market factors' or trends that drive the majority of the market's movement, drastically simplifying their forecasting models.") as tracker:
            self.play(FadeIn(uc1_group, shift=RIGHT))
            self.wait(1.5)

        with self.voiceover(text="Second, in bioinformatics, companies like 23andMe deal with genomic sequencing data, which can have millions of single-nucleotide polymorphisms, or SNPs. PCA is heavily used to compress this massive genomic data down to two or three dimensions, allowing researchers to visualize population clusters and identify genetic variations tied to ancestry or disease.") as tracker:
            self.play(FadeIn(uc2_group, shift=RIGHT))


        self.play(FadeOut(VGroup(section_title, uc1_group, uc2_group)))

    def key_interview_insight_section(self):
        section_title = Tex("Key Interview Insight", color=BLACK, font_size=50)
        section_title.to_edge(UP)

        box = Rectangle(width=10, height=4, color=RED, fill_color=WHITE, fill_opacity=1, stroke_width=4)

        warning_text = Tex("Always scale your data before applying PCA!", color=RED, font_size=40)
        sub_text1 = Tex("PCA is sensitive to the scale of the features.", color=BLACK, font_size=30)
        sub_text2 = Tex("Features with larger ranges will dominate the variance calculation.", color=BLACK, font_size=30)

        content = VGroup(warning_text, sub_text1, sub_text2).arrange(DOWN, buff=0.4)
        box_group = VGroup(box, content)

        with self.voiceover(text="Now for the most important part of this lesson: The Key Interview Insight. If you mention PCA in an interview, the interviewer is almost guaranteed to ask you one specific question to test your practical experience.") as tracker:
            self.play(Write(section_title))
            self.wait(0.5)

        with self.voiceover(text="And that question is: 'What must you do to your data before applying PCA?' The answer is: You must always scale or standardize your data first! Usually, this means using a Standard Scaler to give every feature a mean of zero and a variance of one.") as tracker:
            self.play(Create(box))
            self.play(Write(warning_text))
            self.wait(1.5)

        with self.voiceover(text="Why? Because PCA looks for features with the highest variance. If you don't scale your data, a feature measured in millions, like a house price, will have a massive variance compared to a feature measured in fractions, like an interest rate. PCA will incorrectly assume the house price is the most important feature simply because its numerical scale is larger, completely ignoring the true underlying structure of the data.") as tracker:
            self.play(FadeIn(sub_text1))
            self.play(FadeIn(sub_text2))
            self.wait(1.5)

        # Math derivation showing variance calculation line by line
        math_title = Tex("Variance Calculation", color=BLACK, font_size=36).shift(DOWN*2.5)
        eq1 = MathTex(r"\Sigma", r"=", r"\text{Cov}(X)", color=BLACK, font_size=36).next_to(math_title, DOWN)
        eq2 = MathTex(r"\Sigma", r"=", r"\frac{1}{n-1}", r"X^T", r"X", color=BLACK, font_size=36).next_to(math_title, DOWN)

        with self.voiceover(text="Mathematically, PCA computes the eigenvectors of the data's covariance matrix.") as tracker:
            self.play(FadeIn(math_title))
            self.play(Write(eq1))

        self.wait(1.5)

        with self.voiceover(text="Which is calculated as one over n minus one, times X transpose X.") as tracker:
            self.play(TransformMatchingTex(eq1, eq2))

        self.wait(1.5)

        with self.voiceover(text="If the matrix X is not standardized, the covariance matrix Sigma will be skewed by the arbitrary units of your measurements. Always remember this tradeoff: PCA is not scale-invariant.") as tracker:
            self.wait(1)

        self.wait(1.5)

        self.play(FadeOut(VGroup(section_title, box_group, math_title, eq2)))

    def outro_section(self):
        text1 = Tex("PCA Summary", color=BLACK, font_size=50)
        text2 = Tex("1. Reduces dimensions", color=BLUE, font_size=36)
        text3 = Tex("2. Preserves variance", color=BLUE, font_size=36)
        text4 = Tex("3. Requires feature scaling", color=RED, font_size=36)

        group = VGroup(text1, text2, text3, text4).arrange(DOWN, buff=0.5)

        with self.voiceover(text="To summarize our deep dive on Principal Component Analysis: It is a powerful unsupervised tool that reduces the dimensions of your data, fights the curse of dimensionality, and preserves as much variance as possible. But it strictly requires feature scaling to work correctly. Practice explaining these concepts, and you will ace your next machine learning interview. Thanks for watching, and see you in the next lesson!") as tracker:
            self.play(Write(text1))
            self.play(FadeIn(text2, shift=UP))
            self.play(FadeIn(text3, shift=UP))
            self.play(FadeIn(text4, shift=UP))
            self.wait(3)
