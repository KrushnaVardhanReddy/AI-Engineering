from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np


class RandomForestsAndBagging(VoiceoverScene):
    def construct(self):
        # Setup: Whiteboard Aesthetic
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Section 1: What is it?
        self.section_what_is_it()

        # Section 2: Why do we need it?
        self.section_why_do_we_need_it()

        # Section 3: Use Cases
        self.section_use_cases()

        # Section 4: Key Interview Insight
        self.section_interview_insight()

    def section_what_is_it(self):
        # 1. What is it? — A clear 1-sentence definition
        title = Text("Random Forests & Bagging", color=BLACK, font_size=48).to_edge(UP)

        with self.voiceover(text="Welcome to day 23. Today we are talking about Random Forests and Bagging.") as tracker:
            self.play(Write(title))
            self.wait(1.5)

        definition_text = Text(
            "An ensemble learning method that builds multiple decision\n"
            "trees and merges them together to get a more accurate\n"
            "and stable prediction.",
            color=BLACK,
            font_size=28,
            t2c={"ensemble learning": BLUE, "multiple decision\ntrees": GREEN, "stable prediction": PURPLE}
        ).next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="What is a Random Forest? It is an ensemble learning method that builds multiple decision trees and merges them together to get a more accurate and stable prediction.") as tracker:
            self.play(FadeIn(definition_text))
            self.wait(1.5)

        # Visualizing Bagging (Bootstrap Aggregating)
        bagging_title = Text("Bagging (Bootstrap Aggregating)", color=BLUE, font_size=36).next_to(definition_text, DOWN, buff=0.7)

        dataset = Rectangle(width=4, height=1, color=BLACK).set_fill(LIGHT_GRAY, opacity=0.3).next_to(bagging_title, DOWN, buff=0.5)
        dataset_label = Text("Original Dataset", color=BLACK, font_size=24).move_to(dataset.get_center())

        with self.voiceover(text="The core idea relies on Bagging, which stands for Bootstrap Aggregating. We start with our original dataset.") as tracker:
            self.play(Write(bagging_title), FadeIn(dataset), Write(dataset_label))
            self.wait(1.5)

        # Create bootstrapped samples
        samples = VGroup()
        arrows = VGroup()
        for i in range(3):
            sample = Rectangle(width=2, height=0.6, color=BLACK).set_fill(GREEN_A, opacity=0.3)
            sample_label = Text(f"Sample {i+1}", color=BLACK, font_size=20).move_to(sample.get_center())
            sample_group = VGroup(sample, sample_label)
            samples.add(sample_group)

        samples.arrange(RIGHT, buff=1).next_to(dataset, DOWN, buff=1)

        for sample in samples:
            arrow = Arrow(start=dataset.get_bottom(), end=sample.get_top(), color=BLACK, buff=0.1)
            arrows.add(arrow)

        with self.voiceover(text="We draw multiple random subsets from the original dataset with replacement. These are called bootstrap samples.") as tracker:
            self.play(FadeIn(samples, shift=DOWN), FadeIn(arrows))
            self.wait(1.5)

        # Create trees
        trees = VGroup()
        tree_arrows = VGroup()
        for i, sample in enumerate(samples):
            tree_text = Text(f"Tree {i+1}", color=GREEN, font_size=24)
            tree_circle = Circle(radius=0.5, color=GREEN).surround(tree_text)
            tree_group = VGroup(tree_circle, tree_text).next_to(sample, DOWN, buff=0.7)
            trees.add(tree_group)

            arrow = Arrow(start=sample.get_bottom(), end=tree_group.get_top(), color=BLACK, buff=0.1)
            tree_arrows.add(arrow)

        with self.voiceover(text="Then, we train a separate decision tree on each of these samples.") as tracker:
            self.play(FadeIn(trees, shift=DOWN), FadeIn(tree_arrows))
            self.wait(1.5)

        aggregation = Text("Majority Vote / Average", color=PURPLE, font_size=28)
        aggregation_box = SurroundingRectangle(aggregation, color=PURPLE, corner_radius=0.2)
        agg_group = VGroup(aggregation_box, aggregation).next_to(trees, DOWN, buff=1)

        final_arrows = VGroup()
        for tree in trees:
            arrow = Arrow(start=tree.get_bottom(), end=agg_group.get_top(), color=BLACK, buff=0.1)
            final_arrows.add(arrow)

        with self.voiceover(text="Finally, we aggregate their predictions. For classification, we use a majority vote. For regression, we average the outputs.") as tracker:
            self.play(FadeIn(final_arrows), FadeIn(agg_group, shift=UP))
            self.wait(2)

        self.play(FadeOut(VGroup(title, definition_text, bagging_title, dataset, dataset_label, samples, arrows, trees, tree_arrows, agg_group, final_arrows)))

    def section_why_do_we_need_it(self):
        # 2. Why do we need it?
        title = Text("Why do we need it?", color=BLACK, font_size=48).to_edge(UP)

        # Before: Single Decision Tree (Overfitting)
        before_title = Text("Single Decision Tree", color=RED, font_size=32).shift(LEFT * 3.5 + UP * 2)

        # Draw a complex, overfitting decision boundary
        axes_before = Axes(x_range=[0, 10, 2], y_range=[0, 10, 2], x_length=4, y_length=4,
                           axis_config={"color": BLACK}, tips=False).next_to(before_title, DOWN, buff=0.5)

        # Data points
        np.random.seed(42)
        points_blue = [(np.random.uniform(1, 4), np.random.uniform(1, 9)) for _ in range(15)]
        points_red = [(np.random.uniform(6, 9), np.random.uniform(1, 9)) for _ in range(15)]
        # Add some noise points to cause overfitting
        points_blue.extend([(7.5, 5), (6.5, 7)])
        points_red.extend([(2.5, 4), (3.5, 6)])

        dots_before = VGroup()
        for x, y in points_blue:
            dots_before.add(Dot(axes_before.c2p(x, y), color=BLUE))
        for x, y in points_red:
            dots_before.add(Dot(axes_before.c2p(x, y), color=RED))

        # Overfitted boundary (jagged line)
        boundary_before = axes_before.plot_line_graph(
            x_values=[0, 4.5, 4.5, 3.8, 3.8, 4.5, 4.5, 5.5, 5.5, 8.5, 8.5, 5.5, 5.5, 10],
            y_values=[10, 10, 6.5, 6.5, 5.5, 5.5, 4.5, 4.5, 3.5, 3.5, 2.5, 2.5, 0, 0],
            line_color=BLACK, add_vertex_dots=False
        )

        overfit_text = Text("High Variance (Overfitting)", color=RED, font_size=20).next_to(axes_before, DOWN)

        with self.voiceover(text="Why do we need Random Forests? A single decision tree tends to be very sensitive to noise in the training data, leading to a complex boundary and overfitting.") as tracker:
            self.play(Write(title))
            self.play(Write(before_title), FadeIn(axes_before), FadeIn(dots_before))
            self.play(Write(boundary_before))
            self.play(FadeIn(overfit_text))
            self.wait(1.5)

        # After: Random Forest (Smooth)
        after_title = Text("Random Forest", color=GREEN, font_size=32).shift(RIGHT * 3.5 + UP * 2)

        axes_after = Axes(x_range=[0, 10, 2], y_range=[0, 10, 2], x_length=4, y_length=4,
                          axis_config={"color": BLACK}, tips=False).next_to(after_title, DOWN, buff=0.5)

        dots_after = VGroup()
        for x, y in points_blue:
            dots_after.add(Dot(axes_after.c2p(x, y), color=BLUE))
        for x, y in points_red:
            dots_after.add(Dot(axes_after.c2p(x, y), color=RED))

        # Smooth boundary (generalized)
        boundary_after = axes_after.plot_line_graph(
            x_values=[0, 5, 5, 10],
            y_values=[10, 10, 0, 0],
            line_color=BLACK, add_vertex_dots=False
        )

        robust_text = Text("Low Variance (Robust)", color=GREEN, font_size=20).next_to(axes_after, DOWN)

        with self.voiceover(text="By aggregating many trees, a Random Forest smooths out these irregularities. It significantly reduces variance without increasing bias, giving us a much more robust model.") as tracker:
            self.play(Write(after_title), FadeIn(axes_after), FadeIn(dots_after))
            self.play(Write(boundary_after))
            self.play(FadeIn(robust_text))
            self.wait(2)

        self.play(FadeOut(VGroup(title, before_title, axes_before, dots_before, boundary_before, overfit_text,
                                 after_title, axes_after, dots_after, boundary_after, robust_text)))

    def section_use_cases(self):
        # 3. Use Cases
        title = Text("Real-World Use Cases", color=BLACK, font_size=48).to_edge(UP)

        with self.voiceover(text="Random Forests are incredibly versatile and are used widely across the industry.") as tracker:
            self.play(Write(title))
            self.wait(1)

        # Case 1
        case1_title = Text("1. Healthcare / Bioinformatics", color=BLUE, font_size=32).shift(UP * 1.5)
        case1_desc = Text(
            "Hospitals use Random Forests to predict patient disease\n"
            "risk based on medical records, thanks to its high accuracy\n"
            "and ability to handle missing data.",
            color=BLACK, font_size=24
        ).next_to(case1_title, DOWN, buff=0.5)

        with self.voiceover(text="For example, in Healthcare, hospitals use Random Forests to predict patient disease risk. It handles non-linear relationships and missing medical data exceptionally well.") as tracker:
            self.play(FadeIn(case1_title), FadeIn(case1_desc))
            self.wait(1.5)

        # Case 2
        case2_title = Text("2. Finance (e.g., Capital One, Stripe)", color=GREEN, font_size=32).next_to(case1_desc, DOWN, buff=1)
        case2_desc = Text(
            "Banks and payment processors use it for fraud detection.\n"
            "It can process thousands of transaction features quickly\n"
            "and identify unusual patterns reliably.",
            color=BLACK, font_size=24
        ).next_to(case2_title, DOWN, buff=0.5)

        with self.voiceover(text="In Finance, companies like Capital One and Stripe use them for fraud detection. The algorithm can process thousands of transaction features and identify unusual, fraudulent patterns efficiently.") as tracker:
            self.play(FadeIn(case2_title), FadeIn(case2_desc))
            self.wait(2)

        self.play(FadeOut(VGroup(title, case1_title, case1_desc, case2_title, case2_desc)))

    def section_interview_insight(self):
        # 4. Key Interview Insight
        title = Text("Key Interview Insight", color=BLACK, font_size=48).to_edge(UP)

        with self.voiceover(text="If you are interviewing for a machine learning role, pay close attention to this insight.") as tracker:
            self.play(Write(title))
            self.wait(1)

        insight_box_bg = Rectangle(width=10, height=5, color=RED).set_fill(RED_A, opacity=0.1)
        insight_box_bg.move_to(ORIGIN)

        gotcha_title = Text("The Feature Selection Gotcha", color=RED, font_size=36, weight=BOLD).move_to(insight_box_bg.get_top() + DOWN * 0.7)

        insight_text_1 = Text("Interviewers will ask:", color=BLACK, font_size=24, slant=ITALIC).next_to(gotcha_title, DOWN, buff=0.5)
        insight_text_2 = Text('"Why is it called a RANDOM Forest, not just Bagged Trees?"', color=BLACK, font_size=28, weight=BOLD).next_to(insight_text_1, DOWN, buff=0.3)

        with self.voiceover(text="Interviewers love to ask: Why is it called a Random Forest, and not just Bagged Trees?") as tracker:
            self.play(FadeIn(insight_box_bg), Write(gotcha_title))
            self.play(FadeIn(insight_text_1), FadeIn(insight_text_2))
            self.wait(1.5)

        insight_text_3 = Text("Answer: Feature Randomness", color=BLUE, font_size=28, weight=BOLD).next_to(insight_text_2, DOWN, buff=0.5)

        insight_text_4 = Text(
            "It doesn't just bootstrap the rows (data).\n"
            "It also randomly subsets the columns (features)\n"
            "at each split in the tree.",
            color=BLACK, font_size=24
        ).next_to(insight_text_3, DOWN, buff=0.3)

        with self.voiceover(text="The answer is Feature Randomness. A Random Forest doesn't just bootstrap the rows of data. It also randomly subsets the columns, or features, at each split.") as tracker:
            self.play(FadeIn(insight_text_3))
            self.play(Write(insight_text_4))
            self.wait(1.5)

        tradeoff_text = Text(
            "Tradeoff: This decorrelates the trees, preventing a single dominant\n"
            "feature from taking over, which dramatically reduces overall variance.",
            color=PURPLE, font_size=22
        ).next_to(insight_text_4, DOWN, buff=0.5)

        with self.voiceover(text="The tradeoff and benefit here is that this decorrelates the trees. It prevents a single highly predictive feature from dominating every tree, which dramatically reduces the overall variance of the model.") as tracker:
            self.play(FadeIn(tradeoff_text))
            self.wait(3)

        with self.voiceover(text="That concludes our deep dive into Random Forests and Bagging. See you tomorrow.") as tracker:
            self.play(FadeOut(VGroup(title, insight_box_bg, gotcha_title, insight_text_1, insight_text_2, insight_text_3, insight_text_4, tradeoff_text)))
            self.wait(1)
