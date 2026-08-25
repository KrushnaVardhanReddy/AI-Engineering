from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class DecisionTreesRecursiveSplitting(VoiceoverScene):
    def construct(self):
        # Set up audio service and background
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # Introduction
        title = Text("Decision Trees & Recursive Splitting", color=BLACK).scale(1.2)
        self.play(Write(title))
        self.wait(1.5)

        # Section 1: What is it?
        with self.voiceover(text="Welcome to Day 22 of AI Engineering. Today we are exploring one of the foundational building blocks of modern machine learning: Decision Trees and the algorithm that builds them, known as Recursive Splitting. We will dive deep into how they work, why they are used, some real-world use cases, and finally, a key insight you need for interviews.") as tracker:
            self.play(title.animate.to_edge(UP))
            self.wait(1)

        with self.voiceover(text="So, what exactly is a Decision Tree? At its core, a decision tree is a supervised machine learning model that makes predictions by asking a series of sequential, yes-or-no questions about the features of your data. It looks and acts much like a flowchart.") as tracker:
            definition_title = Text("What is it?", color=BLUE).scale(0.8).next_to(title, DOWN, buff=0.5)
            definition_text = Text("A model that predicts by asking sequential yes-or-no questions.", color=BLACK).scale(0.6).next_to(definition_title, DOWN)
            self.play(FadeIn(definition_title))
            self.play(Write(definition_text))
            self.wait(1.5)

        with self.voiceover(text="Let's look at a concrete, visual example. Imagine we want to predict whether we should play tennis outside today, based on the current weather conditions.") as tracker:
            root_node = Circle(radius=0.8, color=BLACK, fill_color=WHITE, fill_opacity=1)
            root_text = Text("Outlook?", color=BLACK).scale(0.5).move_to(root_node.get_center())
            root_group = VGroup(root_node, root_text).move_to(UP * 0.5)

            self.play(FadeOut(definition_title, definition_text))
            self.play(FadeIn(root_group))
            self.wait(1.5)

        with self.voiceover(text="The tree begins at the root node. To build this tree from our dataset, we use a technique called recursive splitting. We evaluate all possible features, like outlook, humidity, or wind, and we split the data based on the feature that gives us the most information gain, separating the positive cases from the negative cases as cleanly as possible. Here, we split on whether the outlook is Sunny or Rainy.") as tracker:
            left_arrow = Arrow(root_node.get_bottom(), root_node.get_bottom() + DOWN + LEFT*2, color=BLACK, buff=0)
            right_arrow = Arrow(root_node.get_bottom(), root_node.get_bottom() + DOWN + RIGHT*2, color=BLACK, buff=0)

            left_text = Text("Sunny", color=BLACK).scale(0.4).next_to(left_arrow, LEFT)
            right_text = Text("Rainy", color=BLACK).scale(0.4).next_to(right_arrow, RIGHT)

            left_node = Circle(radius=0.8, color=BLACK, fill_color=WHITE, fill_opacity=1).next_to(left_arrow.get_end(), DOWN, buff=0)
            left_node_text = Text("Humidity?", color=BLACK).scale(0.5).move_to(left_node.get_center())
            left_group = VGroup(left_node, left_node_text)

            right_node = Circle(radius=0.8, color=BLACK, fill_color=WHITE, fill_opacity=1).next_to(right_arrow.get_end(), DOWN, buff=0)
            right_node_text = Text("Windy?", color=BLACK).scale(0.5).move_to(right_node.get_center())
            right_group = VGroup(right_node, right_node_text)

            self.play(Write(left_arrow), Write(right_arrow), Write(left_text), Write(right_text))
            self.play(FadeIn(left_group), FadeIn(right_group))
            self.wait(1.5)

        with self.voiceover(text="We continue this process recursively. At each new node, we split the remaining subset of data again and again, until we reach a final prediction, known as a leaf node. To decide which feature is the best to split on at any step, the algorithm uses mathematical metrics to measure node purity, such as Entropy.") as tracker:
            # Show math formula for Entropy
            entropy_formula_1 = MathTex(
                "H(S) = -\\sum p_i \\log_2 (p_i)",
                color=BLACK
            ).scale(0.8).to_edge(DOWN)
            entropy_formula_1.set_color_by_tex("p_i", RED)
            self.play(Write(entropy_formula_1))
            self.wait(1.5)

        with self.voiceover(text="Let's derive how this entropy formula comes together line by line. We start by determining the probability of a specific class, represented as p sub i. We then take the base 2 logarithm of that probability to measure the surprise or information content.") as tracker:
            eq1 = MathTex("p_i", "\\log_2(p_i)", color=BLACK).scale(0.8).to_edge(DOWN)
            eq1.set_color_by_tex("p_i", RED)
            self.play(TransformMatchingTex(entropy_formula_1, eq1))
            self.wait(1.5)

        with self.voiceover(text="Next, we calculate the expected value by multiplying this information content by the probability of the class itself.") as tracker:
            eq2 = MathTex("p_i", "\\cdot", "p_i", "\\log_2(p_i)", color=BLACK).scale(0.8).to_edge(DOWN)
            eq2.set_color_by_tex("p_i", RED)
            self.play(TransformMatchingTex(eq1, eq2))
            self.wait(1.5)

        with self.voiceover(text="Finally, we sum this expected value across all classes, and multiply by negative one to ensure our entropy value is positive, completing our formula for H of S.") as tracker:
            eq3 = MathTex("H(S)", "=", "-", "\\sum", "p_i", "\\log_2(p_i)", color=BLACK).scale(0.8).to_edge(DOWN)
            eq3.set_color_by_tex("p_i", RED)
            self.play(TransformMatchingTex(eq2, eq3))
            self.wait(1.5)

        self.play(FadeOut(root_group, left_arrow, right_arrow, left_text, right_text, left_group, right_group, eq3))
        self.wait(0.5)

        # Section 2: Why do we need it?
        with self.voiceover(text="Now that we understand what a decision tree is and how recursive splitting works, we have to ask: why do we need them? What specific problem do they solve that simpler models do not?") as tracker:
            why_title = Text("Why do we need it?", color=BLUE).scale(0.8).next_to(title, DOWN, buff=0.5)
            self.play(FadeIn(why_title))
            self.wait(1.5)

        with self.voiceover(text="Consider a dataset that has a complex, non-linear relationship. Let's look at this scatter plot where our positive green points are mixed around our negative red points.") as tracker:
            axes = Axes(x_range=[0, 10], y_range=[0, 10], x_length=5, y_length=5, axis_config={"color": BLACK}).shift(DOWN*0.5)

            # Points for non-linear boundary
            points_red = VGroup(*[Dot(axes.c2p(x, y), color=RED) for x, y in [(2,2), (2,8), (3,3), (4,7), (1,5)]])
            points_green = VGroup(*[Dot(axes.c2p(x, y), color=GREEN) for x, y in [(8,2), (7,8), (6,4), (9,6), (8,5), (7,2)]])

            self.play(Write(axes))
            self.play(FadeIn(points_red), FadeIn(points_green))
            self.wait(1.5)

        with self.voiceover(text="If we try to use a linear model, like linear regression or logistic regression, it will attempt to draw a single straight line through the data to separate the classes.") as tracker:
            linear_line = Line(axes.c2p(1, 9), axes.c2p(9, 1), color=PURPLE)
            self.play(Write(linear_line))
            self.wait(1.5)

        with self.voiceover(text="As you can see, this simple straight line fails completely. It misclassifies several points because the true boundary between these classes is not linear.") as tracker:
            self.play(Wiggle(linear_line))
            self.wait(1.5)
            self.play(FadeOut(linear_line))

        with self.voiceover(text="This is exactly where decision trees shine. Instead of a single line, decision trees divide the space into smaller, distinct rectangular regions using recursive binary splits. By repeatedly splitting the space horizontally and vertically, the tree can capture incredibly complex, non-linear patterns perfectly.") as tracker:
            split1 = Line(axes.c2p(5, 0), axes.c2p(5, 10), color=BLUE)
            split2 = Line(axes.c2p(0, 5), axes.c2p(5, 5), color=BLUE)
            split3 = Line(axes.c2p(5, 6), axes.c2p(10, 6), color=BLUE)

            self.play(Write(split1))
            self.wait(0.5)
            self.play(Write(split2))
            self.wait(0.5)
            self.play(Write(split3))
            self.wait(1.5)

        self.play(FadeOut(why_title, axes, points_red, points_green, split1, split2, split3))
        self.wait(0.5)

        # Section 3: Use Cases
        with self.voiceover(text="So, where do we see decision trees used in the real world?") as tracker:
            use_cases_title = Text("Use Cases", color=BLUE).scale(0.8).next_to(title, DOWN, buff=0.5)
            self.play(FadeIn(use_cases_title))
            self.wait(1.5)

        with self.voiceover(text="First, while single decision trees are useful, they are most powerful when combined into ensembles. Random Forests and XGBoost models are ensembles of hundreds of decision trees. These are widely used by companies like Spotify to power their complex music recommendation systems, handling millions of non-linear features.") as tracker:
            spotify_text = Text("1. Spotify: Music Recommendations (XGBoost/Random Forest)", color=BLACK).scale(0.5).shift(UP*0.5)
            self.play(Write(spotify_text))
            self.wait(2)

        with self.voiceover(text="Second, because standard decision trees are so easy to visualize and explain, financial institutions like Capital One use them extensively for credit scoring and fraud detection. When a loan is denied, regulators require the model to be interpretable, and decision trees provide that clear, step-by-step logic.") as tracker:
            finance_text = Text("2. Capital One: Interpretable Fraud Detection", color=BLACK).scale(0.5).next_to(spotify_text, DOWN, buff=0.5)
            self.play(Write(finance_text))
            self.wait(2)

        self.play(FadeOut(use_cases_title, spotify_text, finance_text))
        self.wait(0.5)

        # Section 4: Key Interview Insight
        with self.voiceover(text="Finally, let's look at the key insight interviewers will absolutely test you on.") as tracker:
            insight_title = Text("Key Interview Insight", color=RED).scale(0.8).next_to(title, DOWN, buff=0.5)
            self.play(FadeIn(insight_title))
            self.wait(1.5)

        with self.voiceover(text="The most common pitfall, and the biggest weakness of decision trees, is their massive tendency to overfit the training data.") as tracker:
            overfit_text = Text("Gotcha: Overfitting!", color=BLACK).scale(0.7).shift(UP*0.5)
            self.play(Write(overfit_text))
            self.wait(1.5)

        with self.voiceover(text="Because the algorithm will naturally just keep recursively splitting the data, if you do not strictly limit the depth of the tree, it will split all the way down until it perfectly memorizes every single training point. It essentially memorizes the noise in your data, leading to a massive tree that will completely fail to generalize to new, unseen data.") as tracker:
            box = SurroundingRectangle(overfit_text, color=RED, buff=0.3)
            self.play(Write(box))

            detail_text = Text("Without pruning or max_depth limits,\nthe tree memorizes noise in the data.", color=BLACK).scale(0.5).next_to(box, DOWN, buff=0.5)
            self.play(FadeIn(detail_text))
            self.wait(2)

        with self.voiceover(text="Interviewers want to see that you know how to combat this problem. You must mention setting hyper-parameters like max depth or minimum samples per split to prune the tree. Better yet, you can explain that using ensemble methods, like a Random Forest, leverages bagging to reduce this high variance and prevent overfitting.") as tracker:
            solution_text = Text("Solution: Set max_depth, min_samples_split, or use Ensembles.", color=GREEN).scale(0.5).next_to(detail_text, DOWN, buff=0.5)
            self.play(Write(solution_text))
            self.wait(2.5)

        with self.voiceover(text="Understanding this tradeoff between the flexibility of recursive splits and the danger of overfitting is key to mastering tree-based models. That concludes our deep dive into Decision Trees. Thanks for watching, and keep preparing for those AI engineering interviews.") as tracker:
            self.play(FadeOut(title, insight_title, overfit_text, box, detail_text, solution_text))
            self.wait(2)
