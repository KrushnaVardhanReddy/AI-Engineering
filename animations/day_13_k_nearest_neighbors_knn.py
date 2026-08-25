from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np

class KNNAnimation(VoiceoverScene):
    def construct(self):
        # Setup aesthetics
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Title Screen
        title = Text("K-Nearest Neighbors (KNN)", color=BLACK, font_size=54)
        subtitle = Text("Machine Learning Fundamentals", color=BLUE, font_size=36).next_to(title, DOWN)

        with self.voiceover(text="Welcome to our artificial intelligence and machine learning fundamentals series. Today, we are exploring one of the most intuitive and foundational algorithms in the field of machine learning: K-Nearest Neighbors, commonly referred to as K N N.") as tracker:
            self.play(Write(title))
            self.play(Write(subtitle))
            self.wait(1.5)

        self.play(FadeOut(title), FadeOut(subtitle))

        # Section 1: What is it?
        title1 = Text("What is K-Nearest Neighbors?", color=BLACK, font_size=48).to_edge(UP)
        self.play(Write(title1))

        with self.voiceover(text="So, what exactly is K-Nearest Neighbors? In a single sentence, K-Nearest Neighbors is a simple, supervised machine learning algorithm that classifies a new, unknown data point based on the majority class of its 'K' closest data points in the feature space. Essentially, it tells you that 'birds of a feather flock together.' If an unknown point is surrounded by a certain category, it most likely belongs to that category too.") as tracker:
            definition_text = Text(
                "Classifies a new data point based on the majority class\n"
                "of its 'K' closest neighbors.",
                color=BLACK, font_size=32, line_spacing=1.5
            ).next_to(title1, DOWN, buff=0.5)
            self.play(Write(definition_text))
            self.wait(1.5)

        with self.voiceover(text="Let's break this down visually. Imagine a two-dimensional feature space where we have already classified some data. Here, we plot two distinct classes of data: the blue squares representing one category, and the red triangles representing another category.") as tracker:
            self.play(FadeOut(definition_text))

            # Scatter plot for KNN
            axes = Axes(
                x_range=[0, 10, 1],
                y_range=[0, 10, 1],
                axis_config={"color": BLACK},
                x_length=7,
                y_length=5
            ).shift(DOWN * 0.5)

            # Class A (Blue squares)
            class_a_coords = [(2, 3), (3, 2), (3, 4), (4, 3), (2, 4)]
            class_a = VGroup(*[Square(side_length=0.2, color=BLUE, fill_opacity=0.8).move_to(axes.c2p(x, y)) for x, y in class_a_coords])

            # Class B (Red triangles)
            class_b_coords = [(7, 7), (8, 6), (7, 8), (6, 7), (8, 8)]
            class_b = VGroup(*[Triangle(color=RED, fill_opacity=0.8).scale(0.15).move_to(axes.c2p(x, y)) for x, y in class_b_coords])

            self.play(Create(axes))
            self.play(FadeIn(class_a), FadeIn(class_b))
            self.wait(1.5)

        with self.voiceover(text="Now, suppose we receive a new, unclassified data point. We don't know what category it belongs to. We place this unknown query point in our feature space. Let's represent it as a green circle. We need a robust, mathematical way to determine its identity.") as tracker:
            new_point_coord = (5, 5)
            new_point = Circle(radius=0.15, color=GREEN, fill_opacity=1).move_to(axes.c2p(*new_point_coord))
            new_label = Text("?", color=WHITE, font_size=20).move_to(new_point)
            self.play(FadeIn(new_point), Write(new_label))
            self.wait(2)

        with self.voiceover(text="To classify this point using K N N, we first need to define 'K', which is simply the number of neighbors we want to look at. Let's set K equal to 3. The algorithm then calculates the mathematical distance between our new point and every other point in the dataset. Typically, this is done using the Euclidean distance formula, derived directly from the Pythagorean theorem. Let's quickly review how this fundamental distance metric is calculated in two-dimensional space.") as tracker:
            k_text = MathTex("K = 3", color=BLACK, font_size=42).to_edge(RIGHT).shift(UP * 2.5)
            self.play(Write(k_text))
            self.wait(1)

            # Euclidean Distance Math Derivation
            dist_title = Text("Euclidean Distance", color=PURPLE, font_size=24).next_to(k_text, DOWN, buff=0.5)
            self.play(Write(dist_title))

            eq1 = MathTex("d(p, q)^2", "=", "(q_1 - p_1)^2 + (q_2 - p_2)^2", color=BLACK).scale(0.6).next_to(dist_title, DOWN, buff=0.2)
            eq2 = MathTex("d(p, q)", "=", "\\sqrt{(q_1 - p_1)^2 + (q_2 - p_2)^2}", color=BLACK).scale(0.6).next_to(dist_title, DOWN, buff=0.2)

            self.play(Write(eq1))
            self.wait(1.5)
            self.play(TransformMatchingTex(eq1, eq2))
            self.wait(2)

            # Calculate distances and draw lines
            all_coords = class_a_coords + class_b_coords
            all_points = list(class_a) + list(class_b)

            distances = [np.sqrt((x - new_point_coord[0])**2 + (y - new_point_coord[1])**2) for x, y in all_coords]
            sorted_indices = np.argsort(distances)
            nearest_indices = sorted_indices[:3]

            lines = VGroup()
            for idx in nearest_indices:
                pt = all_points[idx]
                line = DashedLine(new_point.get_center(), pt.get_center(), color=GREEN)
                lines.add(line)

            circle_radius = distances[nearest_indices[2]]
            search_circle = Circle(
                radius=circle_radius * (axes.x_length / 10),
                color=GREEN,
                stroke_width=2
            ).move_to(new_point.get_center())

            self.play(Create(search_circle))
            self.play(Create(lines))
            self.wait(1.5)

        with self.voiceover(text="By drawing a search radius that sequentially captures the three nearest points based on our calculated distances, we can observe the local neighborhood. We have two blue squares and one red triangle. Since the blue squares hold the absolute majority among these three nearest neighbors, the algorithm confidently classifies our new, unknown green query point as a blue square. That is K N N in action: a simple, elegant, distance-based democratic voting process.") as tracker:
            vote_text = Text("Votes: 2 Blue, 1 Red", color=BLACK, font_size=28).next_to(eq2, DOWN, buff=0.5)
            result_text = Text("Result: Class Blue", color=BLUE, font_size=28).next_to(vote_text, DOWN)

            self.play(Write(vote_text))
            self.wait(1)
            self.play(Write(result_text))
            self.wait(1)

            final_square = Square(side_length=0.2, color=BLUE, fill_opacity=0.8).move_to(new_point.get_center())
            self.play(FadeOut(new_point), FadeOut(new_label), FadeIn(final_square))
            self.wait(2.5)

        self.play(
            FadeOut(title1), FadeOut(axes), FadeOut(class_a), FadeOut(class_b),
            FadeOut(final_square), FadeOut(search_circle), FadeOut(lines),
            FadeOut(k_text), FadeOut(dist_title), FadeOut(eq2), FadeOut(vote_text), FadeOut(result_text)
        )

        # Section 2: Why do we need it?
        title2 = Text("Why Do We Need It?", color=BLACK, font_size=48).to_edge(UP)
        self.play(Write(title2))

        with self.voiceover(text="Why do we need K-Nearest Neighbors? Consider a scenario where you have a vast amount of historical data, but no explicit rules to categorize new items. Without K N N, a new data point is just a meaningless coordinate in isolation. It completely lacks context. You would have to manually write complex, hard-coded rules to try and categorize it.") as tracker:
            before_text = Text("Without KNN: Isolated and undefined", color=RED, font_size=32).shift(UP * 2)
            self.play(Write(before_text))

            isolated_point = Circle(radius=0.2, color=GRAY, fill_opacity=1)
            isolated_label = Text("?", color=WHITE, font_size=24).move_to(isolated_point)
            question_marks = VGroup(*[Text("?", color=RED, font_size=40).move_to(isolated_point.get_center() + np.array([np.cos(a), np.sin(a), 0]) * 1.5) for a in np.linspace(0, 2 * np.pi, 5, endpoint=False)])

            self.play(FadeIn(isolated_point), Write(isolated_label))
            self.play(Write(question_marks))
            self.wait(1.5)

        with self.voiceover(text="However, with K N N, we leverage the intrinsic structure of the data itself. We don't need a complex pre-trained mathematical model with millions of parameters. We simply use the data as the model. K N N dynamically creates localized decision boundaries around the data, perfectly adapting to non-linear and irregular shapes in the feature space. It solves the problem of categorization simply by looking at the neighborhood.") as tracker:
            self.play(FadeOut(before_text), FadeOut(isolated_point), FadeOut(isolated_label), FadeOut(question_marks))

            after_text = Text("With KNN: Localized, adaptive boundaries", color=GREEN, font_size=32).shift(UP * 2)
            self.play(Write(after_text))

            # Show a decision boundary idea
            bg_axes = Axes(x_range=[-3, 3, 1], y_range=[-2, 2, 1], x_length=6, y_length=4, axis_config={"color": BLACK}).shift(DOWN*0.5)

            # Generate random points for two classes
            np.random.seed(42)
            pts_a = np.random.uniform(low=[-2, -1], high=[0, 1], size=(20, 2))
            pts_b = np.random.uniform(low=[0, -1], high=[2, 1], size=(20, 2))

            dots_a = VGroup(*[Dot(bg_axes.c2p(x, y), color=BLUE) for x, y in pts_a])
            dots_b = VGroup(*[Dot(bg_axes.c2p(x, y), color=RED) for x, y in pts_b])

            boundary = FunctionGraph(lambda x: np.sin(x*2)*0.5, x_range=[-2, 2], color=PURPLE).shift(DOWN*0.5)
            boundary_label = Text("Non-linear Decision Boundary", color=PURPLE, font_size=24).next_to(boundary, UP)

            self.play(Create(bg_axes))
            self.play(FadeIn(dots_a), FadeIn(dots_b))
            self.wait(0.5)
            self.play(Create(boundary))
            self.play(Write(boundary_label))
            self.wait(1.5)

        self.play(FadeOut(title2), FadeOut(after_text), FadeOut(bg_axes), FadeOut(dots_a), FadeOut(dots_b), FadeOut(boundary), FadeOut(boundary_label))

        # Section 3: Use Cases
        title3 = Text("Real-World Use Cases", color=BLACK, font_size=48).to_edge(UP)
        self.play(Write(title3))

        with self.voiceover(text="So, where is this algorithm used in the real world? Despite its simplicity, it is incredibly powerful for specific tasks. For instance, consider early recommendation systems, like those used by Netflix or Amazon. If you enjoyed a specific set of movies, K N N can find other users who liked those same movies—your 'nearest neighbors'—and recommend what they watched next.") as tracker:
            case1_title = Text("1. Recommendation Systems (Netflix/Amazon)", color=BLUE, font_size=32).shift(UP * 1.5)
            self.play(Write(case1_title))

            user_icon = Circle(radius=0.4, color=BLACK).shift(LEFT * 3)
            user_text = Text("You", color=BLACK, font_size=24).next_to(user_icon, DOWN)
            neighbor_icons = VGroup(
                Circle(radius=0.4, color=BLUE).shift(RIGHT * 3 + UP),
                Circle(radius=0.4, color=BLUE).shift(RIGHT * 3),
                Circle(radius=0.4, color=BLUE).shift(RIGHT * 3 + DOWN)
            )
            neighbor_text = Text("Similar Users", color=BLUE, font_size=24).next_to(neighbor_icons, DOWN)

            links = VGroup(*[DashedLine(user_icon.get_right(), n.get_left(), color=GRAY) for n in neighbor_icons])

            self.play(FadeIn(user_icon), Write(user_text))
            self.play(FadeIn(neighbor_icons), Write(neighbor_text))
            self.play(Create(links))
            self.wait(1.5)

            self.play(FadeOut(case1_title), FadeOut(user_icon), FadeOut(user_text), FadeOut(neighbor_icons), FadeOut(neighbor_text), FadeOut(links))

        with self.voiceover(text="Another excellent, highly critical use case is Anomaly Detection, which is heavily utilized by massive financial institutions like JPMorgan Chase or Capital One for real-time credit card fraud detection. In their systems, the vast majority of your normal, everyday transactions cluster closely together in a high-dimensional feature space. If a sudden, massive transaction occurs halfway across the globe in a strange category, it will map to a coordinate that has very few, or zero, close neighbors from your history. By running K N N, the system immediately flags this isolated, distant point as a severe anomaly, thereby catching and blocking a fraudulent transaction in milliseconds.") as tracker:
            case2_title = Text("2. Fraud Detection (JPMorgan / Capital One)", color=RED, font_size=32).shift(UP * 1.5)
            self.play(Write(case2_title))

            cluster = VGroup(*[Dot(np.array([np.random.normal(-1, 0.5), np.random.normal(0, 0.5), 0]), color=BLACK) for _ in range(30)])
            cluster_label = Text("Normal Transactions", color=BLACK, font_size=24).next_to(cluster, DOWN)

            anomaly = Dot(np.array([4, 1.5, 0]), color=RED).scale(1.5)
            anomaly_label = Text("Isolated = Fraud?", color=RED, font_size=24).next_to(anomaly, UP)

            self.play(FadeIn(cluster), Write(cluster_label))
            self.wait(0.5)
            self.play(FadeIn(anomaly), Write(anomaly_label))
            self.wait(1.5)

        self.play(FadeOut(title3), FadeOut(case2_title), FadeOut(cluster), FadeOut(cluster_label), FadeOut(anomaly), FadeOut(anomaly_label))

        # Section 4: Key Interview Insight
        title4 = Text("Key Interview Insight", color=BLACK, font_size=48).to_edge(UP)
        self.play(Write(title4))

        with self.voiceover(text="Now, pay close attention, because this is the most common K N N question asked in technical interviews. Interviewers will invariably ask you: 'How do you choose the value of K, and what happens if K is too small or too large?' You must understand this tradeoff deeply.") as tracker:
            insight_box = Rectangle(width=12, height=5, color=PURPLE, fill_opacity=0.05).shift(DOWN * 0.5)
            insight_title = Text("The Tradeoff of 'K'", color=BLACK, font_size=36).next_to(insight_box.get_top(), DOWN, buff=0.3)

            self.play(Create(insight_box))
            self.play(Write(insight_title))
            self.wait(1.5)

        with self.voiceover(text="If you aggressively choose a value for K that is far too small—for example, if you set K equal to exactly 1—the algorithm becomes extraordinarily sensitive to statistical noise or tiny outliers in your dataset. It will stubbornly draw incredibly jagged, erratic decision boundaries that might perfectly fit your training data, but will fail miserably when introduced to new, unseen data. In machine learning terminology, this is the classic, textbook definition of high variance and severe overfitting.") as tracker:
            k_small_text = MathTex(r"K \to 1", color=RED, font_size=36).move_to(insight_box.get_left() + RIGHT * 3 + UP * 0.5)
            overfit_text = Text("Result: Overfitting (Sensitive to Noise)", color=BLACK, font_size=24).next_to(k_small_text, DOWN)

            self.play(Write(k_small_text))
            self.play(Write(overfit_text))
            self.wait(2)

        with self.voiceover(text="Conversely, if you choose a value for K that is far too large—for instance, setting K equal to N, which represents the entire size of your dataset—the algorithm collapses into uselessness. Because it looks at everything, it will simply always predict the overall global majority class of the dataset, entirely ignoring the rich, localized patterns that actually matter. The decision boundary becomes completely smooth, rigid, and ultimately blind. This is known as high bias, or underfitting. To nail your interview, make sure you confidently explain that you must actively find the optimal, Goldilocks value for K by using a robust technique called K-Fold Cross-Validation, plotting a validation error curve to scientifically discover the perfect middle ground.") as tracker:
            k_large_text = MathTex(r"K \to N", color=BLUE, font_size=36).move_to(insight_box.get_right() + LEFT * 3 + UP * 0.5)
            underfit_text = Text("Result: Underfitting (Ignores Local Patterns)", color=BLACK, font_size=24).next_to(k_large_text, DOWN)

            self.play(Write(k_large_text))
            self.play(Write(underfit_text))

            arrow = DoubleArrow(start=overfit_text.get_right() + RIGHT*0.2, end=underfit_text.get_left() + LEFT*0.2, color=BLACK)
            cross_val = Text("Use Cross-Validation to find the sweet spot!", color=PURPLE, font_size=28).next_to(arrow, DOWN, buff=1)

            self.play(Create(arrow))
            self.play(Write(cross_val))
            self.wait(2)

        self.play(FadeOut(title4), FadeOut(insight_box), FadeOut(insight_title), FadeOut(k_small_text), FadeOut(overfit_text), FadeOut(k_large_text), FadeOut(underfit_text), FadeOut(arrow), FadeOut(cross_val))

        # Outro
        with self.voiceover(text="That concludes our deep dive into K-Nearest Neighbors. A simple concept, yet fundamentally important. Master the basics, and you will build the future. Thanks for watching.") as tracker:
            outro_text = Text("Master the basics. Build the future.", color=BLACK, font_size=40)
            self.play(Write(outro_text))
            self.wait(2)
            self.play(FadeOut(outro_text))
