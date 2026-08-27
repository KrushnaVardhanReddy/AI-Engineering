from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class DistanceMetricsScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # Introduction
        title = Text("When to Use Which Distance Metric", color=BLACK, font_size=48).to_edge(UP)
        with self.voiceover(text="Welcome to day five of our AI Engineering mastery series. Today, we are taking a deep, step-by-step mathematical dive into a fundamental topic: When to use which distance metric. In the realm of vector databases, natural language processing, and advanced machine learning models, choosing the right way to measure the distance between two high-dimensional vectors is absolutely critical for the performance, latency, and overall accuracy of your system. If you choose the wrong metric, your semantic search might return completely irrelevant results, or your clustering algorithm might fail to converge. We will thoroughly explore three primary distance metrics today: Euclidean Distance, Cosine Similarity, and Manhattan Distance. By the end of this session, you will know exactly how they are calculated, why they are used, where they are applied in production by top tier tech companies, and crucially, what interviewers are looking for when they ask you to compare them. Let us begin our deep dive.") as tracker:
            self.play(Write(title))
            self.wait(2)

        self.play(FadeOut(title))

        # 1. Euclidean Distance
        self.explain_euclidean()

        # 2. Cosine Similarity
        self.explain_cosine()

        # 3. Manhattan Distance
        self.explain_manhattan()

        # Conclusion
        conclusion = Text("Choose your metrics wisely!", color=BLACK).scale(1.2)
        with self.voiceover(text="To summarize our deep dive, the choice of distance metric depends entirely on the nature of your data, the dimensionality of your space, and the specific problem you are trying to solve. Use Euclidean distance when you are dealing with physical spatial dimensions or when exact magnitudes matter. Use Cosine similarity when you are dealing with high-dimensional text embeddings where orientation is more important than total length. And use Manhattan distance when dealing with grid-like routing structures or when you need a metric that is mathematically robust to extreme outliers. Knowing these tradeoffs inside and out will not only make you a better AI engineer, but will guarantee you ace your next machine learning system design interview. Thank you for watching, and see you in the next session.") as tracker:
            self.play(Write(conclusion))
            self.wait(3)
            self.play(FadeOut(conclusion))


    def explain_euclidean(self):
        section_title = Text("1. Euclidean Distance", color=BLUE, font_size=40).to_edge(UP)

        with self.voiceover(text="Let us start with Euclidean Distance. What exactly is it? Euclidean distance is the straight-line distance between two points in Euclidean space, calculated directly using the Pythagorean theorem.") as tracker:
            self.play(Write(section_title))

            # Definition Text
            definition = Text("The straight-line distance between two points, based on the Pythagorean theorem.", color=BLACK, font_size=20).next_to(section_title, DOWN)
            self.play(Write(definition))
            self.wait(1.5)
            self.play(FadeOut(definition))

            # Diagram & Math
            ax = Axes(x_range=[0, 5], y_range=[0, 5], x_length=4, y_length=4, axis_config={"color": BLACK}).shift(LEFT * 3 + DOWN * 0.5)
            p1 = ax.coords_to_point(1, 1)
            p2 = ax.coords_to_point(4, 3)

            dot1 = Dot(p1, color=BLUE)
            dot2 = Dot(p2, color=BLUE)
            label1 = MathTex("p_1(x_1, y_1)", color=BLACK).next_to(dot1, DOWN)
            label2 = MathTex("p_2(x_2, y_2)", color=BLACK).next_to(dot2, UP)

            line = Line(p1, p2, color=RED)

            self.play(Create(ax))
            self.play(FadeIn(dot1), FadeIn(label1), FadeIn(dot2), FadeIn(label2))
            self.play(Create(line))
            self.wait(1)

        with self.voiceover(text="Mathematically, we are trying to find the length of the red line. We do this by breaking it down into its horizontal and vertical components. Let's look at the equation.") as tracker:

            eq1 = MathTex(r"d(p, q) =", r"\sqrt{", r"(x_2 - x_1)^2", r"+", r"(y_2 - y_1)^2", r"}", color=BLACK).shift(RIGHT * 3 + UP * 1)

            self.play(Write(eq1[0]))
            self.wait(1)
            self.play(Write(eq1[1]), Write(eq1[5]))

        with self.voiceover(text="First, we take the difference in the x-coordinates, which represents the horizontal base of our right triangle. We square this difference to ensure it is positive and to follow the Pythagorean rule of A squared plus B squared equals C squared.") as tracker:
            self.play(Write(eq1[2]))
            self.wait(1)

        with self.voiceover(text="Then, we add the squared difference of the y-coordinates, which is the vertical height of our triangle.") as tracker:
            self.play(Write(eq1[3]))
            self.play(Write(eq1[4]))
            self.wait(1.5)

        with self.voiceover(text="While this equation shows a simple 2-dimensional space, in modern AI applications, vectors often have thousands of dimensions. So, we generalize this formula to N dimensions using summation notation.") as tracker:
            eq2 = MathTex(r"d(p, q) =", r"\sqrt{", r"\sum_{i=1}^{n}", r"(q_i - p_i)^2", r"}", color=BLACK).shift(RIGHT * 3 + UP * 1)
            self.play(TransformMatchingTex(eq1, eq2))
            self.wait(2)

        with self.voiceover(text="Why do we need it? Before using Euclidean distance, we might just look at absolute coordinate differences independently, but that completely ignores the true physical shortest path between them. It is essentially giving us an incomplete picture of the spatial relationship.") as tracker:
            dx_line = DashedLine(p1, ax.coords_to_point(4, 1), color=GREEN)
            dy_line = DashedLine(ax.coords_to_point(4, 1), p2, color=GREEN)

            self.play(Create(dx_line), Create(dy_line))
            self.wait(1)

            brace_dx = Brace(dx_line, direction=DOWN, color=BLACK)
            brace_dy = Brace(dy_line, direction=RIGHT, color=BLACK)

            dx_text = brace_dx.get_tex(r"\Delta x").set_color(BLACK)
            dy_text = brace_dy.get_tex(r"\Delta y").set_color(BLACK)

            self.play(FadeIn(brace_dx), FadeIn(dx_text), FadeIn(brace_dy), FadeIn(dy_text))
            self.wait(2)

        with self.voiceover(text="After applying Euclidean distance by taking the square root of the sum of squared differences, we accurately measure the physical, geometric distance between spatial embeddings, giving us a true measure of proximity that respects the continuous nature of Euclidean space.") as tracker:
            self.play(Indicate(line, color=RED, scale_factor=1.2))
            self.wait(2)

        with self.voiceover(text="Let us look at some highly specific real-world Use Cases. Uber uses Euclidean distance extensively in their dispatch systems to rapidly find the closest available driver to a rider on a continuous 2D map. Similarly, Zillow uses it to compute geographic proximity features when finding houses geographically near a specific landmark or point of interest.") as tracker:
            use_case_title = Text("Use Cases:", color=BLACK, font_size=32).shift(RIGHT * 3 + DOWN * 0.5)
            uc1 = Text("- Uber: Finding closest drivers geographically", color=BLACK, font_size=20).next_to(use_case_title, DOWN, aligned_edge=LEFT)
            uc2 = Text("- Zillow: Computing geographic housing proximity", color=BLACK, font_size=20).next_to(uc1, DOWN, aligned_edge=LEFT)

            self.play(Write(use_case_title))
            self.play(FadeIn(uc1))
            self.play(FadeIn(uc2))
            self.wait(2)

        with self.voiceover(text="And now for the Key Interview Insight, which is the most critical part to remember. Euclidean distance is highly sensitive to the magnitude, or length, of vectors. If one vector is simply a scaled-up version of another, the Euclidean distance between them will explode.") as tracker:
            insight_box = Rectangle(width=10, height=2, color=RED, fill_opacity=0.1).to_edge(DOWN)
            insight_text1 = Text("Key Interview Insight", color=RED, font_size=24, weight=BOLD).move_to(insight_box.get_top() + DOWN * 0.3)
            insight_text2 = Text("Highly sensitive to vector magnitude. Distance explodes if vectors are scaled.", color=BLACK, font_size=18).next_to(insight_text1, DOWN)
            insight_text3 = Text("Consequently, it is a poor choice for text embeddings where document length varies.", color=BLACK, font_size=18).next_to(insight_text2, DOWN)

            self.play(Create(insight_box))
            self.play(Write(insight_text1))
            self.play(Write(insight_text2))

        with self.voiceover(text="Because of this sensitivity, it is a poor choice for comparing text embeddings where the underlying document lengths vary significantly, as longer documents will be unfairly penalized. Remember this tradeoff.") as tracker:
            self.play(Write(insight_text3))
            self.wait(3)

        self.play(
            FadeOut(section_title), FadeOut(ax), FadeOut(dot1), FadeOut(dot2),
            FadeOut(label1), FadeOut(label2), FadeOut(line), FadeOut(dx_line),
            FadeOut(dy_line), FadeOut(brace_dx), FadeOut(brace_dy), FadeOut(dx_text),
            FadeOut(dy_text), FadeOut(eq2), FadeOut(use_case_title), FadeOut(uc1),
            FadeOut(uc2), FadeOut(insight_box), FadeOut(insight_text1),
            FadeOut(insight_text2), FadeOut(insight_text3)
        )

    def explain_cosine(self):
        section_title = Text("2. Cosine Similarity", color=GREEN, font_size=40).to_edge(UP)

        with self.voiceover(text="Moving on to our second metric, we have Cosine Similarity. What is it? Cosine similarity measures the cosine of the angle between two non-zero vectors. Crucially, it focuses entirely on their orientation and angle, rather than their absolute magnitude or length.") as tracker:
            self.play(Write(section_title))

            definition = Text("Measures the cosine of the angle between two vectors, focusing on orientation.", color=BLACK, font_size=20).next_to(section_title, DOWN)
            self.play(Write(definition))
            self.wait(1.5)
            self.play(FadeOut(definition))

            # Diagram & Math
            ax = Axes(x_range=[-1, 5], y_range=[-1, 5], x_length=4, y_length=4, axis_config={"color": BLACK}).shift(LEFT * 3 + DOWN * 0.5)
            origin = ax.coords_to_point(0, 0)
            p1 = ax.coords_to_point(4, 2)
            p2 = ax.coords_to_point(2, 4)
            p1_long = ax.coords_to_point(6, 3) # Scaled version of p1

            vec1 = Arrow(origin, p1, buff=0, color=BLUE)
            vec2 = Arrow(origin, p2, buff=0, color=RED)

            label_v1 = MathTex(r"\vec{A}", color=BLUE).next_to(p1, DOWN)
            label_v2 = MathTex(r"\vec{B}", color=RED).next_to(p2, UP)

            angle = Angle(vec1, vec2, radius=1.0, color=GREEN)
            theta = MathTex(r"\theta", color=GREEN).move_to(angle.point_from_proportion(0.5) + UP * 0.2 + RIGHT * 0.2)

            self.play(Create(ax))
            self.play(GrowArrow(vec1), FadeIn(label_v1))
            self.play(GrowArrow(vec2), FadeIn(label_v2))
            self.play(Create(angle), FadeIn(theta))
            self.wait(1)

        with self.voiceover(text="The mathematical derivation of Cosine Similarity stems from the geometric definition of the dot product. The cosine of the angle theta is equal to the dot product of vector A and vector B, divided by the product of their magnitudes.") as tracker:
            eq_cos = MathTex(r"\text{Cosine Similarity} = \cos(", r"\theta", r") =", r"\frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}", color=BLACK).shift(RIGHT * 3 + UP * 1.5)

            self.play(Write(eq_cos[0]), Write(eq_cos[1]), Write(eq_cos[2]))
            self.wait(1)
            self.play(Write(eq_cos[3]))
            self.wait(1)

        with self.voiceover(text="When we expand this into its component form, the numerator becomes the sum of the element-wise products of the two vectors. The denominator normalizes this value by dividing by the square roots of the sum of squared elements for each vector. This normalization ensures the resulting value is strictly bounded between negative one and positive one.") as tracker:
            eq_cos_parts = MathTex(r"=", r"\frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}", color=BLACK).next_to(eq_cos, DOWN)
            self.play(Write(eq_cos_parts))
            self.wait(2)

        with self.voiceover(text="Why do we need it? Consider a scenario where you are comparing a very short article and a massive textbook written about the exact same topic. Because the textbook has vastly higher term frequencies, a standard Euclidean distance calculation would see them as completely different points in space, drastically separated by magnitude.") as tracker:
            vec1_long = Arrow(origin, p1_long, buff=0, color=BLUE)
            self.wait(1)

        with self.voiceover(text="However, after applying Cosine similarity, we realize they contain the same proportional topics. Their vectors point in the exact same direction in feature space. By projecting them onto the unit sphere, Cosine similarity recognizes them as identical in semantic meaning, giving a similarity score of exactly one.") as tracker:
            self.play(Transform(vec1, vec1_long))
            self.wait(1)

            note = Text("Vectors pointing same direction\nhave similarity of exactly 1", color=BLACK, font_size=18).next_to(vec1_long, DOWN)
            self.play(FadeIn(note))
            self.wait(2)
            self.play(FadeOut(note))

        with self.voiceover(text="For our enterprise Use Cases, OpenAI uses Cosine similarity at massive scale to cluster dense text embeddings and retrieve relevant context in ChatGPT's Retrieval-Augmented Generation pipelines. Spotify also leverages it in collaborative filtering to recommend songs by comparing the angular orientation of multi-dimensional user preference vectors.") as tracker:
            use_case_title = Text("Use Cases:", color=BLACK, font_size=32).shift(RIGHT * 3 + DOWN * 0.5)
            uc1 = Text("- OpenAI: Text embedding retrieval in RAG", color=BLACK, font_size=20).next_to(use_case_title, DOWN, aligned_edge=LEFT)
            uc2 = Text("- Spotify: Comparing user preference vectors", color=BLACK, font_size=20).next_to(uc1, DOWN, aligned_edge=LEFT)

            self.play(Write(use_case_title))
            self.play(FadeIn(uc1))
            self.play(FadeIn(uc2))
            self.wait(2)

        with self.voiceover(text="Now for the Key Interview Insight. Cosine similarity ignores vector magnitude completely. It is blind to the total volume or length of the data.") as tracker:
            insight_box = Rectangle(width=10, height=2, color=GREEN, fill_opacity=0.1).to_edge(DOWN)
            insight_text1 = Text("Key Interview Insight", color=GREEN, font_size=24, weight=BOLD).move_to(insight_box.get_top() + DOWN * 0.3)
            insight_text2 = Text("Ignores vector magnitude entirely, operating only on angles.", color=BLACK, font_size=18).next_to(insight_text1, DOWN)
            insight_text3 = Text("Measures feature proportions, but loses total signal strength data.", color=BLACK, font_size=18).next_to(insight_text2, DOWN)

            self.play(Create(insight_box))
            self.play(Write(insight_text1))
            self.play(Write(insight_text2))

        with self.voiceover(text="It only tells you if two items have similar feature proportions or semantic directions, not if one has stronger overall signals. If overall magnitude is a crucial signal for your specific business problem, using cosine similarity will destructively hide that vital information.") as tracker:
            self.play(Write(insight_text3))
            self.wait(3)

        self.play(
            FadeOut(section_title), FadeOut(ax), FadeOut(vec1), FadeOut(vec2),
            FadeOut(label_v1), FadeOut(label_v2), FadeOut(angle), FadeOut(theta),
            FadeOut(eq_cos), FadeOut(eq_cos_parts), FadeOut(use_case_title),
            FadeOut(uc1), FadeOut(uc2), FadeOut(insight_box), FadeOut(insight_text1),
            FadeOut(insight_text2), FadeOut(insight_text3)
        )

    def explain_manhattan(self):
        section_title = Text("3. Manhattan Distance", color=PURPLE, font_size=40).to_edge(UP)

        with self.voiceover(text="Finally, let us comprehensively explore Manhattan Distance. What exactly is it? Manhattan distance, which is also formally known as the L1 norm, is the absolute distance between two points measured along axes at right angles, much like a taxi cab navigating city blocks on a strict grid.") as tracker:
            self.play(Write(section_title))

            definition = Text("Distance measured along right-angled axes, known as the L1 norm.", color=BLACK, font_size=20).next_to(section_title, DOWN)
            self.play(Write(definition))
            self.wait(1.5)
            self.play(FadeOut(definition))

            # Diagram & Math
            ax = Axes(x_range=[0, 6], y_range=[0, 6], x_length=4, y_length=4, axis_config={"color": BLACK}).shift(LEFT * 3 + DOWN * 0.5)
            p1 = ax.coords_to_point(1, 1)
            p2 = ax.coords_to_point(5, 4)

            # Grid lines
            grid = VGroup()
            for i in range(1, 6):
                grid.add(Line(ax.coords_to_point(i, 0), ax.coords_to_point(i, 6), color=GRAY, stroke_width=1, stroke_opacity=0.5))
                grid.add(Line(ax.coords_to_point(0, i), ax.coords_to_point(6, i), color=GRAY, stroke_width=1, stroke_opacity=0.5))

            dot1 = Dot(p1, color=PURPLE)
            dot2 = Dot(p2, color=PURPLE)
            label1 = MathTex("p", color=BLACK).next_to(dot1, DOWN)
            label2 = MathTex("q", color=BLACK).next_to(dot2, UP)

            self.play(Create(ax), Create(grid))
            self.play(FadeIn(dot1), FadeIn(label1), FadeIn(dot2), FadeIn(label2))

            # The manhattan path
            path1 = Line(p1, ax.coords_to_point(5, 1), color=PURPLE, stroke_width=5)
            path2 = Line(ax.coords_to_point(5, 1), p2, color=PURPLE, stroke_width=5)

        with self.voiceover(text="Mathematically, instead of squaring the differences like in Euclidean distance, we calculate the sum of the absolute differences of their coordinates. This prevents negative values while preserving linear scaling.") as tracker:
            eq1 = MathTex(r"d_1(p, q) =", r"|x_2 - x_1|", r"+", r"|y_2 - y_1|", color=BLACK).shift(RIGHT * 3 + UP * 1)

            self.play(Create(path1))
            self.play(Write(eq1[0]), Write(eq1[1]))
            self.play(Create(path2))
            self.play(Write(eq1[2]), Write(eq1[3]))
            self.wait(1)

        with self.voiceover(text="Once again, when extending this to high-dimensional spaces, we represent this using the summation of the absolute differences across all N dimensions, denoted as the L1 norm.") as tracker:
            eq2 = MathTex(r"d_1(p, q) =", r"\sum_{i=1}^{n}", r"|p_i - q_i|", color=BLACK).shift(RIGHT * 3 + UP * 1)
            self.play(TransformMatchingTex(eq1, eq2))
            self.wait(1.5)

        with self.voiceover(text="Why do we need it? Before using Manhattan distance, an algorithm relying on Euclidean distance might calculate a diagonal path that is literally impossible to traverse in a constrained grid-like system, drastically underestimating the true travel cost.") as tracker:
            euclid_line = DashedLine(p1, p2, color=RED)
            cross = Cross(euclid_line, color=RED)

            self.play(Create(euclid_line))
            self.play(Create(cross))
            self.wait(2)
            self.play(FadeOut(euclid_line), FadeOut(cross))

        with self.voiceover(text="After applying Manhattan distance, we correctly map the true cost of navigating grid-like structures. Notice that in this space, there are actually multiple optimal paths that yield the exact same distance calculation, highlighting how features are evaluated independently on their respective axes.") as tracker:
            alt_path1 = Line(p1, ax.coords_to_point(1, 4), color=BLUE, stroke_width=3)
            alt_path2 = Line(ax.coords_to_point(1, 4), p2, color=BLUE, stroke_width=3)
            self.play(Create(alt_path1), Create(alt_path2))

            note = Text("Multiple valid paths yield the same distance", color=BLACK, font_size=14).next_to(alt_path2, UP)
            self.play(FadeIn(note))
            self.wait(2)
            self.play(FadeOut(note), FadeOut(alt_path1), FadeOut(alt_path2))

        with self.voiceover(text="For our industrial Use Cases, Amazon heavily relies on Manhattan distance in their vast warehouse robotics systems to calculate optimal paths for robots moving exclusively along strict, grid-like storage aisles. In the financial sector, high-frequency trading algorithms often use it to measure absolute deviations in price movements without over-penalizing rare, extreme market shocks.") as tracker:
            use_case_title = Text("Use Cases:", color=BLACK, font_size=32).shift(RIGHT * 3 + DOWN * 0.5)
            uc1 = Text("- Amazon: Warehouse robotics grid pathing", color=BLACK, font_size=20).next_to(use_case_title, DOWN, aligned_edge=LEFT)
            uc2 = Text("- Finance: Absolute price deviations", color=BLACK, font_size=20).next_to(uc1, DOWN, aligned_edge=LEFT)

            self.play(Write(use_case_title))
            self.play(FadeIn(uc1))
            self.play(FadeIn(uc2))
            self.wait(2)

        with self.voiceover(text="And here is the final Key Interview Insight. Manhattan distance is fundamentally more robust to statistical outliers than Euclidean distance.") as tracker:
            insight_box = Rectangle(width=10, height=2, color=PURPLE, fill_opacity=0.1).to_edge(DOWN)
            insight_text1 = Text("Key Interview Insight", color=PURPLE, font_size=24, weight=BOLD).move_to(insight_box.get_top() + DOWN * 0.3)
            insight_text2 = Text("Significantly more robust to outliers than Euclidean distance.", color=BLACK, font_size=18).next_to(insight_text1, DOWN)
            insight_text3 = Text("Does not square differences, making it preferable for high-dimensional sparse data.", color=BLACK, font_size=18).next_to(insight_text2, DOWN)

            self.play(Create(insight_box))
            self.play(Write(insight_text1))
            self.play(Write(insight_text2))

        with self.voiceover(text="Because it absolutely does not square the differences, extreme, anomalous values do not heavily dominate the total distance calculation, making it highly preferable for evaluating high-dimensional, sparse datasets.") as tracker:
            self.play(Write(insight_text3))
            self.wait(3)

        self.play(
            FadeOut(section_title), FadeOut(ax), FadeOut(grid), FadeOut(dot1),
            FadeOut(dot2), FadeOut(label1), FadeOut(label2), FadeOut(path1),
            FadeOut(path2), FadeOut(eq2), FadeOut(use_case_title), FadeOut(uc1),
            FadeOut(uc2), FadeOut(insight_box), FadeOut(insight_text1),
            FadeOut(insight_text2), FadeOut(insight_text3)
        )
