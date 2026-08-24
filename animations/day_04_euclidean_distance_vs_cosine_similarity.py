from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class DistanceVsSimilarity(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # Color palette
        TEXT_COLOR = BLACK
        H1_COLOR = BLUE
        H2_COLOR = PURPLE
        EUCLIDEAN_COLOR = RED
        COSINE_COLOR = GREEN

        # --- TITLE ---
        title = Tex("Euclidean Distance vs Cosine Similarity", color=TEXT_COLOR, font_size=48)
        subtitle = Tex("AI Engineering Interview Prep", color=H1_COLOR, font_size=36)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.5)

        with self.voiceover(text="Welcome back to our AI Engineering Interview Prep series. Today, we're diving into Euclidean Distance versus Cosine Similarity.") as tracker:
            self.play(Write(title_group))

        self.wait(1.5)
        self.play(FadeOut(title_group))

        # --- SECTION 1: WHAT IS IT? ---
        section1_title = Tex("1. What is it?", color=H1_COLOR, font_size=42).to_corner(UL)

        with self.voiceover(text="First, what are they? Let's start with definitions.") as tracker:
            self.play(FadeIn(section1_title))

        euclidean_def = Tex("Euclidean Distance: The straight-line distance between two points.", color=TEXT_COLOR, font_size=32).next_to(section1_title, DOWN, aligned_edge=LEFT, buff=0.5)

        with self.voiceover(text="Euclidean Distance is the straight-line distance between two points in space.") as tracker:
            self.play(Write(euclidean_def))

        # Draw axes
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            axis_config={"color": BLACK, "include_tip": True}
        ).scale(0.6).to_edge(RIGHT, buff=1)

        p1_coords = axes.c2p(1, 1)
        p2_coords = axes.c2p(4, 4)

        p1 = Dot(p1_coords, color=EUCLIDEAN_COLOR)
        p2 = Dot(p2_coords, color=EUCLIDEAN_COLOR)

        p1_label = MathTex("(x_1, y_1)", color=TEXT_COLOR, font_size=24).next_to(p1, LEFT)
        p2_label = MathTex("(x_2, y_2)", color=TEXT_COLOR, font_size=24).next_to(p2, RIGHT)

        line = Line(p1_coords, p2_coords, color=EUCLIDEAN_COLOR)

        with self.voiceover(text="If you have two points, you plot them on your coordinate system and connect them with a straight line.") as tracker:
            self.play(Create(axes), FadeIn(p1, p1_label), FadeIn(p2, p2_label))
            self.play(Create(line))

        euclid_formula_1 = MathTex("d^2", "=", "(x_2 - x_1)^2", "+", "(y_2 - y_1)^2", color=EUCLIDEAN_COLOR, font_size=32).next_to(euclidean_def, DOWN, aligned_edge=LEFT, buff=0.5)
        euclid_formula_2 = MathTex("d", "=", "\\sqrt{", "(x_2 - x_1)^2", "+", "(y_2 - y_1)^2", "}", color=EUCLIDEAN_COLOR, font_size=32).next_to(euclidean_def, DOWN, aligned_edge=LEFT, buff=0.5)

        with self.voiceover(text="Using the Pythagorean theorem, you first find the squared differences of their coordinates.") as tracker:
            self.play(Write(euclid_formula_1))

        self.wait(1.5)

        with self.voiceover(text="Then, you take the square root to find the actual distance.") as tracker:
            self.play(TransformMatchingTex(euclid_formula_1, euclid_formula_2))

        self.wait(1.5)

        cosine_def = Tex("Cosine Similarity: The cosine of the angle between two vectors.", color=TEXT_COLOR, font_size=32).next_to(euclid_formula_2, DOWN, aligned_edge=LEFT, buff=0.5)

        with self.voiceover(text="On the other hand, Cosine Similarity measures the cosine of the angle between two vectors.") as tracker:
            self.play(Write(cosine_def))

        origin = axes.c2p(0, 0)
        vec1 = Arrow(origin, p1_coords, buff=0, color=COSINE_COLOR)
        vec2 = Arrow(origin, axes.c2p(4, 2), buff=0, color=COSINE_COLOR) # Different vector for angle

        angle = Angle(vec1, vec2, radius=1.0, color=PURPLE)
        theta_label = MathTex("\\theta", color=PURPLE, font_size=24).next_to(angle, RIGHT, buff=0.1)

        with self.voiceover(text="Instead of looking at the points themselves, we treat them as arrows originating from the origin, and we measure the angle theta between them.") as tracker:
            self.play(FadeOut(line), FadeOut(p1_label), FadeOut(p2_label), FadeOut(p2))
            self.play(GrowArrow(vec1), GrowArrow(vec2))
            self.play(Create(angle), Write(theta_label))

        cosine_formula_1 = MathTex("A", "\\cdot", "B", "=", "\\|A\\|", "\\|B\\|", "\\cos(\\theta)", color=COSINE_COLOR, font_size=32).next_to(cosine_def, DOWN, aligned_edge=LEFT, buff=0.5)
        cosine_formula_2 = MathTex("\\cos(\\theta)", "=", "{", "A", "\\cdot", "B", "\\over", "\\|A\\|", "\\|B\\|", "}", color=COSINE_COLOR, font_size=32).next_to(cosine_def, DOWN, aligned_edge=LEFT, buff=0.5)

        with self.voiceover(text="The dot product of two vectors is equal to the product of their magnitudes times the cosine of the angle.") as tracker:
            self.play(Write(cosine_formula_1))

        self.wait(1.5)

        with self.voiceover(text="By rearranging this equation, we isolate the cosine of the angle, giving us our similarity metric.") as tracker:
            self.play(TransformMatchingTex(cosine_formula_1, cosine_formula_2))

        self.wait(1.5)
        self.play(
            FadeOut(VGroup(section1_title, euclidean_def, euclid_formula_2, cosine_def, cosine_formula_2, axes, p1, vec1, vec2, angle, theta_label))
        )

        # --- SECTION 2: WHY DO WE NEED IT? ---
        section2_title = Tex("2. Why do we need it?", color=H1_COLOR, font_size=42).to_corner(UL)

        with self.voiceover(text="Why do we need both metrics? Let's look at a specific problem: comparing documents of different lengths.") as tracker:
            self.play(FadeIn(section2_title))

        doc1 = Rectangle(height=1, width=1.5, color=BLACK).set_fill(BLUE, opacity=0.2)
        doc1_label = Tex("Short Artcl", color=BLACK, font_size=24).move_to(doc1)
        doc1_group = VGroup(doc1, doc1_label).shift(LEFT*4 + UP*1)

        doc2 = Rectangle(height=2.5, width=1.5, color=BLACK).set_fill(BLUE, opacity=0.2)
        doc2_label = Tex("Long Artcl", color=BLACK, font_size=24).move_to(doc2)
        doc2_group = VGroup(doc2, doc2_label).shift(LEFT*1 + UP*1)

        doc3 = Rectangle(height=1, width=1.5, color=BLACK).set_fill(RED, opacity=0.2)
        doc3_label = Tex("Other Topic", color=BLACK, font_size=24).move_to(doc3)
        doc3_group = VGroup(doc3, doc3_label).shift(RIGHT*3 + UP*1)

        with self.voiceover(text="Imagine we have a short article and a long article on the exact same topic, and a third short article on a completely different topic.") as tracker:
            self.play(FadeIn(doc1_group), FadeIn(doc2_group), FadeIn(doc3_group))

        axes2 = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            axis_config={"color": BLACK, "include_tip": True}
        ).scale(0.6).shift(DOWN*1.5)

        x_label = Tex("Word 'AI'", color=BLACK, font_size=24).next_to(axes2.x_axis, DOWN)
        y_label = Tex("Word 'Data'", color=BLACK, font_size=24).next_to(axes2.y_axis, LEFT)
        axes2_group = VGroup(axes2, x_label, y_label)

        d1_p = axes2.c2p(2, 2)
        d2_p = axes2.c2p(8, 8)
        d3_p = axes2.c2p(1, 8)

        d1_vec = Arrow(axes2.c2p(0,0), d1_p, buff=0, color=BLUE)
        d2_vec = Arrow(axes2.c2p(0,0), d2_p, buff=0, color=BLUE)
        d3_vec = Arrow(axes2.c2p(0,0), d3_p, buff=0, color=RED)

        with self.voiceover(text="If we count word frequencies and plot them as vectors, the short article and the long article point in the same direction, but have vastly different lengths.") as tracker:
            self.play(Create(axes2_group))
            self.play(GrowArrow(d1_vec), GrowArrow(d2_vec), GrowArrow(d3_vec))

        euclid_line = Line(d1_p, d2_p, color=RED, stroke_width=4)
        euclid_text = Tex("Large Euclidean Dist", color=RED, font_size=24).next_to(euclid_line, RIGHT, buff=0.1)

        with self.voiceover(text="Without Cosine Similarity, if we just use Euclidean Distance, the short and long articles seem very different because the physical distance between their endpoints is huge.") as tracker:
            self.play(Create(euclid_line), Write(euclid_text))

        self.wait(1.5)

        cosine_text = Tex("Cosine Sim = 1.0 (Angle = 0)", color=GREEN, font_size=28).move_to(euclid_text.get_center())

        with self.voiceover(text="With Cosine Similarity, we only look at the angle. Because they point in the exact same direction, the angle is zero, and their cosine similarity is one.") as tracker:
            self.play(FadeOut(euclid_line), FadeOut(euclid_text))
            self.play(Indicate(d1_vec, color=GREEN, scale_factor=1.2), Indicate(d2_vec, color=GREEN, scale_factor=1.2))
            self.play(Write(cosine_text))

        with self.voiceover(text="It perfectly captures that they are about the same topic, regardless of their word count magnitudes.") as tracker:
            self.wait(1.5)

        self.play(
            FadeOut(VGroup(section2_title, doc1_group, doc2_group, doc3_group, axes2_group, d1_vec, d2_vec, d3_vec, cosine_text))
        )

        # --- SECTION 3: USE CASES ---
        section3_title = Tex("3. Use Cases", color=H1_COLOR, font_size=42).to_corner(UL)

        with self.voiceover(text="Let's look at real-world use cases where you would choose one over the other.") as tracker:
            self.play(FadeIn(section3_title))

        uc_euclid = VGroup(
            Tex("Euclidean Distance", color=EUCLIDEAN_COLOR, font_size=36),
            Tex("• Google Maps: Routing and physical distance.", color=TEXT_COLOR, font_size=28),
            Tex("• Uber: K-Means Clustering on rider pickup locations.", color=TEXT_COLOR, font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT).shift(UP*1 + LEFT*3)

        with self.voiceover(text="You use Euclidean Distance when absolute magnitude or physical space matters. For example, Google Maps routing, or Uber clustering rider pickup locations using K-Means.") as tracker:
            for item in uc_euclid:
                self.play(FadeIn(item, shift=UP*0.5))
                self.wait(0.5)

        self.wait(1.5)

        uc_cosine = VGroup(
            Tex("Cosine Similarity", color=COSINE_COLOR, font_size=36),
            Tex("• ChatGPT: Semantic search and document retrieval.", color=TEXT_COLOR, font_size=28),
            Tex("• Spotify: Song recommendations based on user preference vectors.", color=TEXT_COLOR, font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT).shift(DOWN*1.5 + LEFT*3)

        with self.voiceover(text="You use Cosine Similarity when relative distribution or orientation matters, but magnitude doesn't. For example, ChatGPT using semantic search to find relevant documents, or Spotify comparing user preference vectors for song recommendations.") as tracker:
            for item in uc_cosine:
                self.play(FadeIn(item, shift=UP*0.5))
                self.wait(0.5)

        self.wait(1.5)
        self.play(FadeOut(VGroup(section3_title, uc_euclid, uc_cosine)))

        # --- SECTION 4: KEY INTERVIEW INSIGHT ---
        section4_title = Tex("4. Key Interview Insight", color=H1_COLOR, font_size=42).to_corner(UL)

        with self.voiceover(text="Finally, here is the key insight interviewers want you to know.") as tracker:
            self.play(FadeIn(section4_title))

        box = Rectangle(width=10, height=5, color=PURPLE, stroke_width=4).set_fill(PURPLE, opacity=0.1)
        box_title = Tex("The Magnitude Gotcha", color=PURPLE, font_size=36).next_to(box.get_top(), DOWN, buff=0.2)

        insight_text = VGroup(
            Tex("Cosine similarity entirely ignores vector magnitude.", color=BLACK, font_size=32),
            Tex("If user A buys 1 apple and 1 banana, and user B buys 100 apples and 100 bananas,", color=BLACK, font_size=28),
            Tex("Cosine Sim = 1.0 (Identical preferences).", color=GREEN, font_size=28),
            Tex("If the total volume purchased matters for your model, Cosine Similarity will fail you.", color=RED, font_size=28)
        ).arrange(DOWN, buff=0.4).next_to(box_title, DOWN, buff=0.4)

        insight_group = VGroup(box, box_title, insight_text).move_to(ORIGIN)

        with self.voiceover(text="This is the Magnitude Gotcha. Cosine similarity entirely ignores vector magnitude.") as tracker:
            self.play(Create(box), Write(box_title))
            self.play(Write(insight_text[0]))

        self.wait(1.5)

        with self.voiceover(text="If user A buys one apple and one banana, and user B buys 100 apples and 100 bananas, their cosine similarity is 1.0. The algorithm thinks they are perfectly identical.") as tracker:
            self.play(Write(insight_text[1]))
            self.play(Write(insight_text[2]))

        self.wait(1.5)

        with self.voiceover(text="If the total volume purchased matters for your business model—like distinguishing a casual shopper from a wholesaler—Cosine Similarity will fail you, and you must use Euclidean distance or a combined approach.") as tracker:
            self.play(Write(insight_text[3]))

        self.wait(2)

        with self.voiceover(text="That concludes our deep dive into Euclidean Distance versus Cosine Similarity. Good luck with your interviews!") as tracker:
            self.play(FadeOut(VGroup(section4_title, insight_group)))

        final_text = Tex("Keep building.", color=BLACK, font_size=48)
        self.play(FadeIn(final_text))
        self.wait(2)
        self.play(FadeOut(final_text))
