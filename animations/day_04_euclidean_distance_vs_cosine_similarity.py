from manim import *

class EuclideanVsCosine(Scene):
    def construct(self):
        # Aesthetic Requirements: Whiteboard Style
        self.camera.background_color = WHITE

        # Title
        title = Text(
            "Euclidean Distance vs Cosine Similarity",
            color=BLACK,
            font_size=40
        ).to_edge(UP)
        self.play(FadeIn(title))
        self.wait(1)

        # Draw Coordinate System
        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            axis_config={"color": BLACK, "include_numbers": False, "tip_shape": StealthTip},
            x_length=6,
            y_length=6
        ).shift(DOWN * 0.5)
        self.play(Create(axes), run_time=1.5)

        # Define points
        point_A = [2, 4, 0]
        point_B = [4, 3, 0]

        coord_A = axes.c2p(*point_A[:2])
        coord_B = axes.c2p(*point_B[:2])
        origin = axes.c2p(0, 0)

        # Plot vectors
        vec_A = Arrow(origin, coord_A, buff=0, color=BLUE, stroke_width=4, max_tip_length_to_length_ratio=0.08)
        vec_B = Arrow(origin, coord_B, buff=0, color=RED, stroke_width=4, max_tip_length_to_length_ratio=0.08)

        label_A = MathTex("A(2, 4)", color=BLUE, font_size=32).next_to(coord_A, UP + LEFT, buff=0.1)
        label_B = MathTex("B(4, 3)", color=RED, font_size=32).next_to(coord_B, UP + RIGHT, buff=0.1)

        self.play(
            GrowArrow(vec_A), Write(label_A),
            GrowArrow(vec_B), Write(label_B),
            run_time=2
        )
        self.wait(1)

        # --- Euclidean Distance Section ---
        euclidean_line = DashedLine(coord_A, coord_B, color=GREEN, stroke_width=4)
        euclidean_label = Text("Euclidean Distance", color=GREEN, font_size=24).next_to(euclidean_line, UP + RIGHT, buff=0.1)

        self.play(Create(euclidean_line), Write(euclidean_label))
        self.wait(2)

        self.play(FadeOut(euclidean_line), FadeOut(euclidean_label))
        self.wait(0.5)

        # --- Cosine Similarity Section ---
        # Draw arc between vectors
        angle_arc = Angle(
            vec_B, vec_A, radius=1.0, color=PURPLE, stroke_width=4
        )
        angle_label = MathTex(r"\theta", color=PURPLE, font_size=36).move_to(
            Angle(vec_B, vec_A, radius=1.4).point_from_proportion(0.5)
        )

        cosine_text = Text("Cosine Similarity (angle)", color=PURPLE, font_size=24).next_to(angle_arc, RIGHT, buff=1.0)

        self.play(Create(angle_arc), Write(angle_label))
        self.play(Write(cosine_text))
        self.wait(2)

        # --- Explanation Section ---
        explanation = VGroup(
            Text("Euclidean measures magnitude difference.", color=GREEN, font_size=24),
            Text("Cosine measures directional alignment.", color=PURPLE, font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_corner(DL)

        self.play(FadeIn(explanation))
        self.wait(3)

        # Clean up
        self.play(
            FadeOut(title), FadeOut(axes), FadeOut(vec_A), FadeOut(vec_B),
            FadeOut(label_A), FadeOut(label_B), FadeOut(angle_arc),
            FadeOut(angle_label), FadeOut(cosine_text), FadeOut(explanation)
        )
        self.wait(1)
