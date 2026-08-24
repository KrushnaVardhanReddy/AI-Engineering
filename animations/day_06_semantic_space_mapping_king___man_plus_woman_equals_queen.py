from manim import *

class SemanticSpaceMapping(Scene):
    def construct(self):
        # 1. Aesthetic Requirements: Whiteboard Style
        self.camera.background_color = WHITE

        # Title
        title = Text("Semantic Space Mapping", color=BLACK, font_size=40)
        subtitle = Text("King - Man + Woman = Queen", color=BLACK, font_size=32)
        subtitle.next_to(title, DOWN)

        title_group = VGroup(title, subtitle).to_edge(UP)

        self.play(Write(title_group))
        self.wait(1)

        # 2. Axes for the semantic space
        axes = Axes(
            x_range=[-1, 7, 1],
            y_range=[-1, 6, 1],
            x_length=8,
            y_length=6,
            axis_config={
                "color": BLACK,
                "include_numbers": False,
                "tip_shape": StealthTip
            }
        ).shift(DOWN*0.5)

        # Using Text instead of MathTex for labels to ensure it compiles without LaTeX dependencies
        x_label = Text("Gender", color=BLACK, font_size=24).next_to(axes.x_axis, RIGHT)
        y_label = Text("Royalty", color=BLACK, font_size=24).next_to(axes.y_axis, UP)
        axes_labels = VGroup(x_label, y_label)

        self.play(Create(axes), FadeIn(axes_labels))

        # 3. Define vector positions in our abstract 2D semantic space
        # Man: left side (male), lower side (commoner)
        man_coords = [1, 1]
        # King: left side (male), upper side (royal)
        king_coords = [1, 4]
        # Woman: right side (female), lower side (commoner)
        woman_coords = [5, 1]
        # Queen: right side (female), upper side (royal)
        queen_coords = [5, 4]

        # 4. Helper function to create point and label
        def create_point_and_label(coords, label_text, color):
            dot = Dot(axes.c2p(*coords), color=color, radius=0.1)
            label = Text(label_text, color=BLACK, font_size=24).next_to(dot, RIGHT)
            return dot, label

        man_dot, man_label = create_point_and_label(man_coords, "Man", BLUE)
        king_dot, king_label = create_point_and_label(king_coords, "King", BLUE)
        woman_dot, woman_label = create_point_and_label(woman_coords, "Woman", RED)
        queen_dot, queen_label = create_point_and_label(queen_coords, "Queen", PURPLE)

        # 5. Vectors from origin
        man_vec = Arrow(axes.c2p(0, 0), axes.c2p(*man_coords), buff=0, color=BLUE)
        king_vec = Arrow(axes.c2p(0, 0), axes.c2p(*king_coords), buff=0, color=BLUE)
        woman_vec = Arrow(axes.c2p(0, 0), axes.c2p(*woman_coords), buff=0, color=RED)
        queen_vec = Arrow(axes.c2p(0, 0), axes.c2p(*queen_coords), buff=0, color=PURPLE)

        # Animate base vectors for male concepts
        self.play(
            GrowArrow(man_vec), FadeIn(man_dot), FadeIn(man_label),
            GrowArrow(king_vec), FadeIn(king_dot), FadeIn(king_label)
        )
        self.wait(1)

        # 6. The relationship "Royalty" vector = King - Man
        royalty_vec = Arrow(axes.c2p(*man_coords), axes.c2p(*king_coords), buff=0, color=GREEN)
        royalty_label = Text("Royalty Vector (King - Man)", color=GREEN, font_size=20).next_to(royalty_vec, LEFT).shift(RIGHT*0.2)

        self.play(GrowArrow(royalty_vec), FadeIn(royalty_label))
        self.wait(1)

        # 7. Introduce Woman
        self.play(GrowArrow(woman_vec), FadeIn(woman_dot), FadeIn(woman_label))
        self.wait(1)

        # 8. Apply Royalty vector to Woman (shift copy of vector)
        royalty_vec_copy = royalty_vec.copy()

        self.play(
            royalty_vec_copy.animate.shift(axes.c2p(*woman_coords) - axes.c2p(*man_coords))
        )
        self.wait(1)

        # 9. Show Queen vector resulting from Woman + Royalty
        self.play(
            GrowArrow(queen_vec), FadeIn(queen_dot), FadeIn(queen_label)
        )
        self.wait(1)

        # 10. Final Equation Breakdown
        equation = VGroup(
            Text("King", color=BLUE, font_size=32),
            Text(" - ", color=BLACK, font_size=32),
            Text("Man", color=BLUE, font_size=32),
            Text(" + ", color=BLACK, font_size=32),
            Text("Woman", color=RED, font_size=32),
            Text(" = ", color=BLACK, font_size=32),
            Text("Queen", color=PURPLE, font_size=32)
        ).arrange(RIGHT)

        # Add a background rectangle to make it clearly visible over the axes
        equation_bg = BackgroundRectangle(equation, color=WHITE, fill_opacity=0.9, buff=0.2)
        eq_group = VGroup(equation_bg, equation).to_edge(DOWN)

        self.play(FadeIn(eq_group))
        self.wait(3)
