"""
Day 1: Vector Basics and Coordinates
Whiteboard Style Manim Animation
"""

from manim import *

class VectorBasicsAndCoordinates(Scene):
    """
    A Manim Scene explaining the basics of vectors and their coordinates
    in a 2D Cartesian plane using a whiteboard style.
    """
    def construct(self):
        # Aesthetic Requirements: Whiteboard Style
        self.camera.background_color = WHITE

        # Introduction Title
        title = Text("Vector Basics and Coordinates", color=BLACK, font_size=48)
        self.play(FadeIn(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # Setup Axes
        # Use black for primary text, equations, and shapes (axes)
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-4, 4, 1],
            x_length=10,
            y_length=8,
            axis_config={"color": BLACK},
        ).add_coordinates()

        axes_labels = axes.get_axis_labels(
            x_label=Text("x", color=BLACK), y_label=Text("y", color=BLACK)
        )

        # The add_coordinates labels might not be black by default, let's explicitly color them
        for number in axes.x_axis.numbers:
            number.set_color(BLACK)
        for number in axes.y_axis.numbers:
            number.set_color(BLACK)

        self.play(Create(axes), FadeIn(axes_labels))
        self.wait(1)

        # Define a vector at coordinates (3, 2)
        # Use vibrant secondary colors (blue) for emphasis
        vector = Arrow(axes.c2p(0, 0), axes.c2p(3, 2), buff=0, color=BLUE)

        # Label the vector
        vector_label = MathTex("\\vec{v} = \\begin{bmatrix} 3 \\\\ 2 \\end{bmatrix}", color=BLUE)
        vector_label.next_to(vector.get_end(), UP + RIGHT, buff=0.1)

        # Show vector creation
        self.play(GrowArrow(vector))
        self.play(Write(vector_label))
        self.wait(2)

        # Explain coordinates visually using dashed lines
        # X component
        dashed_x = DashedLine(
            start=axes.c2p(3, 0),
            end=axes.c2p(3, 2),
            color=RED
        )
        x_label = MathTex("x = 3", color=RED).next_to(dashed_x, RIGHT)

        # Y component
        dashed_y = DashedLine(
            start=axes.c2p(0, 2),
            end=axes.c2p(3, 2),
            color=GREEN
        )
        y_label = MathTex("y = 2", color=GREEN).next_to(dashed_y, UP)

        self.play(Create(dashed_x), Write(x_label))
        self.wait(1)
        self.play(Create(dashed_y), Write(y_label))
        self.wait(2)

        # Vector Addition Concept (Briefly)
        # Introduce a second vector
        vector_2 = Arrow(axes.c2p(0, 0), axes.c2p(-2, 1), buff=0, color=RED)

        vector_2_label = MathTex("\\vec{u} = \\begin{bmatrix} -2 \\\\ 1 \\end{bmatrix}", color=RED)
        vector_2_label.next_to(vector_2.get_end(), UP + LEFT, buff=0.1)

        self.play(GrowArrow(vector_2), Write(vector_2_label))
        self.wait(2)

        # Cleanup and Conclusion
        self.play(
            FadeOut(vector),
            FadeOut(vector_label),
            FadeOut(dashed_x),
            FadeOut(x_label),
            FadeOut(dashed_y),
            FadeOut(y_label),
            FadeOut(vector_2),
            FadeOut(vector_2_label),
        )

        conclusion_text = Text("Vectors are directed line segments\ndefined by components.", color=BLACK)
        conclusion_text.move_to(ORIGIN)

        # Fade out axes and fade in conclusion
        self.play(FadeOut(axes), FadeOut(axes_labels), FadeIn(conclusion_text))
        self.wait(3)
        self.play(FadeOut(conclusion_text), FadeOut(title))
        self.wait(1)
