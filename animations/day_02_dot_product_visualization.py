from manim import *

class DotProductVisualization(Scene):
    def construct(self):
        # Set background to pure white as per whiteboard aesthetic requirements
        self.camera.background_color = WHITE

        # Title
        title = Text("Dot Product Visualization", color=BLACK).to_edge(UP)
        self.play(FadeIn(title))

        # Create vectors
        # Using vibrant secondary colors (BLUE and RED) against WHITE background
        vec_a = Vector([3, 1], color=BLUE)
        vec_b = Vector([2, 3], color=RED)

        # Labels for vectors
        label_a = MathTex("\\vec{a}", color=BLUE).next_to(vec_a.get_end(), RIGHT)
        label_b = MathTex("\\vec{b}", color=RED).next_to(vec_b.get_end(), UP)

        # Coordinate system / Origin dot
        origin = Dot(ORIGIN, color=BLACK)

        # Group vectors and labels
        vectors = VGroup(vec_a, vec_b, label_a, label_b, origin)

        # Shift everything slightly down and left to make room
        vectors.shift(DOWN * 1 + LEFT * 2)

        # Draw vectors
        self.play(FadeIn(origin))
        self.play(GrowArrow(vec_a), Write(label_a))
        self.play(GrowArrow(vec_b), Write(label_b))
        self.wait(1)

        # Show projection line from b to a
        # Calculate projection of b onto a
        # a . b = 3*2 + 1*3 = 9
        # |a|^2 = 3^2 + 1^2 = 10
        # proj_a(b) = (a.b / |a|^2) * a = 0.9 * [3, 1] = [2.7, 0.9]
        proj_end = vec_a.get_start() + 0.9 * (vec_a.get_end() - vec_a.get_start())

        # Draw dashed line for projection
        proj_line = DashedLine(vec_b.get_end(), proj_end, color=BLACK)
        self.play(Create(proj_line))

        # Draw projection vector
        vec_proj = Vector(0.9 * np.array([3, 1, 0]), color=GREEN).shift(vec_a.get_start())
        label_proj = MathTex("proj_{\\vec{a}}(\\vec{b})", color=GREEN).next_to(vec_proj, DOWN, buff=0.1)

        self.play(GrowArrow(vec_proj), Write(label_proj))
        self.wait(1)

        # Equation showing dot product algebraically
        eq_text = MathTex(
            "\\vec{a} \\cdot \\vec{b}", "=", "(3)(2) + (1)(3)", "=", "9",
            color=BLACK
        ).to_edge(DOWN)

        # Color specific parts of the equation
        eq_text[0][0:2].set_color(BLUE)  # vec a
        eq_text[0][3:5].set_color(RED)   # vec b
        eq_text[2][1].set_color(BLUE)    # 3
        eq_text[2][4].set_color(RED)     # 2
        eq_text[2][8].set_color(BLUE)    # 1
        eq_text[2][11].set_color(RED)    # 3

        self.play(Write(eq_text))
        self.wait(2)

        # Geometric equation
        geom_eq = MathTex(
            "\\vec{a} \\cdot \\vec{b} = |\\vec{a}| |\\vec{b}| \\cos(\\theta)",
            color=BLACK
        ).next_to(eq_text, UP, buff=0.5)

        # Highlight theta
        angle = Angle(vec_a, vec_b, radius=0.8, color=BLACK)
        angle_label = MathTex("\\theta", color=BLACK).next_to(angle, RIGHT, buff=0.1).shift(UP*0.2)

        self.play(Create(angle), Write(angle_label))
        self.play(Write(geom_eq))

        self.wait(3)

        # Fade out everything
        self.play(
            FadeOut(VGroup(title, vectors, proj_line, vec_proj, label_proj, eq_text, geom_eq, angle, angle_label))
        )
