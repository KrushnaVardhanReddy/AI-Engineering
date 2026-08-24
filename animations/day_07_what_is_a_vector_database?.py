from manim import *
import logging

class VectorDatabaseExplanation(Scene):
    def construct(self):
        try:
            self.setup_scene()
            self.animate_traditional_db()
            self.animate_vector_db()
            self.conclude()
        except Exception as e:
            # Fallback mechanism for unexpected errors during rendering
            logging.error(f"Animation failed: {e}")
            self.clear()
            error_text = Text("An error occurred during rendering.", color=RED)
            self.add(error_text)

    def setup_scene(self):
        # Whiteboard style background
        self.camera.background_color = WHITE

    def animate_traditional_db(self):
        # Title
        title = Text("Traditional Database", color=BLACK).to_edge(UP)
        self.play(FadeIn(title))

        # A simple table to represent structured data
        table_data = [["ID", "Name"], ["1", "Alice"], ["2", "Bob"]]
        table = Table(
            table_data,
            include_outer_lines=True,
        ).scale(0.6).set_color(BLACK)

        self.play(FadeIn(table))
        self.wait(1)

        # Explain keyword search
        search_text = Text("Keyword Search: 'Alice'", color=BLUE).next_to(table, DOWN)
        self.play(Write(search_text))

        # Highlight the result
        highlight = table.get_highlighted_cell((2, 2), color=BLUE)
        self.play(FadeIn(highlight))
        self.wait(2)

        # Clean up
        self.play(FadeOut(table), FadeOut(search_text), FadeOut(highlight), FadeOut(title))

    def animate_vector_db(self):
        # Title
        title = Text("Vector Database", color=BLACK).to_edge(UP)
        self.play(FadeIn(title))

        # Show unstructured data (e.g., text, image)
        data_text = Text("Unstructured Data\n(Text, Images, Audio)", color=BLACK, font_size=24).shift(LEFT * 4 + UP * 1)
        self.play(FadeIn(data_text))

        # Arrow pointing to embedding model
        arrow1 = Arrow(start=LEFT * 2.5 + UP * 1, end=LEFT * 1 + UP * 1, color=BLACK)
        self.play(GrowArrow(arrow1))

        # Embedding Model box
        model_box = Rectangle(width=2, height=1, color=RED, fill_opacity=0.2).move_to(RIGHT * 0.5 + UP * 1)
        model_text = Text("Embedding\nModel", color=BLACK, font_size=20).move_to(model_box.get_center())
        self.play(Create(model_box), Write(model_text))

        # Arrow pointing to Vector
        arrow2 = Arrow(start=RIGHT * 2 + UP * 1, end=RIGHT * 3.5 + UP * 1, color=BLACK)
        self.play(GrowArrow(arrow2))

        # Vector representation
        vector_text = MathTex(r"[0.1, -0.5, 0.8, \dots]", color=BLUE).next_to(arrow2, RIGHT)
        self.play(Write(vector_text))
        self.wait(2)

        # Transition to vector space
        self.play(
            FadeOut(data_text), FadeOut(arrow1), FadeOut(model_box),
            FadeOut(model_text), FadeOut(arrow2), FadeOut(vector_text)
        )

        # Show vector space
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"color": BLACK},
        ).scale(0.8).shift(DOWN * 0.5)
        self.play(Create(axes))

        # Add points (vectors)
        point1 = Dot(axes.c2p(2, 2), color=BLUE)
        label1 = Text("Apple", font_size=20, color=BLACK).next_to(point1, UP)

        point2 = Dot(axes.c2p(1.8, 1.5), color=BLUE)
        label2 = Text("Orange", font_size=20, color=BLACK).next_to(point2, DOWN)

        point3 = Dot(axes.c2p(-2, -1), color=RED)
        label3 = Text("Car", font_size=20, color=BLACK).next_to(point3, UP)

        self.play(FadeIn(point1), Write(label1), FadeIn(point2), Write(label2), FadeIn(point3), Write(label3))

        # Semantic Search explanation
        query_text = Text("Query: 'Fruit'", color=GREEN, font_size=24).to_edge(LEFT).shift(UP * 2)
        self.play(Write(query_text))

        # Query point
        query_point = Dot(axes.c2p(1.5, 2.5), color=GREEN)
        self.play(FadeIn(query_point))

        # Distance/Similarity
        line = DashedLine(query_point.get_center(), point1.get_center(), color=BLACK)
        self.play(Create(line))

        sim_text = Text("Semantic Similarity", color=BLACK, font_size=20).next_to(line, LEFT)
        self.play(Write(sim_text))

        self.wait(3)

        # Clean up
        self.play(
            FadeOut(axes), FadeOut(point1), FadeOut(label1), FadeOut(point2), FadeOut(label2),
            FadeOut(point3), FadeOut(label3), FadeOut(query_text), FadeOut(query_point),
            FadeOut(line), FadeOut(sim_text), FadeOut(title)
        )

    def conclude(self):
        # Summary
        summary = Text("Vector Databases power AI by\nfinding meaning, not just keywords.", color=BLACK).scale(0.8)
        self.play(Write(summary))
        self.wait(3)
        self.play(FadeOut(summary))
