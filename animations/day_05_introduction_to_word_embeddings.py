from manim import *

class WordEmbeddingsScene(Scene):
    def construct(self):
        # Set background to white
        self.camera.background_color = WHITE

        # Title
        title = Text("Introduction to Word Embeddings", color=BLACK, font_size=40)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # 1. Show the words
        words_group = VGroup(
            Text("King", color=BLACK, font_size=32),
            Text("Queen", color=BLACK, font_size=32),
            Text("Man", color=BLACK, font_size=32),
            Text("Woman", color=BLACK, font_size=32)
        ).arrange(DOWN, buff=1).shift(LEFT * 4)

        self.play(FadeIn(words_group, shift=RIGHT))
        self.wait(1)

        # 2. Show corresponding vectors (embeddings)
        # Using vibrant secondary colors for emphasis: Blue, Red, Green
        def create_embedding_array(values, color):
            # Create a simple array representation
            squares = VGroup(*[Square(side_length=0.6, fill_color=color, fill_opacity=0.3, color=color) for _ in values])
            squares.arrange(RIGHT, buff=0.1)

            texts = VGroup(*[Text(str(v), color=BLACK, font_size=20) for v in values])
            for i, sq in enumerate(squares):
                texts[i].move_to(sq.get_center())

            arr = VGroup(squares, texts)
            return arr

        # Dummy values to illustrate embedding dimensions (e.g., [Gender, Royalty])
        vec_king = create_embedding_array([0.9, 0.8], BLUE)
        vec_queen = create_embedding_array([-0.9, 0.8], RED)
        vec_man = create_embedding_array([0.9, 0.1], GREEN)
        vec_woman = create_embedding_array([-0.9, 0.1], ORANGE)

        vectors_group = VGroup(vec_king, vec_queen, vec_man, vec_woman)

        for i, (word, vec) in enumerate(zip(words_group, vectors_group)):
            vec.next_to(word, RIGHT, buff=1.5)
            # Add arrows
            arrow = Arrow(word.get_right(), vec.get_left(), buff=0.1, color=BLACK)
            self.play(GrowArrow(arrow), FadeIn(vec))

        self.wait(2)

        # 3. Show the semantic relationship: King - Man + Woman = Queen
        self.play(
            FadeOut(words_group),
            FadeOut(VGroup(*[mob for mob in self.mobjects if isinstance(mob, Arrow)]))
        )

        self.play(
            vec_king.animate.move_to(UP * 2 + LEFT * 3),
            vec_man.animate.move_to(UP * 2 + LEFT * 0.5),
            vec_woman.animate.move_to(UP * 2 + RIGHT * 2.5),
            vec_queen.animate.move_to(DOWN * 1)
        )

        minus = Text("-", color=BLACK, font_size=40).move_to(UP * 2 + LEFT * 1.75)
        plus = Text("+", color=BLACK, font_size=40).move_to(UP * 2 + RIGHT * 1)
        equals = Text("≈", color=BLACK, font_size=40).move_to(UP * 0.5)

        # Labels for the equation
        label_king = Text("King", color=BLACK, font_size=24).next_to(vec_king, UP)
        label_man = Text("Man", color=BLACK, font_size=24).next_to(vec_man, UP)
        label_woman = Text("Woman", color=BLACK, font_size=24).next_to(vec_woman, UP)
        label_queen = Text("Queen", color=BLACK, font_size=24).next_to(vec_queen, DOWN)

        self.play(
            Write(minus), Write(plus), Write(equals),
            FadeIn(label_king), FadeIn(label_man), FadeIn(label_woman), FadeIn(label_queen)
        )

        self.wait(3)

        # Fade out everything
        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(1)
