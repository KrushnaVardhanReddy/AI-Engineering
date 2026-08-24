from manim import *

class CosineSimilarity3D(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        text = Text("Hello World", color=BLACK)
        self.play(Write(text))
        self.wait(1)
