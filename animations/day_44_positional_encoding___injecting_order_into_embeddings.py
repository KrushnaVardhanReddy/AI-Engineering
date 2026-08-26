from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class PositionalEncodingScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # ---------------------------------------------------------
        # Step 1: Input Setup
        # ---------------------------------------------------------
        with self.voiceover(text="Welcome. Today we will explore Positional Encoding, and how we inject order into embeddings. We start with three tokens: The, cat, sat.") as tracker:
            title = Text("Positional Encoding", color=BLACK, font_size=48).to_edge(UP)
            self.play(Write(title))

            tokens = ["The", "cat", "sat"]
            token_boxes = VGroup()
            for i, token_text in enumerate(tokens):
                box = Rectangle(width=2, height=1, color=BLACK, fill_opacity=0)
                text = Text(token_text, color=BLACK, font_size=36)
                text.move_to(box.get_center())
                group = VGroup(box, text)
                token_boxes.add(group)

            token_boxes.arrange(RIGHT, buff=0.5).shift(UP*1.5)
            self.play(FadeIn(token_boxes))

            # Highlight "cat"
            cat_group = token_boxes[1]
            cat_box = cat_group[0]
            cat_text = cat_group[1]
            self.play(
                cat_box.animate.set_fill(BLUE, opacity=0.3),
                cat_box.animate.set_stroke(color=BLUE),
                cat_text.animate.set_color(BLUE)
            )

            self.wait(1)

        # ---------------------------------------------------------
        # Step 2: Data Flow - Initial Embedding
        # ---------------------------------------------------------
        with self.voiceover(text="First, the token 'cat' is mapped to its semantic word embedding. This vector contains its meaning, but not its position.") as tracker:
            emb_arrow = Arrow(start=cat_group.get_bottom(), end=cat_group.get_bottom() + DOWN*1.5, color=BLACK)
            self.play(GrowArrow(emb_arrow))

            emb_vector = MathTex(
                r"\begin{bmatrix} 0.5 \\ 0.2 \\ -0.1 \\ 0.8 \end{bmatrix}",
                color=BLACK
            ).next_to(emb_arrow, DOWN)

            emb_label = Text("Word Embedding", color=BLACK, font_size=24).next_to(emb_vector, RIGHT)

            self.play(Write(emb_vector), FadeIn(emb_label))
            self.wait(2)

        # ---------------------------------------------------------
        # Step 3: Data Flow - Positional Encoding
        # ---------------------------------------------------------
        with self.voiceover(text="Next, we calculate the positional encoding for position 1, using sines and cosines. We then add this positional vector to our word embedding.") as tracker:
            pos_vector = MathTex(
                r"+ \begin{bmatrix} 0.84 \\ 0.54 \\ 0.00 \\ 1.00 \end{bmatrix}",
                color=BLACK
            ).next_to(emb_vector, RIGHT, buff=2)

            pos_label = Text("Positional Encoding (pos=1)", color=BLACK, font_size=24).next_to(pos_vector, RIGHT)

            self.play(
                emb_label.animate.next_to(emb_vector, LEFT),
                Write(pos_vector),
                FadeIn(pos_label)
            )

            self.wait(2)

            result_vector = MathTex(
                r"= \begin{bmatrix} 1.34 \\ 0.74 \\ -0.10 \\ 1.80 \end{bmatrix}",
                color=GREEN
            ).next_to(pos_vector, DOWN, buff=1).shift(LEFT*1)

            result_label = Text("Context-Aware Embedding", color=GREEN, font_size=24).next_to(result_vector, RIGHT)

            self.play(Write(result_vector), FadeIn(result_label))
            self.wait(2)

        # ---------------------------------------------------------
        # Step 4: Data Flow - Query Generation (Q)
        # ---------------------------------------------------------
        with self.voiceover(text="Now let's trace this context-aware vector through the attention mechanism. Here, the token 'cat' is multiplied by the Query weight matrix W_Q to produce the query vector q.") as tracker:
            self.play(
                FadeOut(emb_arrow),
                FadeOut(emb_vector),
                FadeOut(pos_vector),
                FadeOut(emb_label),
                FadeOut(pos_label),
                result_vector.animate.shift(UP*3 + LEFT*2),
                result_label.animate.shift(UP*3 + LEFT*2)
            )

            q_arrow = Arrow(start=result_vector.get_bottom(), end=result_vector.get_bottom() + DOWN*1.5, color=BLACK)
            self.play(GrowArrow(q_arrow))

            # Context vector * W_Q = q
            # Using smaller 2D vectors for simplicity in multiplication display
            math_expr = MathTex(
                r"\begin{bmatrix} 1.34 & 0.74 \end{bmatrix}",
                r"\times",
                r"\begin{bmatrix} 0.1 & -0.2 \\ 0.3 & 0.4 \end{bmatrix}",
                r"=",
                r"\begin{bmatrix} 0.36 & 0.03 \end{bmatrix}",
                color=BLACK
            ).next_to(q_arrow, DOWN)

            # Color coding
            math_expr[0].set_color(GREEN) # Input
            math_expr[2].set_color(PURPLE) # W_Q
            math_expr[4].set_color(GREEN) # Output q

            labels = VGroup(
                Text("x", color=GREEN, font_size=20).next_to(math_expr[0], DOWN),
                Text("W_Q", color=PURPLE, font_size=20).next_to(math_expr[2], DOWN),
                Text("q", color=GREEN, font_size=20).next_to(math_expr[4], DOWN)
            )

            self.play(Write(math_expr), FadeIn(labels))
            self.wait(2)

        # ---------------------------------------------------------
        # Step 5: Backpropagation (Reverse Pass)
        # ---------------------------------------------------------
        with self.voiceover(text="In the forward pass, data flows forward. But during backprop, the gradient flows back to update the weights.") as tracker:
            # We fade out the forward math to show backprop
            self.play(
                FadeOut(math_expr),
                FadeOut(labels),
                FadeOut(q_arrow),
                FadeOut(result_vector),
                FadeOut(result_label),
                FadeOut(token_boxes),
                title.animate.shift(UP*0.5)
            )

            loss_text = Text("Loss (L)", color=RED, font_size=36).to_edge(RIGHT).shift(UP*1)
            q_text = Text("q", color=GREEN, font_size=36).shift(UP*1)
            wq_text = Text("W_Q", color=PURPLE, font_size=36).to_edge(LEFT).shift(UP*1)

            fwd_arrow1 = Arrow(start=wq_text.get_right(), end=q_text.get_left(), color=BLACK)
            fwd_arrow2 = Arrow(start=q_text.get_right(), end=loss_text.get_left(), color=BLACK)

            self.play(FadeIn(loss_text, q_text, wq_text))
            self.play(GrowArrow(fwd_arrow1), GrowArrow(fwd_arrow2))
            self.wait(1)

            # Backward pass
            bwd_arrow1 = DashedLine(
                start=loss_text.get_bottom() + LEFT*0.5,
                end=q_text.get_bottom() + RIGHT*0.5,
                color=RED
            ).add_tip()
            bwd_arrow2 = DashedLine(
                start=q_text.get_bottom() + LEFT*0.5,
                end=wq_text.get_bottom() + RIGHT*0.5,
                color=RED
            ).add_tip()

            self.play(Create(bwd_arrow1))
            self.wait(1)
            self.play(Create(bwd_arrow2))

            # Gradient Derivation
            grad_eq = MathTex(
                r"\frac{\partial L}{\partial W_Q} = \frac{\partial L}{\partial q} \times \frac{\partial q}{\partial W_Q}",
                color=RED
            ).shift(DOWN*1.5)

            self.play(Write(grad_eq))
            self.wait(2)

            grad_eq_step2 = MathTex(
                r"\frac{\partial L}{\partial W_Q} = \delta_q \times x^T",
                color=RED
            ).next_to(grad_eq, DOWN)

            self.play(Write(grad_eq_step2))
            self.wait(2)

            update_text = Text("Weights Updated!", color=GREEN, font_size=32).next_to(grad_eq_step2, DOWN)
            self.play(FadeIn(update_text), wq_text.animate.set_color(GREEN))
            self.wait(2)

        # ---------------------------------------------------------
        # Step 6: Interview Gotcha Callout
        # ---------------------------------------------------------
        with self.voiceover(text="A common interview gotcha: Why do we use sines and cosines instead of absolute positional indices? Because absolute indices would grow unbounded, causing model instability. Sines and cosines are bounded between -1 and 1, and allow the model to easily learn relative positions.") as tracker:
            self.play(
                FadeOut(loss_text, q_text, wq_text, fwd_arrow1, fwd_arrow2, bwd_arrow1, bwd_arrow2, grad_eq, grad_eq_step2, update_text)
            )

            callout_box = Rectangle(width=10, height=4, color=RED, fill_color=WHITE, fill_opacity=1)
            callout_title = Text("Interview Gotcha!", color=RED, font_size=36, weight=BOLD).next_to(callout_box.get_top(), DOWN, buff=0.2)

            gotcha_text = Text(
                "Why use Sines/Cosines instead of 1, 2, 3...?",
                color=BLACK, font_size=28
            ).next_to(callout_title, DOWN, buff=0.5)

            gotcha_bullet1 = Text(
                "1. Absolute indices grow unbounded, destabilizing training.",
                color=BLACK, font_size=24
            ).next_to(gotcha_text, DOWN, buff=0.3).align_to(gotcha_text, LEFT)

            gotcha_bullet2 = Text(
                "2. Sine/Cosine functions are bounded [-1, 1].",
                color=BLACK, font_size=24
            ).next_to(gotcha_bullet1, DOWN, buff=0.2).align_to(gotcha_text, LEFT)

            gotcha_bullet3 = Text(
                "3. They allow easy learning of relative positions.",
                color=BLACK, font_size=24
            ).next_to(gotcha_bullet2, DOWN, buff=0.2).align_to(gotcha_text, LEFT)

            callout_group = VGroup(callout_box, callout_title, gotcha_text, gotcha_bullet1, gotcha_bullet2, gotcha_bullet3)
            callout_group.move_to(ORIGIN)

            self.play(FadeIn(callout_group))
            self.wait(3)

        with self.voiceover(text="That concludes our deep dive into positional encoding.") as tracker:
            self.play(FadeOut(callout_group, title))
            self.wait(1)
