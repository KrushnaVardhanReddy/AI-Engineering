from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np

class TokenEmbeddingScene(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # =========================================================================
        # 1. Input Setup
        # =========================================================================
        tokens = ["The", "cat", "sat"]

        token_boxes = VGroup()
        for token in tokens:
            box = Rectangle(width=2, height=1, color=BLACK).set_fill(WHITE, opacity=1)
            text = Text(token, color=BLACK)
            box_text = VGroup(box, text)
            token_boxes.add(box_text)

        token_boxes.arrange(RIGHT, buff=0.5).move_to(UP*2)

        with self.voiceover(text="We start with three tokens: 'The', 'cat', and 'sat'.") as tracker:
            self.play(FadeIn(token_boxes), run_time=tracker.duration)

        with self.voiceover(text="Our active token is 'cat'. Let's trace how it becomes a dense mathematical vector.") as tracker:
            active_box = token_boxes[1][0]
            self.play(active_box.animate.set_color(BLUE).set_fill(BLUE, opacity=0.2), run_time=tracker.duration)
            self.wait(1)

        # =========================================================================
        # 2. Embedding Matrix Multiplication
        # =========================================================================
        with self.voiceover(text="First, the token acts as an index into an embedding matrix.") as tracker:
            embedding_text = Text("Embedding Matrix (W_E)", color=PURPLE).scale(0.6).move_to(DOWN*1)
            self.play(Write(embedding_text), run_time=tracker.duration)

        # Let's show a fake 1x3 one hot vector
        one_hot = Matrix([[0, 1, 0]], v_buff=0.8, h_buff=0.8).set_color(BLACK)
        one_hot.scale(0.7).next_to(token_boxes[1], DOWN, buff=1)

        with self.voiceover(text="We represent 'cat' as a one-hot vector where only the second position is active.") as tracker:
            self.play(Write(one_hot), run_time=tracker.duration)

        # W_E matrix (3x3 for simplicity)
        w_e = Matrix([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]).set_color(PURPLE).scale(0.7).next_to(one_hot, DOWN, buff=1)

        with self.voiceover(text="This one-hot vector is multiplied by the embedding weight matrix W_E.") as tracker:
            self.play(FadeIn(w_e), run_time=tracker.duration)

        # Calculate result: [0, 1, 0] * W_E = [0.4, 0.5, 0.6]
        emb_res = Matrix([[0.4, 0.5, 0.6]]).set_color(GREEN).scale(0.7).next_to(w_e, RIGHT, buff=1)

        with self.voiceover(text="Multiplying the vector extracts the corresponding row from the matrix.") as tracker:
            self.play(Write(emb_res), run_time=tracker.duration)
            self.wait(2)

        with self.voiceover(text="Notice how the 1 in the one-hot vector simply pulls out the second row of the embedding matrix.") as tracker:
            w_e_entries = w_e.get_entries()
            second_row = VGroup(w_e_entries[3], w_e_entries[4], w_e_entries[5])
            self.play(second_row.animate.set_color(GREEN), run_time=tracker.duration)
            self.wait(2)

        # Clear space for next step
        with self.voiceover(text="Now we have our input embedding vector for 'cat'.") as tracker:
            self.play(
                FadeOut(one_hot),
                FadeOut(w_e),
                FadeOut(embedding_text),
                emb_res.animate.move_to(UP*0.5 + LEFT*3),
                token_boxes.animate.shift(UP*0.5)
            )

        # =========================================================================
        # 3. Query Matrix Multiplication (Q = X * W_Q)
        # =========================================================================

        w_q_text = Text("Query Weight Matrix (W_Q)", color=PURPLE).scale(0.6).next_to(emb_res, UP, buff=0.5)
        w_q = Matrix([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 0.5]
        ]).set_color(PURPLE).scale(0.7).next_to(emb_res, RIGHT, buff=0.5)

        with self.voiceover(text="Next, to participate in self-attention, we project this embedding into a query vector.") as tracker:
            self.play(Write(w_q_text), FadeIn(w_q), run_time=tracker.duration)

        with self.voiceover(text="We multiply the embedding vector by the Query weight matrix W_Q.") as tracker:
            # emb_res is [0.4, 0.5, 0.6]
            # W_Q is
            # 1.0 0.0
            # 0.5 1.0
            # 0.0 0.5
            # Result: [0.4(1) + 0.5(0.5) + 0.6(0) , 0.4(0) + 0.5(1) + 0.6(0.5)] = [0.4+0.25+0, 0+0.5+0.3] = [0.65, 0.8]
            self.wait(1)

        q_res = Matrix([[0.65, 0.8]]).set_color(GREEN).scale(0.7).next_to(w_q, RIGHT, buff=1)

        with self.voiceover(text="Here, the token 'cat' is multiplied by the Query weight matrix W_Q to produce the query vector q.") as tracker:
            self.play(Write(q_res), run_time=tracker.duration)
            self.wait(2)

        with self.voiceover(text="This resulting query vector represents what the 'cat' token is looking for in other parts of the sentence.") as tracker:
            self.wait(2)

        # Clear space
        with self.voiceover(text="Similar operations happen for Keys and Values, using W_K and W_V.") as tracker:
            self.play(
                FadeOut(w_q_text),
                FadeOut(w_q),
                FadeOut(emb_res),
                q_res.animate.move_to(DOWN*1)
            )

        # =========================================================================
        # 4. Forward vs Backward Pass Diagram
        # =========================================================================

        with self.voiceover(text="Now let's zoom out and look at the whole data flow.") as tracker:
            self.play(FadeOut(token_boxes), FadeOut(q_res))

        # Draw forward pass
        box_emb = Rectangle(width=3, height=1, color=BLACK).set_fill(WHITE, opacity=1).move_to(UP*2 + LEFT*3)
        text_emb = Text("Embedding (W_E)", color=BLACK).scale(0.5).move_to(box_emb)
        vg_emb = VGroup(box_emb, text_emb)

        box_attn = Rectangle(width=3, height=1, color=BLACK).set_fill(WHITE, opacity=1).move_to(UP*0 + LEFT*3)
        text_attn = Text("Attention (W_Q, W_K, W_V, W_O)", color=BLACK).scale(0.4).move_to(box_attn)
        vg_attn = VGroup(box_attn, text_attn)

        box_ffn = Rectangle(width=3, height=1, color=BLACK).set_fill(WHITE, opacity=1).move_to(DOWN*2 + LEFT*3)
        text_ffn = Text("FFN", color=BLACK).scale(0.5).move_to(box_ffn)
        vg_ffn = VGroup(box_ffn, text_ffn)

        arr_fw1 = Arrow(box_emb.get_bottom(), box_attn.get_top(), color=BLACK)
        arr_fw2 = Arrow(box_attn.get_bottom(), box_ffn.get_top(), color=BLACK)

        with self.voiceover(text="In the forward pass, data flows from the Embedding layer, through Attention, and into the Feed Forward Network.") as tracker:
            self.play(FadeIn(vg_emb), FadeIn(vg_attn), FadeIn(vg_ffn))
            self.play(FadeIn(arr_fw1), FadeIn(arr_fw2), run_time=tracker.duration)
            self.wait(1)

        arr_bw1 = DashedLine(box_ffn.get_right() + RIGHT*0.1, box_attn.get_right() + RIGHT*0.1, color=RED).add_tip()
        arr_bw2 = DashedLine(box_attn.get_right() + RIGHT*0.1, box_emb.get_right() + RIGHT*0.1, color=RED).add_tip()

        with self.voiceover(text="During backprop, the gradient flows back from the loss function, propagating error signals backwards.") as tracker:
            self.play(FadeIn(arr_bw1), FadeIn(arr_bw2), run_time=tracker.duration)
            self.wait(1)

        # Show Gradient Math
        grad_math = MathTex(
            r"\frac{\partial L}{\partial W_Q} = \frac{\partial L}{\partial q} \cdot \frac{\partial q}{\partial W_Q}",
            color=BLACK
        ).scale(0.8).move_to(UP*1 + RIGHT*3)

        with self.voiceover(text="Using the chain rule, we calculate how much each weight contributed to the final error.") as tracker:
            self.play(Write(grad_math), run_time=tracker.duration)
            self.wait(2)

        grad_math_2 = MathTex(
            r"\Delta W_Q = - \alpha \frac{\partial L}{\partial W_Q}",
            color=BLACK
        ).scale(0.8).next_to(grad_math, DOWN, buff=0.5)

        with self.voiceover(text="We then update the weights, like W_Q, W_K, W_V, W_O, and FFN weights, to minimize this loss.") as tracker:
            self.play(Write(grad_math_2), run_time=tracker.duration)
            self.wait(2)

        # Highlight updated weights
        with self.voiceover(text="All these learned matrices are adjusted step-by-step during training.") as tracker:
            self.play(
                vg_emb.animate.set_fill(GREEN, opacity=0.2),
                vg_attn.animate.set_fill(GREEN, opacity=0.2),
                vg_ffn.animate.set_fill(GREEN, opacity=0.2),
                run_time=tracker.duration
            )
            self.wait(1)

        self.play(FadeOut(VGroup(vg_emb, vg_attn, vg_ffn, arr_fw1, arr_fw2, arr_bw1, arr_bw2, grad_math, grad_math_2)))

        # =========================================================================
        # 5. Key Interview Insight
        # =========================================================================

        callout_box = Rectangle(width=10, height=4, color=RED).set_fill(WHITE, opacity=1)
        callout_title = Text("Key Interview Gotcha", color=RED).scale(0.8).next_to(callout_box.get_top(), DOWN, buff=0.3)

        callout_text = Text(
            "An embedding layer lookup is mathematically equivalent\n"
            "to multiplying a one-hot encoded vector by a weight matrix.\n"
            "In practice, frameworks optimize this as a simple index lookup O(1)\n"
            "rather than doing an O(N) matrix multiplication.",
            color=BLACK,
            t2w={'embedding layer lookup': BOLD, 'one-hot encoded vector': BOLD, 'index lookup O(1)': BOLD}
        ).scale(0.5).next_to(callout_title, DOWN, buff=0.5)

        insight_group = VGroup(callout_box, callout_title, callout_text)

        with self.voiceover(text="Here is a common interview question. Is an embedding layer a matrix multiplication?") as tracker:
            self.play(FadeIn(callout_box), Write(callout_title), run_time=tracker.duration)

        with self.voiceover(text="The answer is yes, mathematically. It is equivalent to multiplying a one-hot vector by the weight matrix.") as tracker:
            self.play(Write(callout_text[:83]), run_time=tracker.duration)

        with self.voiceover(text="However, in practice, deep learning frameworks optimize this as a simple O(1) index lookup to save compute, rather than an O(N) matrix multiplication.") as tracker:
            self.play(Write(callout_text[83:]), run_time=tracker.duration)
            self.wait(3)

        with self.voiceover(text="That concludes our deep dive into tokenization and input embeddings.") as tracker:
            self.play(FadeOut(insight_group))
            self.wait(1)
