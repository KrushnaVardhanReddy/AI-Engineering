from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class AttentionMasking(VoiceoverScene):
    def construct(self):
        # 7. Aesthetic (Whiteboard Style)
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # ==============================================================================
        # 1. Introduction and Input Setup
        # ==============================================================================
        with self.voiceover(text="Welcome to this deep-dive on Attention Masking. In this video, we will explore the critical differences between Padding Masks and Causal Masks, tracing the exact data flow of a sequence through the attention mechanism. This is a crucial concept for understanding how large language models like GPT and BERT process text, and a very common topic in advanced machine learning interviews. Let us begin.") as tracker:
            title = Text("Attention Masking: Padding vs Causal", color=BLACK, font_size=48).to_edge(UP)
            self.play(Write(title))
            self.wait(2)

        with self.voiceover(text="We start with three input tokens representing a simple sequence: 'The', 'cat', and 'sat'. In a real model, these tokens are converted into high-dimensional embeddings, but for our visual demonstration, we will use small, simplified vectors.") as tracker:
            self.play(FadeOut(title))

            tokens_text = ["The", "cat", "sat"]
            token_boxes = VGroup()
            for i, text in enumerate(tokens_text):
                box = Rectangle(width=2, height=1, color=BLACK)
                t = Text(text, color=BLACK).move_to(box.get_center())
                group = VGroup(box, t)
                group.shift(RIGHT * (i - 1) * 2.5)
                token_boxes.add(group)

            token_boxes.shift(UP * 2.5)
            self.play(FadeIn(token_boxes))
            self.wait(2)

        with self.voiceover(text="Let us focus our attention on the token 'cat'. We highlight the active token in blue. We will trace how this specific token gathers context from its surroundings and how masks dictate what it is allowed to see.") as tracker:
            active_index = 1
            self.play(token_boxes[active_index][0].animate.set_fill(BLUE, opacity=0.3))
            self.wait(2)

        # ==============================================================================
        # 2. Data Flow - Projection Step (Q, K, V)
        # ==============================================================================
        with self.voiceover(text="The first step in the self-attention mechanism is to project our input embeddings into three distinct spaces: Queries, Keys, and Values. The input vector for 'cat' arrives as a row in our input matrix, which is multiplied by the Query weight matrix, W Q.") as tracker:
            arrow_in = Arrow(start=token_boxes[active_index].get_bottom(), end=token_boxes[active_index].get_bottom() + DOWN*1.5, color=BLACK)
            self.play(FadeIn(arrow_in))
            self.wait(1)

            x_matrix_tex = MathTex(
                r"X = \begin{bmatrix} 1 & 0 & -1 \\ 2 & 1 & 0 \\ 0 & -1 & 1 \end{bmatrix}", color=BLUE
            ).shift(LEFT * 4 + DOWN * 1)

            self.play(Write(x_matrix_tex))
            self.wait(2)

        with self.voiceover(text="Notice the input matrix X, where each row represents a token. The second row, two, one, zero, corresponds to our active token, 'cat'. We multiply this matrix by the Query weight matrix.") as tracker:
            times_tex = MathTex(r"\times", color=BLACK).next_to(x_matrix_tex, RIGHT)
            wq_matrix_tex = MathTex(
                r"W_Q = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ -1 & 0 & 1 \end{bmatrix}", color=PURPLE
            ).next_to(times_tex, RIGHT)

            self.play(Write(times_tex), Write(wq_matrix_tex))
            self.wait(2)

        with self.voiceover(text="Let us compute the resulting Query matrix Q. For the first cell of the second row, we take the dot product of the 'cat' input vector, two, one, zero, with the first column of W Q, one, zero, negative one. Two times one is two, plus one times zero is zero, plus zero times negative one is zero. The result is two.") as tracker:
            equals_tex = MathTex(r"=", color=BLACK).next_to(wq_matrix_tex, RIGHT)
            q_matrix_tex = MathTex(
                r"Q = \begin{bmatrix} 2 & 0 & 0 \\ 2 & 1 & 2 \\ -1 & -1 & 1 \end{bmatrix}", color=GREEN
            ).next_to(equals_tex, RIGHT)

            self.play(Write(equals_tex))
            self.wait(1)
            self.play(Write(q_matrix_tex))
            self.wait(2)

        with self.voiceover(text="Following the exact same procedure, we multiply the input matrix X by the Key weight matrix W K, and the Value weight matrix W V, to produce the Key matrix K and the Value matrix V.") as tracker:
            self.play(FadeOut(VGroup(arrow_in, x_matrix_tex, times_tex, wq_matrix_tex, equals_tex, q_matrix_tex)))

            k_matrix_tex = MathTex(
                r"K = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 2 & 1 \\ 1 & 0 & -1 \end{bmatrix}", color=GREEN
            ).shift(LEFT * 3)

            v_matrix_tex = MathTex(
                r"V = \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ -1 & 1 & 0 \end{bmatrix}", color=GREEN
            ).shift(RIGHT * 3)

            self.play(Write(k_matrix_tex), Write(v_matrix_tex))
            self.wait(3)
            self.play(FadeOut(k_matrix_tex), FadeOut(v_matrix_tex))

        # ==============================================================================
        # 3. Attention Scores and Masking
        # ==============================================================================
        with self.voiceover(text="Now we arrive at the core of the attention mechanism. We compute the raw attention scores by taking the dot product of the Query matrix Q and the transposed Key matrix K. This results in a three by three matrix representing the unnormalized attention weights between all pairs of our three tokens.") as tracker:
            qk_text = MathTex(r"\text{Raw Scores} = QK^T", color=BLACK).move_to(UP * 0.5)
            scores_matrix = MathTex(
                r"\begin{bmatrix} "
                r"2 & 0 & 2 \\ "
                r"3 & 4 & 1 \\ "
                r"-2 & -1 & -2 "
                r"\end{bmatrix}", color=BLACK
            ).next_to(qk_text, DOWN)

            self.play(Write(qk_text))
            self.play(Write(scores_matrix))
            self.wait(2)

        with self.voiceover(text="Look closely at this matrix. The rows represent the queries, or the tokens that are 'looking', and the columns represent the keys, the tokens being 'looked at'. The middle row corresponds to 'cat' looking at 'The', 'cat', and 'sat'.") as tracker:
            cat_row_rect = SurroundingRectangle(scores_matrix[0][5:8], color=BLUE, buff=0.1)
            self.play(Create(cat_row_rect))
            self.wait(2)
            self.play(FadeOut(cat_row_rect))

        with self.voiceover(text="But here we introduce masking. Consider a Causal Mask, which is fundamentally required in autoregressive decoders like GPT. A causal mask ensures that a token can only attend to previous tokens and itself, preventing information from the future from leaking backwards during training.") as tracker:
            causal_title = Text("Causal Masking", color=RED, font_size=36).to_edge(LEFT, buff=1).shift(UP * 2)
            self.play(Write(causal_title))
            self.wait(2)

        with self.voiceover(text="For our sequence, 'The' cannot look at 'cat' or 'sat'. 'cat' cannot look at 'sat'. These forbidden connections correspond strictly to the upper triangular elements of our score matrix, above the main diagonal.") as tracker:
            # Highlight upper triangular elements
            # Cell indices in the MathTex matrix (ignoring brackets):
            # Row 0: 2 (idx 1), 0 (idx 2), 2 (idx 3) -> Upper tri are idx 2, 3
            # Row 1: 3 (idx 4), 4 (idx 5), 1 (idx 6) -> Upper tri is idx 6
            # Row 2: -2, -1, -2 (idx 7,8,9)

            # Using precise indexing based on typical MathTex parsing for \begin{bmatrix}
            # scores_matrix[0] contains the elements. Let's find them dynamically or use tight bounding boxes.

            # The indices for the upper right elements are:
            # Row 1, Col 2 -> 0
            # Row 1, Col 3 -> 2
            # Row 2, Col 3 -> 1

            # Actually, let's just create a new matrix with colors to highlight safely.
            highlighted_scores = MathTex(
                r"\begin{bmatrix} "
                r"2 & 0 & 2 \\ "
                r"3 & 4 & 1 \\ "
                r"-2 & -1 & -2 "
                r"\end{bmatrix}", color=BLACK
            ).next_to(qk_text, DOWN)

            r1 = SurroundingRectangle(highlighted_scores[0][2:4], color=RED, buff=0.05)
            r2 = SurroundingRectangle(highlighted_scores[0][6], color=RED, buff=0.05)

            self.play(Transform(scores_matrix, highlighted_scores))
            self.play(Create(r1), Create(r2))
            self.wait(2)
            self.play(FadeOut(r1), FadeOut(r2))

        with self.voiceover(text="To enforce this causal constraint, we apply the Causal Mask by adding negative infinity to these specific forbidden cells. Watch as the upper triangular elements are updated to negative infinity.") as tracker:
            masked_matrix = MathTex(
                r"\begin{bmatrix} "
                r"2 & -\infty & -\infty \\ "
                r"3 & 4 & -\infty \\ "
                r"-2 & -1 & -2 "
                r"\end{bmatrix}", color=BLACK
            ).next_to(qk_text, DOWN)

            self.play(TransformMatchingTex(scores_matrix, masked_matrix))
            self.wait(2)

            # Highlight the updated cells with a surrounding rectangle
            # We highlight the -\infty symbols
            rect1 = SurroundingRectangle(masked_matrix[0][2:4], color=RED, buff=0.05) # First row -inf, -inf
            rect2 = SurroundingRectangle(masked_matrix[0][6], color=RED, buff=0.05)   # Second row -inf
            self.play(Create(rect1), Create(rect2))
            self.wait(2)
            self.play(FadeOut(rect1), FadeOut(rect2))

        with self.voiceover(text="Alternatively, we have Padding Masks. In tasks like text classification using an encoder like BERT, bidirectional context is allowed. However, we often batch sequences of different lengths together by padding them with special tokens.") as tracker:
            self.play(FadeOut(causal_title))
            padding_title = Text("Padding Masking", color=BLUE, font_size=36).to_edge(LEFT, buff=1).shift(UP * 2)
            self.play(Write(padding_title))
            self.wait(2)

        with self.voiceover(text="If 'sat' was actually a padding token, we would want to prevent all real tokens from attending to it, because padding carries no semantic meaning. A Padding mask would therefore apply negative infinity to the entire column corresponding to the pad token, regardless of position.") as tracker:
            pad_masked_matrix = MathTex(
                r"\begin{bmatrix} "
                r"2 & 0 & -\infty \\ "
                r"3 & 4 & -\infty \\ "
                r"-2 & -1 & -\infty "
                r"\end{bmatrix}", color=BLACK
            ).next_to(qk_text, DOWN)

            self.play(TransformMatchingTex(masked_matrix, pad_masked_matrix))
            self.wait(2)

            rect3 = SurroundingRectangle(pad_masked_matrix[0][3], color=BLUE, buff=0.05) # Col 3
            rect4 = SurroundingRectangle(pad_masked_matrix[0][7], color=BLUE, buff=0.05) # Col 3
            rect5 = SurroundingRectangle(pad_masked_matrix[0][11], color=BLUE, buff=0.05) # Col 3
            self.play(Create(rect3), Create(rect4), Create(rect5))
            self.wait(2)
            self.play(FadeOut(rect3), FadeOut(rect4), FadeOut(rect5))

        with self.voiceover(text="Let us return to our Causal Mask for the remainder of this demonstration.") as tracker:
            self.play(FadeOut(padding_title))
            self.play(Write(causal_title))
            self.play(TransformMatchingTex(pad_masked_matrix, masked_matrix))
            self.wait(2)

        with self.voiceover(text="After applying the mask, we pass these raw scores through a Softmax function along each row. Softmax exponentiates the values. The mathematical elegance of setting the mask to negative infinity becomes clear here. E to the power of negative infinity approaches exactly zero.") as tracker:
            softmax_text = MathTex(r"\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)", color=BLACK).move_to(UP * 0.5)

            # Simulated softmax output
            probs_matrix = MathTex(
                r"\begin{bmatrix} "
                r"1.0 & 0 & 0 \\ "
                r"0.27 & 0.73 & 0 \\ "
                r"0.42 & 0.16 & 0.42 "
                r"\end{bmatrix}", color=GREEN
            ).next_to(softmax_text, DOWN)

            self.play(Transform(qk_text, softmax_text), TransformMatchingTex(masked_matrix, probs_matrix))
            self.wait(2)

        with self.voiceover(text="Because of the negative infinities, the attention probabilities for the forbidden future tokens become identically zero. The active token 'cat', in the second row, now distributes its attention weight as twenty seven percent on 'The' and seventy three percent on itself, with absolute zero weight on the future token 'sat'.") as tracker:
            prob_rect = SurroundingRectangle(probs_matrix[0][4:7], color=GREEN, buff=0.1) # Highlight second row
            self.play(Create(prob_rect))
            self.wait(3)
            self.play(FadeOut(prob_rect))

        with self.voiceover(text="These normalized probabilities are then multiplied by the Value matrix V, producing the final updated token representations, which flow out as arrows to the next stage of the transformer, such as the Feed Forward Network.") as tracker:
            arrow_out = Arrow(start=probs_matrix.get_bottom(), end=probs_matrix.get_bottom() + DOWN*1.5, color=BLACK)
            self.play(FadeIn(arrow_out))
            self.wait(2)
            self.play(FadeOut(qk_text), FadeOut(softmax_text), FadeOut(probs_matrix), FadeOut(arrow_out), FadeOut(causal_title))


        # ==============================================================================
        # 4. Forward Pass vs Reverse Pass (Backprop)
        # ==============================================================================
        with self.voiceover(text="Now that we have traced the forward pass, let us examine the reverse pass, or backpropagation. In the forward pass, data flows with solid arrows from our inputs X, through the attention weights W Q, W K, W V, to produce the output Y.") as tracker:
            node_x = Circle(radius=0.5, color=BLUE, fill_opacity=0.2).shift(LEFT * 4)
            node_w = Rectangle(width=1.5, height=1, color=PURPLE, fill_opacity=0.2).shift(UP*1)
            node_y = Circle(radius=0.5, color=GREEN, fill_opacity=0.2).shift(RIGHT * 4)

            x_label = MathTex(r"X", color=BLACK).move_to(node_x)
            w_label = MathTex(r"W_{Q,K,V}", color=BLACK).move_to(node_w)
            y_label = MathTex(r"Y", color=BLACK).move_to(node_y)

            fwd_arrow1 = Arrow(start=node_x.get_right(), end=node_w.get_left(), color=BLACK)
            fwd_arrow2 = Arrow(start=node_w.get_right(), end=node_y.get_left(), color=BLACK)

            self.play(FadeIn(node_x, x_label), FadeIn(node_w, w_label), FadeIn(node_y, y_label))
            self.play(FadeIn(fwd_arrow1), FadeIn(fwd_arrow2))
            self.wait(3)

        with self.voiceover(text="During backpropagation, the error signal or gradient flows backwards from the final loss function, indicated by dashed red arrows. We must compute how to update our weight matrices to minimize the loss.") as tracker:
            bwd_arrow1 = DashedLine(start=node_y.get_top(), end=node_w.get_right(), color=RED).add_tip(tip_length=0.2)
            bwd_arrow1.shift(UP * 0.3)

            bwd_arrow2 = DashedLine(start=node_w.get_left(), end=node_x.get_top(), color=RED).add_tip(tip_length=0.2)
            bwd_arrow2.shift(UP * 0.3)

            self.play(FadeIn(bwd_arrow1), FadeIn(bwd_arrow2))
            self.wait(2)

        with self.voiceover(text="Let us derive the gradient for the Query weight matrix, W Q. Using the chain rule, the derivative of the Loss with respect to W Q is equal to the derivative of the Loss with respect to the output Y, multiplied by the derivative of Y with respect to the Attention Scores, multiplied by the derivative of the Scores with respect to Q, and finally multiplied by the derivative of Q with respect to W Q.") as tracker:
            grad_eq = MathTex(
                r"\frac{\partial L}{\partial W_Q} = \frac{\partial L}{\partial Y} \cdot \frac{\partial Y}{\partial S} \cdot \frac{\partial S}{\partial Q} \cdot \frac{\partial Q}{\partial W_Q}",
                color=RED
            ).shift(DOWN * 1.5)

            self.play(Write(grad_eq))
            self.wait(4)

        with self.voiceover(text="Because masking sets certain elements of the scores to negative infinity before the softmax, their resulting gradients during the backward pass are exactly zero. The network learns that it receives no error signal from masked, forbidden connections, meaning it will not update weights to try and utilize information from those masked tokens.") as tracker:
            grad_eq2 = MathTex(
                r"\frac{\partial S_{ij}}{\partial W_Q} = 0 \quad \text{if } M_{ij} = -\infty",
                color=RED
            ).next_to(grad_eq, DOWN, buff=0.5)

            self.play(Write(grad_eq2))
            self.wait(4)

        with self.voiceover(text="This same chain rule principle is applied to update the Key weights W K, Value weights W V, Output weights W O, and Feed Forward Network weights. All gradients flow efficiently backward, correctly modulated by the masking.") as tracker:
            self.wait(3)

        self.play(FadeOut(VGroup(node_x, node_w, node_y, x_label, w_label, y_label, fwd_arrow1, fwd_arrow2, bwd_arrow1, bwd_arrow2, grad_eq, grad_eq2, token_boxes)))


        # ==============================================================================
        # 5. Key Interview Insight
        # ==============================================================================
        with self.voiceover(text="Before we finish, here is a highly critical interview gotcha that is frequently asked by AI engineering interviewers.") as tracker:
            box = Rectangle(width=12, height=4, color=RED, fill_opacity=0.1)
            title = Text("Key Interview Gotcha", color=RED, font_size=48).next_to(box.get_top(), DOWN, buff=0.3)
            gotcha_text = Text(
                "Masking MUST be applied BEFORE Softmax,\n"
                "by adding -∞ to the raw, unnormalized logits.\n\n"
                "If you apply the mask AFTER Softmax (e.g., multiplying by 0),\n"
                "the probabilities will no longer sum to 1,\n"
                "violating the fundamental property of attention.",
                font_size=28,
                color=BLACK,
                t2c={"BEFORE": RED, "AFTER": RED, "-∞": BLUE, "0": BLUE}
            ).next_to(title, DOWN, buff=0.4)

            gotcha_group = VGroup(box, title, gotcha_text).move_to(ORIGIN)
            self.play(FadeIn(box), Write(title))
            self.wait(1)
            self.play(Write(gotcha_text))
            self.wait(5)

        with self.voiceover(text="Always remember to mask the raw logits. That concludes our deep dive into the data flow of Attention Masking. By understanding exactly how padding and causal masks shape the probability distributions in the forward pass, and how they modulate gradients in the backward pass, you are well-equipped to design, debug, and discuss state-of-the-art transformer architectures. Thank you for watching.") as tracker:
            self.wait(4)
            self.play(FadeOut(gotcha_group))
            self.wait(2)
