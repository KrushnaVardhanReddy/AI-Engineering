from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class TransformerDataFlowScene(VoiceoverScene):
    def construct(self):
        # 7. Aesthetic (Whiteboard Style):
        self.camera.background_color = WHITE

        # 1. Audio Narration Setup
        self.set_speech_service(GTTSService())

        # =========================================================================
        # Introduction
        # =========================================================================
        with self.voiceover(text="Welcome to Day 42. Today we will explore the end-to-end data flow in a Transformer architecture. We are taking a slow, mathematical deep dive into how information travels through a single attention head, tracing an individual token step-by-step.") as tracker:
            title = Text("Transformer Data Flow Deep-Dive", color=BLACK).scale(1.2)
            self.play(Write(title))
            self.wait(2)
            self.play(title.animate.to_edge(UP))
            self.wait(2)

        # =========================================================================
        # 2. Input Setup
        # =========================================================================
        tokens = ["The", "cat", "sat"]
        boxes = VGroup()
        labels = VGroup()
        for i, t in enumerate(tokens):
            box = Rectangle(height=1.5, width=2.5, color=BLACK)
            box.move_to(RIGHT * (i - 1) * 3)
            label = Text(t, color=BLACK)
            label.move_to(box.get_center())
            boxes.add(box)
            labels.add(label)

        token_group = VGroup(boxes, labels).next_to(title, DOWN, buff=1)

        with self.voiceover(text="We start with three tokens representing our sequence: 'The', 'cat', and 'sat'.") as tracker:
            self.play(FadeIn(boxes), Write(labels))
            self.wait(3)

        active_idx = 1
        with self.voiceover(text="In this detailed animation, we will trace the journey of the active token, 'cat', highlighted here in blue. We will see exactly what happens to its mathematical representation at each stage.") as tracker:
            self.play(
                boxes[active_idx].animate.set_fill(BLUE, opacity=0.3),
                boxes[active_idx].animate.set_color(BLUE)
            )
            self.wait(3)

        with self.voiceover(text="First, each token is converted into an embedding vector. For our calculations, let's represent the token 'cat' as a simple 2 by 2 input matrix, X, to clearly show how matrices interact.") as tracker:
            self.play(token_group.animate.scale(0.5).to_corner(UL))
            self.wait(2)

        # =========================================================================
        # 3. Data Flow - Forward Pass: Q, K, V
        # =========================================================================
        # X matrix
        input_matrix_label = MathTex("X_{cat} = ", color=BLACK).move_to(LEFT * 4 + UP * 1)
        input_matrix = IntegerMatrix([[2, 1], [0, 3]], v_buff=0.8, h_buff=0.8).next_to(input_matrix_label, RIGHT)
        input_matrix.set_color(BLACK)
        input_group = VGroup(input_matrix_label, input_matrix)

        arrow_in = Arrow(start=token_group[0][active_idx].get_bottom(), end=input_matrix_label.get_top(), color=BLACK)

        with self.voiceover(text="The input matrix for 'cat' arrives at the attention mechanism. Here is our 2 by 2 matrix X.") as tracker:
            self.play(GrowArrow(arrow_in))
            self.play(Write(input_group))
            self.wait(3)

        # ---------------------------------------------------------
        # Query Matrix Multiplication
        # ---------------------------------------------------------
        wq_label = MathTex("W_Q = ", color=PURPLE).move_to(RIGHT * 1 + UP * 1)
        wq_matrix = IntegerMatrix([[1, 0], [0, 1]], v_buff=0.8, h_buff=0.8).next_to(wq_label, RIGHT)
        wq_matrix.set_color(PURPLE)
        wq_group = VGroup(wq_label, wq_matrix)

        with self.voiceover(text="The token 'cat' is first multiplied by the Query weight matrix, W_Q, a learned parameter, to produce the Query representation. Let's use the identity matrix for W_Q.") as tracker:
            self.play(Write(wq_group))
            self.wait(3)

        q_label = MathTex("Q_{cat} = X_{cat} W_Q = ", color=GREEN).move_to(LEFT * 2 + DOWN * 1.5)
        q_matrix = IntegerMatrix([[2, 1], [0, 3]], v_buff=0.8, h_buff=0.8).next_to(q_label, RIGHT)
        q_matrix.set_color(GREEN)
        q_group = VGroup(q_label, q_matrix)

        with self.voiceover(text="We perform the matrix multiplication to compute the Query vector, Q. Notice how the values from X map exactly over because W_Q is the identity.") as tracker:
            self.play(Write(q_label))
            self.play(input_matrix.get_entries().animate.set_color(BLUE), wq_matrix.get_entries().animate.set_color(BLUE))
            self.play(Write(q_matrix.get_brackets()), Write(q_matrix.get_entries()))
            self.play(input_matrix.get_entries().animate.set_color(BLACK), wq_matrix.get_entries().animate.set_color(PURPLE))
            self.wait(3)

        with self.voiceover(text="We now have our Query matrix. Next, we will calculate the Key matrix.") as tracker:
            self.play(FadeOut(wq_group), FadeOut(q_group))
            self.wait(2)

        # ---------------------------------------------------------
        # Key Matrix Multiplication
        # ---------------------------------------------------------
        wk_label = MathTex("W_K = ", color=PURPLE).move_to(RIGHT * 1 + UP * 1)
        wk_matrix = IntegerMatrix([[0, 1], [1, 0]], v_buff=0.8, h_buff=0.8).next_to(wk_label, RIGHT)
        wk_matrix.set_color(PURPLE)
        wk_group = VGroup(wk_label, wk_matrix)

        with self.voiceover(text="Now we multiply the input matrix X by the Key weight matrix, W_K. Let's use a permutation matrix for W_K.") as tracker:
            self.play(Write(wk_group))
            self.wait(3)

        k_label = MathTex("K_{cat} = X_{cat} W_K = ", color=GREEN).move_to(LEFT * 2 + DOWN * 1.5)
        # X * W_K: [[2,1],[0,3]] * [[0,1],[1,0]] = [[1,2],[3,0]]
        k_matrix = IntegerMatrix([[1, 2], [3, 0]], v_buff=0.8, h_buff=0.8).next_to(k_label, RIGHT)
        k_matrix.set_color(GREEN)
        k_group = VGroup(k_label, k_matrix)

        with self.voiceover(text="By multiplying X and W_K, we compute the Key vector, K. The permutation matrix swaps the columns of our input.") as tracker:
            self.play(Write(k_label))
            self.play(input_matrix.get_entries().animate.set_color(BLUE), wk_matrix.get_entries().animate.set_color(BLUE))
            self.play(Write(k_matrix.get_brackets()), Write(k_matrix.get_entries()))
            self.play(input_matrix.get_entries().animate.set_color(BLACK), wk_matrix.get_entries().animate.set_color(PURPLE))
            self.wait(3)

        with self.voiceover(text="We now have our Key matrix. Finally, we calculate the Value matrix.") as tracker:
            self.play(FadeOut(wk_group), FadeOut(k_group))
            self.wait(2)

        # ---------------------------------------------------------
        # Value Matrix Multiplication
        # ---------------------------------------------------------
        wv_label = MathTex("W_V = ", color=PURPLE).move_to(RIGHT * 1 + UP * 1)
        wv_matrix = IntegerMatrix([[2, 0], [0, 2]], v_buff=0.8, h_buff=0.8).next_to(wv_label, RIGHT)
        wv_matrix.set_color(PURPLE)
        wv_group = VGroup(wv_label, wv_matrix)

        with self.voiceover(text="Lastly, we multiply the input matrix X by the Value weight matrix, W_V. We will use a scaling matrix of factor 2.") as tracker:
            self.play(Write(wv_group))
            self.wait(3)

        v_label = MathTex("V_{cat} = X_{cat} W_V = ", color=GREEN).move_to(LEFT * 2 + DOWN * 1.5)
        # X * W_V: [[2,1],[0,3]] * [[2,0],[0,2]] = [[4,2],[0,6]]
        v_matrix = IntegerMatrix([[4, 2], [0, 6]], v_buff=0.8, h_buff=0.8).next_to(v_label, RIGHT)
        v_matrix.set_color(GREEN)
        v_group = VGroup(v_label, v_matrix)

        with self.voiceover(text="Multiplying X and W_V gives us the Value vector, V, which scales our original input.") as tracker:
            self.play(Write(v_label))
            self.play(input_matrix.get_entries().animate.set_color(BLUE), wv_matrix.get_entries().animate.set_color(BLUE))
            self.play(Write(v_matrix.get_brackets()), Write(v_matrix.get_entries()))
            self.play(input_matrix.get_entries().animate.set_color(BLACK), wv_matrix.get_entries().animate.set_color(PURPLE))
            self.wait(3)

        with self.voiceover(text="Excellent. We have now transformed the 'cat' token into Query, Key, and Value representations.") as tracker:
            self.play(
                FadeOut(VGroup(input_group, arrow_in, wv_group, v_group))
            )
            self.wait(2)

        # =========================================================================
        # 3c: Attention Scores & Softmax Walkthrough
        # =========================================================================
        with self.voiceover(text="Now we compute the attention scores. The Query of 'cat' is dot-producted with the Keys of all tokens to determine relevance.") as tracker:
            q_k_formula = MathTex(r"S = Q_{cat} \cdot K^T", color=BLACK).move_to(UP * 2)
            self.play(Write(q_k_formula))
            self.wait(3)

        with self.voiceover(text="Let's assume our dot products with the keys for 'The', 'cat', and 'sat' yield the raw scores: 2, 8, and 5.") as tracker:
            raw_scores = MathTex("Scores = [2, 8, 5]", color=BLACK).next_to(q_k_formula, DOWN, buff=0.5)
            self.play(Write(raw_scores))
            self.wait(3)

        with self.voiceover(text="Next, these scores are scaled by the square root of the key dimension, d_k. Let's assume the square root of d_k is 2.") as tracker:
            scaled_scores = MathTex(r"\text{Scaled} = \frac{[2, 8, 5]}{2} = [1, 4, 2.5]", color=BLACK).next_to(raw_scores, DOWN, buff=0.5)
            self.play(Write(scaled_scores))
            self.wait(3)

        with self.voiceover(text="We then apply the Softmax function over the sequence dimension to normalize these scores into probabilities. This means we exponentiate each value and divide by the sum of all exponents.") as tracker:
            softmax_formula = MathTex(r"\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}", color=BLACK).next_to(scaled_scores, DOWN, buff=0.5)
            self.play(Write(softmax_formula))
            self.wait(3)

        with self.voiceover(text="The resulting attention weights sum to 1. Notice how 'cat' strongly attends to itself with an 80 percent probability, and pays less attention to 'The' and 'sat'.") as tracker:
            final_weights = MathTex(r"\approx [0.04, 0.80, 0.16]", color=GREEN).next_to(softmax_formula, DOWN, buff=0.5)
            self.play(Write(final_weights))
            self.wait(4)

        with self.voiceover(text="These weights are used to compute a weighted sum of the Value vectors for all tokens. This weighted sum is the final output of the attention head.") as tracker:
            out_box = Text("Attention Output", color=GREEN).move_to(DOWN * 3.5)
            arrow_out = Arrow(start=final_weights.get_bottom(), end=out_box.get_top(), color=BLACK)
            self.play(GrowArrow(arrow_out), FadeIn(out_box))
            self.wait(3)

        self.play(FadeOut(VGroup(q_k_formula, raw_scores, scaled_scores, softmax_formula, final_weights, arrow_out, out_box)))

        # =========================================================================
        # 4. Forward Pass vs. Reverse (Backpropagation)
        # =========================================================================
        layer1 = Rectangle(width=2, height=1, color=BLACK).move_to(LEFT * 4)
        l1_text = Text("Input X", color=BLACK).scale(0.5).move_to(layer1.get_center())

        layer2 = Rectangle(width=2, height=1, color=PURPLE).move_to(ORIGIN)
        l2_text = Text("Attention W", color=BLACK).scale(0.4).move_to(layer2.get_center())

        layer3 = Rectangle(width=2, height=1, color=BLACK).move_to(RIGHT * 4)
        l3_text = Text("Output Y", color=BLACK).scale(0.5).move_to(layer3.get_center())

        fwd_arrow1 = Arrow(start=layer1.get_right(), end=layer2.get_left(), color=GREEN)
        fwd_arrow2 = Arrow(start=layer2.get_right(), end=layer3.get_left(), color=GREEN)

        with self.voiceover(text="In the forward pass, data flows continuously from the input, through the attention mechanism and its weight matrices, and out to the subsequent layers.") as tracker:
            self.play(FadeIn(VGroup(layer1, l1_text, layer2, l2_text, layer3, l3_text)))
            self.play(GrowArrow(fwd_arrow1))
            self.play(GrowArrow(fwd_arrow2))
            self.wait(3)

        bwd_arrow2 = DashedLine(start=layer3.get_bottom(), end=layer2.get_bottom(), color=RED).add_tip()
        bwd_arrow2.shift(DOWN * 0.5)
        bwd_arrow1 = DashedLine(start=layer2.get_bottom(), end=layer1.get_bottom(), color=RED).add_tip()
        bwd_arrow1.shift(DOWN * 0.5)

        with self.voiceover(text="During backprop, the gradient flows back from the loss function, reversing the path, to update our weights.") as tracker:
            self.play(Create(bwd_arrow2))
            self.play(Create(bwd_arrow1))
            self.wait(3)

        # Full Gradient derivation
        with self.voiceover(text="Let's derive the gradient for the Query weight matrix W_Q step by step. We start with the chain rule.") as tracker:
            grad_step1 = MathTex(
                r"\frac{\partial L}{\partial W_Q} = \frac{\partial L}{\partial Q} \cdot \frac{\partial Q}{\partial W_Q}",
                color=RED
            ).move_to(DOWN * 2)
            self.play(Write(grad_step1))
            self.wait(3)

        with self.voiceover(text="Since Q is equal to X times W_Q, the partial derivative of Q with respect to W_Q is simply the input X.") as tracker:
            grad_step2 = MathTex(
                r"\text{Since } Q = X W_Q, \quad \frac{\partial Q}{\partial W_Q} = X",
                color=BLACK
            ).next_to(grad_step1, DOWN, buff=0.3)
            self.play(Write(grad_step2))
            self.wait(3)

        with self.voiceover(text="Substituting this back in, and accounting for matrix dimensions, we get the final gradient for W_Q.") as tracker:
            grad_step3 = MathTex(
                r"\frac{\partial L}{\partial W_Q} = X^T \left( \frac{\partial L}{\partial Q} \right)",
                color=RED
            ).next_to(grad_step2, DOWN, buff=0.3)
            self.play(Write(grad_step3))
            self.play(layer2.animate.set_fill(PURPLE, opacity=0.3))
            self.wait(4)

        self.play(FadeOut(VGroup(layer1, l1_text, layer2, l2_text, layer3, l3_text, fwd_arrow1, fwd_arrow2, bwd_arrow1, bwd_arrow2, grad_step1, grad_step2, grad_step3)))
        self.play(FadeOut(token_group))

        # =========================================================================
        # 5. Key Interview Insight
        # =========================================================================
        insight_box = Rectangle(width=10, height=4, color=BLACK)
        insight_title = Text("Interview Gotcha!", color=RED).scale(1.2).next_to(insight_box.get_top(), DOWN, buff=0.5)
        insight_text = Text(
            "Softmax is applied over the sequence dimension (Keys),\nnot the feature dimension. Forgetting this means\nattention weights won't sum to 1 across the sequence.",
            color=BLACK,
            t2c={"sequence dimension (Keys)": BLUE, "feature dimension": PURPLE, "sum to 1": GREEN}
        ).scale(0.6).next_to(insight_title, DOWN, buff=0.5)

        with self.voiceover(text="Finally, a critical interview gotcha. When computing attention, the softmax is applied over the sequence dimension, meaning across the Keys for a given Query, not the feature dimension.") as tracker:
            self.play(Create(insight_box))
            self.play(Write(insight_title))
            self.play(Write(insight_text))
            self.wait(4)

        with self.voiceover(text="Forgetting this detail means your attention weights will not sum to one across the tokens, completely breaking the mechanism.") as tracker:
            self.wait(4)

        with self.voiceover(text="This concludes our deep dive into the Transformer data flow. Thank you for watching.") as tracker:
            self.play(FadeOut(VGroup(insight_box, insight_title, insight_text, title)))
            self.wait(3)
