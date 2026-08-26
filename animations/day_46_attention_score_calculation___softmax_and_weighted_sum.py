from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class AttentionScoreCalculation(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # --- 0. Introduction ---
        with self.voiceover(text="Welcome to this deep dive into the Transformer architecture. Today, we will explore the core of the self-attention mechanism: the attention score calculation, focusing specifically on the Softmax transformation and the final Weighted Sum.") as tracker:
            title = Text("Attention Score Calculation", color=BLACK).scale(1.2)
            subtitle = Text("Softmax and Weighted Sum", color=BLACK).next_to(title, DOWN)
            self.play(Write(title))
            self.play(Write(subtitle))
            self.wait(2)

        with self.voiceover(text="This is not just a high-level overview. We are going to trace the exact data flow for a specific token through this stage, examining the mathematical operations cell by cell. We will look at both the forward pass and the reverse backpropagation process.") as tracker:
            self.wait(2)
            self.play(FadeOut(VGroup(title, subtitle)))

        # --- 1. Input Setup ---
        with self.voiceover(text="Let us begin with our example input. We have a simple sequence of three tokens: 'The', 'cat', and 'sat'.") as tracker:
            tokens = ["The", "cat", "sat"]
            token_boxes = VGroup(*[
                VGroup(
                    Rectangle(width=2.5, height=1.2, color=BLACK),
                    Text(t, color=BLACK)
                ) for t in tokens
            ]).arrange(RIGHT, buff=0.8)
            token_boxes.move_to(UP * 2.5)
            self.play(FadeIn(token_boxes))
            self.wait(2)

        with self.voiceover(text="In this animation, we will focus our analysis on the active token, 'cat'. We want to see how 'cat' computes its contextualized representation by attending to the other tokens in the sequence.") as tracker:
            cat_box = token_boxes[1][0]
            self.play(cat_box.animate.set_fill(BLUE, opacity=0.3))
            self.wait(2)

        with self.voiceover(text="Before attention begins, each token is mapped to a dense embedding vector. Let us define a simplified 3-dimensional vector for our token 'cat'.") as tracker:
            x_label = MathTex(r"x_{\text{cat}} = ", color=BLACK)
            x_vec = Matrix([["1"], ["0"], ["2"]]).set_color(BLACK)
            x_group = VGroup(x_label, x_vec).arrange(RIGHT)
            x_group.next_to(token_boxes, DOWN, buff=1.5)

            arrow_in = Arrow(start=token_boxes[1].get_bottom(), end=x_group.get_top(), color=BLACK)
            self.play(Write(arrow_in))
            self.play(Write(x_group))
            self.wait(2)

        with self.voiceover(text="We will also assume the embedding vectors for 'The' and 'sat' have been extracted similarly. Data now flows into the linear projection stage.") as tracker:
            self.play(FadeOut(arrow_in), FadeOut(token_boxes))
            self.play(x_group.animate.to_edge(LEFT).shift(UP * 2))
            self.wait(2)

        # --- 2. Linear Projections (Q, K, V) ---
        with self.voiceover(text="To perform self-attention, we project our input vector into three distinct spaces: the Query, the Key, and the Value. We do this by multiplying the input by three learned weight matrices.") as tracker:
            wq_label = MathTex(r"W_Q", color=PURPLE)
            wk_label = MathTex(r"W_K", color=PURPLE)
            wv_label = MathTex(r"W_V", color=PURPLE)
            labels_group = VGroup(wq_label, wk_label, wv_label).arrange(RIGHT, buff=2).move_to(UP * 2 + RIGHT * 2)
            self.play(FadeIn(labels_group))
            self.wait(2)

        with self.voiceover(text="Let us compute the Query vector for 'cat' by multiplying its input vector with the Query weight matrix. The layout shows the input on the left, the weight matrix on top, and the output on the right.") as tracker:
            self.play(FadeOut(wk_label), FadeOut(wv_label), wq_label.animate.move_to(UP * 2.5 + RIGHT * 2))

            x_mat = Matrix([["1"], ["0"], ["2"]]).set_color(BLUE)
            x_mat_label = MathTex(r"x_{\text{cat}}", color=BLACK).next_to(x_mat, LEFT)
            input_g = VGroup(x_mat_label, x_mat).move_to(LEFT * 4 + DOWN * 1)

            wq_mat = Matrix([["1", "0", "1"], ["0", "1", "0"], ["1", "0", "0"]]).set_color(PURPLE)
            wq_g = VGroup(MathTex(r"W_Q", color=BLACK).next_to(wq_mat, LEFT), wq_mat).move_to(UP * 2.5 + RIGHT * 1)

            q_mat = Matrix([["0"], ["0"], ["0"]]).set_color(GREEN)
            q_mat_label = MathTex(r"q_{\text{cat}}", color=BLACK).next_to(q_mat, RIGHT)
            output_g = VGroup(q_mat, q_mat_label).move_to(RIGHT * 4 + DOWN * 1)

            self.play(
                Transform(x_group, input_g),
                FadeIn(wq_g),
                FadeOut(wq_label),
                FadeIn(output_g)
            )
            self.wait(2)

        with self.voiceover(text="We will animate this matrix multiplication cell by cell. For the first element of the output, we take the dot product of the first row of the weight matrix and the input vector.") as tracker:
            row1 = wq_mat.get_rows()[0]
            col = x_mat.get_columns()[0]

            self.play(row1.animate.set_color(RED), col.animate.set_color(RED))
            self.wait(1)

            calc1 = MathTex(r"(1 \cdot 1) + (0 \cdot 0) + (1 \cdot 2) = 1 + 0 + 2 = 3", color=BLACK).move_to(DOWN * 3)
            self.play(Write(calc1))
            self.wait(2)

            new_val1 = MathTex("3", color=GREEN).move_to(q_mat.get_entries()[0].get_center())
            self.play(Transform(q_mat.get_entries()[0], new_val1))
            self.play(FadeOut(calc1), row1.animate.set_color(PURPLE), col.animate.set_color(BLUE))
            self.wait(1)

        with self.voiceover(text="For the second element, we use the second row of the weight matrix.") as tracker:
            row2 = wq_mat.get_rows()[1]
            self.play(row2.animate.set_color(RED), col.animate.set_color(RED))
            self.wait(1)

            calc2 = MathTex(r"(0 \cdot 1) + (1 \cdot 0) + (0 \cdot 2) = 0 + 0 + 0 = 0", color=BLACK).move_to(DOWN * 3)
            self.play(Write(calc2))
            self.wait(2)

            new_val2 = MathTex("0", color=GREEN).move_to(q_mat.get_entries()[1].get_center())
            self.play(Transform(q_mat.get_entries()[1], new_val2))
            self.play(FadeOut(calc2), row2.animate.set_color(PURPLE), col.animate.set_color(BLUE))
            self.wait(1)

        with self.voiceover(text="And for the third element, we multiply the third row by the input vector.") as tracker:
            row3 = wq_mat.get_rows()[2]
            self.play(row3.animate.set_color(RED), col.animate.set_color(RED))
            self.wait(1)

            calc3 = MathTex(r"(1 \cdot 1) + (0 \cdot 0) + (0 \cdot 2) = 1 + 0 + 0 = 1", color=BLACK).move_to(DOWN * 3)
            self.play(Write(calc3))
            self.wait(2)

            new_val3 = MathTex("1", color=GREEN).move_to(q_mat.get_entries()[2].get_center())
            self.play(Transform(q_mat.get_entries()[2], new_val3))
            self.play(FadeOut(calc3), row3.animate.set_color(PURPLE), col.animate.set_color(BLUE))
            self.wait(2)

        with self.voiceover(text="We have successfully derived the Query vector for 'cat'. In a full Transformer, this process is repeated precisely for the Key and Value matrices for all tokens.") as tracker:
            self.play(FadeOut(wq_g), FadeOut(x_group))
            self.play(output_g.animate.move_to(LEFT * 4 + UP * 2))
            self.wait(2)

        with self.voiceover(text="To move forward, we will place the computed Query, Key, and Value vectors for 'cat' on the screen. Assume the Key vectors for the other tokens have also been computed.") as tracker:
            q_cat_g = output_g

            k_cat_mat = Matrix([["1"], ["2"], ["1"]]).set_color(GREEN)
            k_cat_g = VGroup(k_cat_mat, MathTex(r"k_{\text{cat}}", color=BLACK).next_to(k_cat_mat, RIGHT)).move_to(LEFT * 4 + ORIGIN)

            v_cat_mat = Matrix([["0"], ["1"], ["2"]]).set_color(GREEN)
            v_cat_g = VGroup(v_cat_mat, MathTex(r"v_{\text{cat}}", color=BLACK).next_to(v_cat_mat, RIGHT)).move_to(LEFT * 4 + DOWN * 2)

            self.play(FadeIn(k_cat_g), FadeIn(v_cat_g))
            self.wait(2)

        # --- 3. Attention Scores (Dot Product) ---
        with self.voiceover(text="Now we reach the crucial step of evaluating attention scores. We want to know how much 'cat' should focus on 'The', itself, and 'sat'.") as tracker:
            self.play(FadeOut(k_cat_g), FadeOut(v_cat_g))
            self.play(q_cat_g.animate.move_to(LEFT * 4 + UP * 2))
            self.wait(2)

        with self.voiceover(text="We compute these scores by taking the dot product of our Query vector for 'cat' with the Key vector of each token.") as tracker:
            k_the_mat = Matrix([["1"], ["0"], ["0"]]).set_color(BLUE)
            k_the_g = VGroup(MathTex(r"k_{\text{The}}", color=BLACK).next_to(k_the_mat, LEFT), k_the_mat).move_to(RIGHT * 2 + UP * 2)

            k_cat_mat2 = Matrix([["1"], ["2"], ["1"]]).set_color(BLUE)
            k_cat_g2 = VGroup(MathTex(r"k_{\text{cat}}", color=BLACK).next_to(k_cat_mat2, LEFT), k_cat_mat2).move_to(RIGHT * 2 + ORIGIN)

            k_sat_mat = Matrix([["2"], ["1"], ["0"]]).set_color(BLUE)
            k_sat_g = VGroup(MathTex(r"k_{\text{sat}}", color=BLACK).next_to(k_sat_mat, LEFT), k_sat_mat).move_to(RIGHT * 2 + DOWN * 2)

            self.play(FadeIn(k_the_g), FadeIn(k_cat_g2), FadeIn(k_sat_g))
            self.wait(2)

        with self.voiceover(text="Let us calculate the raw attention score for 'cat' attending to 'The'. This is the dot product of q_cat and k_The.") as tracker:
            dot_calc_the = MathTex(r"q_{\text{cat}} \cdot k_{\text{The}} = (3)(1) + (0)(0) + (1)(0) = 3", color=BLACK).move_to(DOWN * 3.5)
            self.play(q_cat_g.animate.set_color(RED), k_the_g.animate.set_color(RED))
            self.play(Write(dot_calc_the))
            self.wait(2)

            score_the = MathTex(r"s_{\text{cat}\to\text{The}} = 3", color=GREEN).next_to(k_the_g, RIGHT, buff=1)
            self.play(Transform(dot_calc_the, score_the))
            self.play(q_cat_g.animate.set_color(GREEN), k_the_g.animate.set_color(BLUE))
            self.wait(2)

        with self.voiceover(text="Next, 'cat' attending to itself. We take the dot product of q_cat and k_cat.") as tracker:
            dot_calc_cat = MathTex(r"q_{\text{cat}} \cdot k_{\text{cat}} = (3)(1) + (0)(2) + (1)(1) = 3 + 0 + 1 = 4", color=BLACK).move_to(DOWN * 3.5)
            self.play(q_cat_g.animate.set_color(RED), k_cat_g2.animate.set_color(RED))
            self.play(Write(dot_calc_cat))
            self.wait(2)

            score_cat = MathTex(r"s_{\text{cat}\to\text{cat}} = 4", color=GREEN).next_to(k_cat_g2, RIGHT, buff=1)
            self.play(Transform(dot_calc_cat, score_cat))
            self.play(q_cat_g.animate.set_color(GREEN), k_cat_g2.animate.set_color(BLUE))
            self.wait(2)

        with self.voiceover(text="Finally, 'cat' attending to 'sat'. The dot product of q_cat and k_sat.") as tracker:
            dot_calc_sat = MathTex(r"q_{\text{cat}} \cdot k_{\text{sat}} = (3)(2) + (0)(1) + (1)(0) = 6 + 0 + 0 = 6", color=BLACK).move_to(DOWN * 3.5)
            self.play(q_cat_g.animate.set_color(RED), k_sat_g.animate.set_color(RED))
            self.play(Write(dot_calc_sat))
            self.wait(2)

            score_sat = MathTex(r"s_{\text{cat}\to\text{sat}} = 6", color=GREEN).next_to(k_sat_g, RIGHT, buff=1)
            self.play(Transform(dot_calc_sat, score_sat))
            self.play(q_cat_g.animate.set_color(GREEN), k_sat_g.animate.set_color(BLUE))
            self.wait(2)

        # --- 4. Softmax ---
        with self.voiceover(text="These raw scores (3, 4, and 6) represent the unbounded relevance of each token. We need to convert them into a valid probability distribution that sums to exactly 1. We do this using the Softmax function.") as tracker:
            self.play(FadeOut(q_cat_g), FadeOut(k_the_g), FadeOut(k_cat_g2), FadeOut(k_sat_g))

            scores_group = VGroup(dot_calc_the, dot_calc_cat, dot_calc_sat).arrange(RIGHT, buff=1.5).move_to(UP * 2)
            self.play(scores_group.animate.move_to(UP * 2))
            self.wait(2)

            softmax_title = Text("Softmax Function", color=BLACK).to_edge(UP)
            self.play(Write(softmax_title))
            self.wait(2)

        with self.voiceover(text="The Softmax function operates by taking the exponential of each score, and then dividing by the sum of all exponentials. Let us write down the formula.") as tracker:
            softmax_formula = MathTex(r"\text{Softmax}(s_i) = \frac{e^{s_i}}{\sum_j e^{s_j}}", color=BLACK).next_to(softmax_title, DOWN, buff=0.5)
            self.play(Write(softmax_formula))
            self.wait(3)

        with self.voiceover(text="First, we calculate the denominator by exponentiating each score and summing them up. Exponential functions grow very rapidly.") as tracker:
            denom_math = MathTex(r"\sum_j e^{s_j} = e^3 + e^4 + e^6", color=BLACK).move_to(ORIGIN)
            self.play(Write(denom_math))
            self.wait(2)

            denom_eval = MathTex(r"\approx 20.09 + 54.60 + 403.43 = 478.12", color=BLACK).next_to(denom_math, DOWN, buff=0.5)
            self.play(Write(denom_eval))
            self.wait(3)

        with self.voiceover(text="Now we divide each individual exponential by this total sum to find the final attention weights.") as tracker:
            w_the = MathTex(r"a_{\text{The}} = \frac{e^3}{478.12} \approx 0.04", color=GREEN)
            w_cat = MathTex(r"a_{\text{cat}} = \frac{e^4}{478.12} \approx 0.11", color=GREEN)
            w_sat = MathTex(r"a_{\text{sat}} = \frac{e^6}{478.12} \approx 0.85", color=GREEN)

            weights_group = VGroup(w_the, w_cat, w_sat).arrange(RIGHT, buff=1.5).move_to(DOWN * 2.5)
            self.play(Write(weights_group))
            self.wait(3)

        with self.voiceover(text="Notice the power of the exponential function. Even though the raw score 6 is only twice as large as 3, its final attention weight of 0.85 completely dominates the distribution. The model pays almost all its attention to 'sat'.") as tracker:
            self.play(w_sat.animate.set_color(BLUE).scale(1.2))
            self.wait(3)
            self.play(w_sat.animate.set_color(GREEN).scale(1/1.2))

        # --- 5. Weighted Sum ---
        with self.voiceover(text="We arrive at the final stage of self-attention: the Weighted Sum. We will multiply each token's Value vector by its corresponding attention weight, and sum the results.") as tracker:
            self.play(FadeOut(softmax_title), FadeOut(softmax_formula), FadeOut(denom_math), FadeOut(denom_eval), FadeOut(scores_group))
            self.play(weights_group.animate.move_to(UP * 2.5))

            wsum_title = Text("Weighted Sum", color=BLACK).to_edge(UP)
            self.play(Write(wsum_title))
            self.wait(2)

        with self.voiceover(text="Let us bring in the Value vectors for all three tokens. Remember, these are formed by projecting the original input vectors through the W_V weight matrix.") as tracker:
            v_the_mat = Matrix([["1"], ["1"], ["0"]]).set_color(BLACK)
            v_the_g = VGroup(MathTex(r"v_{\text{The}}", color=BLACK).next_to(v_the_mat, LEFT), v_the_mat)

            v_cat_mat3 = Matrix([["0"], ["1"], ["2"]]).set_color(BLACK)
            v_cat_g3 = VGroup(MathTex(r"v_{\text{cat}}", color=BLACK).next_to(v_cat_mat3, LEFT), v_cat_mat3)

            v_sat_mat = Matrix([["2"], ["0"], ["1"]]).set_color(BLACK)
            v_sat_g = VGroup(MathTex(r"v_{\text{sat}}", color=BLACK).next_to(v_sat_mat, LEFT), v_sat_mat)

            values_group = VGroup(v_the_g, v_cat_g3, v_sat_g).arrange(RIGHT, buff=2).move_to(ORIGIN)
            self.play(FadeIn(values_group))
            self.wait(3)

        with self.voiceover(text="We now multiply each Value vector by its attention scalar. 0.04 times v_The, plus 0.11 times v_cat, plus 0.85 times v_sat.") as tracker:
            eq_text = MathTex(r"\text{Output} = 0.04 \cdot", r"v_{\text{The}}", r"+ 0.11 \cdot", r"v_{\text{cat}}", r"+ 0.85 \cdot", r"v_{\text{sat}}", color=BLACK).next_to(values_group, DOWN, buff=1)
            self.play(Write(eq_text))
            self.wait(3)

        with self.voiceover(text="We will calculate this row by row. For the first row, we have 0.04 times 1, plus 0.11 times 0, plus 0.85 times 2, which gives us 1.74.") as tracker:
            row_1_calc = MathTex(r"= \begin{bmatrix} 0.04(1) + 0.11(0) + 0.85(2) \\ 0.04(1) + 0.11(1) + 0.85(0) \\ 0.04(0) + 0.11(2) + 0.85(1) \end{bmatrix}", color=BLACK).move_to(DOWN * 2)
            self.play(FadeOut(eq_text), FadeOut(values_group))
            self.play(Write(row_1_calc))
            self.wait(3)

        with self.voiceover(text="Evaluating the other rows similarly, we obtain our final output vector.") as tracker:
            final_out_mat = Matrix([["1.74"], ["0.15"], ["1.07"]]).set_color(GREEN)
            final_out_g = VGroup(MathTex(r"y_{\text{cat}} =", color=BLACK).next_to(final_out_mat, LEFT), final_out_mat).move_to(DOWN * 2 + RIGHT * 4)
            self.play(Transform(row_1_calc, final_out_g))
            self.wait(3)

        with self.voiceover(text="This vector y_cat is the new, context-aware representation for the token 'cat'. It flows forward, through the output weight matrix W_O, and into the Feed Forward Network.") as tracker:
            arrow_out = Arrow(start=row_1_calc.get_bottom(), end=row_1_calc.get_bottom() + DOWN * 1.5, color=BLACK)
            self.play(Write(arrow_out))
            self.wait(3)

        # --- 6. Forward vs Reverse (Backprop) ---
        with self.voiceover(text="We have successfully traced the entire forward pass. However, an AI engineer must understand how this network learns. We will now look at backpropagation.") as tracker:
            self.play(FadeOut(wsum_title), FadeOut(weights_group), FadeOut(row_1_calc), FadeOut(arrow_out))
            self.wait(2)

        with self.voiceover(text="In the forward pass, data flows from left to right. It passes through our weight matrices, calculates attention, and passes through the output matrices W_O and the FFN weights.") as tracker:
            fw_title = Text("Forward Pass vs Backpropagation", color=BLACK).to_edge(UP)
            self.play(Write(fw_title))

            node_x = Text("Input", color=BLUE).move_to(LEFT * 5)
            node_wqkv = Text("W_Q, W_K, W_V", color=PURPLE).move_to(LEFT * 1)
            node_attn = Text("Attention", color=BLACK).move_to(RIGHT * 2)
            node_wo_ffn = Text("W_O & FFN", color=PURPLE).move_to(RIGHT * 5)

            arr1 = Arrow(node_x.get_right(), node_wqkv.get_left(), color=GREEN)
            arr2 = Arrow(node_wqkv.get_right(), node_attn.get_left(), color=GREEN)
            arr3 = Arrow(node_attn.get_right(), node_wo_ffn.get_left(), color=GREEN)

            flow_group = VGroup(node_x, node_wqkv, node_attn, node_wo_ffn, arr1, arr2, arr3)
            self.play(FadeIn(flow_group))
            self.wait(4)

        with self.voiceover(text="During backpropagation, we calculate the error or loss at the end of the network. This error gradient flows backward, moving from right to left.") as tracker:
            bp_arr1 = DashedLine(node_wo_ffn.get_bottom() + DOWN * 0.5, node_attn.get_bottom() + DOWN * 0.5, color=RED).add_tip()
            bp_arr2 = DashedLine(node_attn.get_bottom() + DOWN * 0.5, node_wqkv.get_bottom() + DOWN * 0.5, color=RED).add_tip()
            bp_arr3 = DashedLine(node_wqkv.get_bottom() + DOWN * 0.5, node_x.get_bottom() + DOWN * 0.5, color=RED).add_tip()

            self.play(Write(bp_arr1))
            self.play(Write(bp_arr2))
            self.play(Write(bp_arr3))
            self.wait(3)

        with self.voiceover(text="As the gradient flows through each component, it computes the partial derivatives needed to update the learned weights: the FFN weights, W_O, and finally W_Q, W_K, and W_V.") as tracker:
            rect1 = SurroundingRectangle(node_wo_ffn, color=RED, buff=0.2)
            rect2 = SurroundingRectangle(node_wqkv, color=RED, buff=0.2)
            self.play(Write(rect1))
            self.play(Write(rect2))
            self.wait(3)

        with self.voiceover(text="Let us write out the mathematical formulation for updating the Query weights, W_Q, using gradient descent.") as tracker:
            self.play(FadeOut(rect1), FadeOut(rect2), FadeOut(bp_arr1), FadeOut(bp_arr2), FadeOut(bp_arr3), FadeOut(flow_group))

            grad_eq = MathTex(r"W_Q^{(new)} = W_Q^{(old)} - \eta \frac{\partial L}{\partial W_Q}", color=RED).move_to(UP * 0.5)
            self.play(Write(grad_eq))
            self.wait(3)

        with self.voiceover(text="Deriving the gradient through the Softmax operation requires calculating its Jacobian matrix. Because the output of one Softmax probability depends on all the input scores in the denominator, the derivative is non-trivial.") as tracker:
            jacobian = MathTex(r"\frac{\partial \text{Softmax}(s_i)}{\partial s_j} = \text{Softmax}(s_i)(\delta_{ij} - \text{Softmax}(s_j))", color=RED).move_to(DOWN * 1.5)
            self.play(Write(jacobian))
            self.wait(4)

        with self.voiceover(text="Here, the Kronecker delta is 1 if i equals j, and 0 otherwise. This equation represents how a change in any raw score affects the final attention probabilities, and is central to how Transformers learn effectively.") as tracker:
            self.wait(4)

        # --- 7. Interview Insight ---
        with self.voiceover(text="To conclude our deep dive, let us highlight a crucial architectural detail that is a highly common interview question.") as tracker:
            self.play(FadeOut(fw_title), FadeOut(grad_eq), FadeOut(jacobian))
            self.wait(2)

        with self.voiceover(text="If we use the raw dot products directly inside the Softmax, we run into a major issue. In production models with high embedding dimensions, these dot products can grow extremely large.") as tracker:
            gotcha_box = Rectangle(width=12, height=4.5, color=RED, fill_color=WHITE, fill_opacity=1)
            gotcha_title = Text("Key Interview Insight: Scaled Dot-Product", color=RED).move_to(gotcha_box.get_top() + DOWN * 0.5)

            # Using substrings in MathTex for precise coloring
            math_formula = MathTex(r"\text{Attention}(Q, K, V) = \text{Softmax}\left(", r"\frac{QK^T}{\sqrt{d_k}}", r"\right)V", color=BLACK)

            gotcha_text = VGroup(
                Text("Raw dot products grow large in high dimensions.", color=BLACK, font_size=28),
                Text("Large inputs push Softmax into regions with tiny gradients,", color=BLACK, font_size=28),
                Text("causing the vanishing gradient problem during backpropagation.", color=BLACK, font_size=28),
                math_formula
            ).arrange(DOWN, buff=0.4).next_to(gotcha_title, DOWN, buff=0.5)

            insight_group = VGroup(gotcha_box, gotcha_title, gotcha_text)
            self.play(FadeIn(insight_group))
            self.wait(4)

        with self.voiceover(text="To counter this, we scale the dot products. We divide them by the square root of the key dimension, d_k. This stabilizes the gradients and ensures the network can learn. Always mention the scaling factor!") as tracker:
            scale_part = math_formula[1]
            self.play(scale_part.animate.set_color(RED))
            self.wait(5)

        with self.voiceover(text="This brings our data flow analysis of Attention Scores to an end. Understanding these steps in detail separates practitioners from true AI engineers. See you next time.") as tracker:
            self.play(FadeOut(insight_group))
            self.wait(3)
