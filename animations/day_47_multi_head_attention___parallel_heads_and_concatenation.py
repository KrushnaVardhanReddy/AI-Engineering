from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class MultiHeadAttentionScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # --- Prologue ---
        with self.voiceover(text="""Welcome to another deep dive in our AI Engineering series.
        Today, we are exploring one of the most critical and powerful components of the Transformer architecture: Multi-Head Attention.
        Specifically, we will look at how multiple attention heads operate in parallel and how their results are combined through concatenation and linear projection.
        This is a deep, step-by-step mathematical tracing of how data flows through this stage, which is a common topic in advanced machine learning interviews.
        So grab your notebook, and let's get started.
        Before we compute anything, let's understand why we need multiple heads in the first place.""") as tracker:
            title = Text("Multi-Head Attention", color=BLACK).scale(1.2).to_edge(UP)
            self.play(Write(title))
            subtitle = Text("Parallel Heads and Concatenation", color=BLACK).scale(0.8).next_to(title, DOWN)
            self.play(Write(subtitle))
            self.wait(1)

        with self.voiceover(text="""In standard single-head self-attention, the model learns to focus on specific parts of the input sequence.
        However, natural language is complex. A single word might have multiple relationships with other words in a sentence.
        For example, it might relate to the subject, the action, and the object all at once.
        Multi-Head Attention solves this by running multiple self-attention mechanisms in parallel.
        Each 'head' learns to focus on different aspects or representations of the data, allowing the model to capture richer, more complex relationships.
        Let's visualize this using a concrete example.""") as tracker:
            self.play(FadeOut(subtitle))
            self.wait(1)

        # --- Input Setup ---
        with self.voiceover(text="""We start with a simple sentence containing three tokens: 'The', 'cat', and 'sat'.
        Imagine these tokens have already been processed by an embedding layer and perhaps previous transformer blocks, so they are represented as continuous dense vectors.
        For this walkthrough, we will trace the data flow from the perspective of the active token, which is 'cat'.""") as tracker:
            tokens_text = ["The", "cat", "sat"]
            token_boxes = VGroup()
            for t in tokens_text:
                box = Rectangle(width=2, height=1, color=BLACK)
                text = Text(t, color=BLACK).scale(0.8)
                text.move_to(box.get_center())
                token_box = VGroup(box, text)
                token_boxes.add(token_box)

            token_boxes.arrange(RIGHT, buff=0.5).move_to(LEFT * 3 + UP * 1.5)
            self.play(FadeIn(token_boxes))

            # Highlight 'cat'
            active_idx = 1
            cat_box = token_boxes[active_idx][0]
            cat_text = token_boxes[active_idx][1]
            self.play(
                cat_box.animate.set_fill(BLUE, opacity=0.3),
                cat_text.animate.set_color(BLUE)
            )
            self.wait(2)

        with self.voiceover(text="""Let's represent the input token 'cat' as a vector X.
        To keep the math tractable for this visualization, let's say X is a two-dimensional vector: [1, 2].
        In reality, this would be the hidden dimension of the transformer, often denoted as d_model, which could be 512, 768, or even larger.""") as tracker:
            x_label = MathTex("X_{\\text{cat}} =", "\\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}", color=BLACK).scale(0.8)
            x_label.next_to(token_boxes[active_idx], DOWN, buff=0.5)
            self.play(Write(x_label))
            self.wait(2)

        # --- Head 1 Computation ---
        with self.voiceover(text="""Now we enter the Multi-Head Attention mechanism. Let's assume we have two attention heads: Head 1 and Head 2.
        First, we will look at Head 1.
        In Head 1, the input vector X is multiplied by three separate learned weight matrices to produce three new vectors: Query (Q), Key (K), and Value (V).
        These weight matrices are denoted as W_Q, W_K, and W_V.
        Let's explicitly perform the matrix multiplication for the Query vector in Head 1.""") as tracker:
            head1_group = VGroup()
            head1_title = Text("Head 1", color=BLACK).scale(0.7).to_edge(LEFT).shift(UP * 0.5)
            self.play(Write(head1_title))
            self.wait(1)

        with self.voiceover(text="""The Query weight matrix for Head 1, let's call it W_Q1, projects the input X into the query space.
        Let's say W_Q1 is a 2x2 matrix: [[1, 0], [0, 1]].
        We multiply W_Q1 by X to get the Query vector Q1.""") as tracker:
            wq1_eq = MathTex(
                "Q_1 =", "W_{Q1}", "X", "=",
                "\\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \\end{bmatrix}",
                "\\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}", color=BLACK
            ).scale(0.7).next_to(head1_title, DOWN, aligned_edge=LEFT)
            wq1_eq[1].set_color(PURPLE) # W_Q1

            self.play(Write(wq1_eq))
            self.wait(2)

        with self.voiceover(text="""Let's calculate the result cell by cell.
        For the first element: 1 times 1, plus 0 times 2, equals 1.
        For the second element: 0 times 1, plus 1 times 2, equals 2.
        So, our Query vector Q1 is [1, 2].""") as tracker:
            q1_res = MathTex("=", "\\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}", color=BLACK).scale(0.7).next_to(wq1_eq, RIGHT)
            q1_res[1].set_color(GREEN)
            self.play(Write(q1_res))
            self.wait(2)

        with self.voiceover(text="""Similarly, we use separate weight matrices W_K1 and W_V1 to compute the Key (K1) and Value (V1) vectors for Head 1.
        For the sake of time, we will abstract the calculation for K1 and V1 and assume the self-attention process, which involves scaled dot products and softmax against all other tokens, has been computed.
        This process outputs an attention-weighted sum of the Value vectors.""") as tracker:
            kv1_text = Text("Compute K1, V1 -> Scaled Dot-Product Attention", color=BLACK).scale(0.6).next_to(wq1_eq, DOWN, aligned_edge=LEFT)
            self.play(Write(kv1_text))
            self.wait(2)

        with self.voiceover(text="""Let's say the final output from Head 1's self-attention, which we will call Z1, is [2, 3].
        This vector Z1 represents the contextualized information gathered by Head 1 focusing on specific relationships.""") as tracker:
            z1_eq = MathTex("Z_1 =", "\\begin{bmatrix} 2 \\\\ 3 \\end{bmatrix}", color=BLACK).scale(0.7).next_to(kv1_text, DOWN, aligned_edge=LEFT)
            z1_eq[1].set_color(GREEN)
            self.play(Write(z1_eq))
            self.wait(2)

        # --- Head 2 Computation ---
        with self.voiceover(text="""Now, what makes this 'Multi-Head' is that entirely in parallel, Head 2 is doing the exact same thing, but with its own set of independent, learned weight matrices.""") as tracker:
            head2_title = Text("Head 2", color=BLACK).scale(0.7).to_edge(RIGHT).shift(UP * 0.5 + LEFT * 2)
            self.play(Write(head2_title))
            self.wait(1)

        with self.voiceover(text="""Head 2 uses W_Q2, W_K2, and W_V2.
        These weights start with different random initializations and learn different patterns during training.
        Let's say W_Q2 is [[0, 1], [1, 0]].
        Notice how it is distinct from W_Q1.""") as tracker:
            wq2_eq = MathTex(
                "Q_2 =", "W_{Q2}", "X", "=",
                "\\begin{bmatrix} 0 & 1 \\\\ 1 & 0 \\end{bmatrix}",
                "\\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}", color=BLACK
            ).scale(0.7).next_to(head2_title, DOWN, aligned_edge=LEFT)
            wq2_eq[1].set_color(PURPLE) # W_Q2
            self.play(Write(wq2_eq))
            self.wait(2)

        with self.voiceover(text="""Let's do the math again.
        First element: 0 times 1, plus 1 times 2, equals 2.
        Second element: 1 times 1, plus 0 times 2, equals 1.
        Our Query vector Q2 is [2, 1].""") as tracker:
            q2_res = MathTex("=", "\\begin{bmatrix} 2 \\\\ 1 \\end{bmatrix}", color=BLACK).scale(0.7).next_to(wq2_eq, RIGHT)
            q2_res[1].set_color(GREEN)
            self.play(Write(q2_res))
            self.wait(2)

        with self.voiceover(text="""Just like Head 1, Head 2 computes its Keys, Values, and applies scaled dot-product attention independently.""") as tracker:
            kv2_text = Text("Compute K2, V2 -> Scaled Dot-Product Attention", color=BLACK).scale(0.6).next_to(wq2_eq, DOWN, aligned_edge=LEFT)
            self.play(Write(kv2_text))
            self.wait(2)

        with self.voiceover(text="""Let's assume the final context-aware output from Head 2, denoted as Z2, is [4, 5].
        Head 2 has learned different relationships and thus produced a different representation than Head 1.""") as tracker:
            z2_eq = MathTex("Z_2 =", "\\begin{bmatrix} 4 \\\\ 5 \\end{bmatrix}", color=BLACK).scale(0.7).next_to(kv2_text, DOWN, aligned_edge=LEFT)
            z2_eq[1].set_color(GREEN)
            self.play(Write(z2_eq))
            self.wait(2)

        # --- Concatenation ---
        with self.voiceover(text="""We now have multiple context-aware representations of the token 'cat', one from each head.
        Head 1 gave us Z1, and Head 2 gave us Z2.
        The next step in the data flow is to combine these parallel representations.
        We do this by simply concatenating the output vectors from all heads.""") as tracker:
            self.play(
                FadeOut(wq1_eq), FadeOut(q1_res), FadeOut(kv1_text),
                FadeOut(wq2_eq), FadeOut(q2_res), FadeOut(kv2_text),
                FadeOut(head1_title), FadeOut(head2_title)
            )

            z1_copy = z1_eq.copy().move_to(LEFT * 3 + DOWN * 1)
            z2_copy = z2_eq.copy().move_to(RIGHT * 3 + DOWN * 1)

            self.play(
                Transform(z1_eq, z1_copy),
                Transform(z2_eq, z2_copy)
            )
            self.wait(2)

        with self.voiceover(text="""Let's define the concatenated vector as Z_concat.
        It is formed by stacking Z1 and Z2 together.
        Since Z1 is [2, 3] and Z2 is [4, 5], our Z_concat will be a 4-dimensional vector: [2, 3, 4, 5].""") as tracker:
            concat_eq = MathTex("Z_{\\text{concat}} = [Z_1, Z_2] =", "\\begin{bmatrix} 2 \\\\ 3 \\\\ 4 \\\\ 5 \\end{bmatrix}", color=BLACK).scale(0.8)
            concat_eq.move_to(DOWN * 1)
            concat_eq[1].set_color(GREEN)

            # Arrows pointing to concat
            arrow1 = Arrow(start=z1_eq.get_right(), end=concat_eq.get_left(), color=BLACK, buff=0.1)
            arrow2 = Arrow(start=z2_eq.get_left(), end=concat_eq.get_right(), color=BLACK, buff=0.1)

            self.play(GrowArrow(arrow1), GrowArrow(arrow2))
            self.play(Write(concat_eq))
            self.wait(2)

        # --- Linear Projection ---
        with self.voiceover(text="""Notice that our concatenated vector Z_concat is now 4-dimensional, but our original input X was 2-dimensional.
        The transformer architecture typically requires the input and output of the Multi-Head Attention sub-layer to have the exact same dimension, d_model.
        This allows for residual connections and consistent layer stacking.
        To project this 4-dimensional vector back down to the required 2 dimensions, we apply a final linear transformation.
        This is done using another learned weight matrix, often called the output weight matrix, W_O.""") as tracker:
            self.play(
                FadeOut(z1_eq), FadeOut(z2_eq), FadeOut(arrow1), FadeOut(arrow2), FadeOut(x_label)
            )
            self.play(concat_eq.animate.move_to(LEFT * 3 + DOWN * 2))
            self.wait(2)

        with self.voiceover(text="""Since we need to project from dimension 4 down to dimension 2, the weight matrix W_O must have a shape of 2 by 4.
        Let's say W_O is initialized as this matrix: [[1, 0, 1, 0], [0, 1, 0, 1]].
        We compute the final output of the Multi-Head Attention block, which we'll call Z_final, by multiplying W_O and Z_concat.""") as tracker:
            wo_eq = MathTex(
                "Z_{\\text{final}} =", "W_O", "Z_{\\text{concat}}", "=",
                "\\begin{bmatrix} 1 & 0 & 1 & 0 \\\\ 0 & 1 & 0 & 1 \\end{bmatrix}",
                "\\begin{bmatrix} 2 \\\\ 3 \\\\ 4 \\\\ 5 \\end{bmatrix}", color=BLACK
            ).scale(0.7).next_to(concat_eq, RIGHT, buff=0.5)
            wo_eq[1].set_color(PURPLE) # W_O
            self.play(Write(wo_eq))
            self.wait(2)

        with self.voiceover(text="""Let's calculate this final matrix multiplication.
        First row of W_O times Z_concat: 1*2 + 0*3 + 1*4 + 0*5. That's 2 plus 4, which is 6.
        Second row of W_O times Z_concat: 0*2 + 1*3 + 0*4 + 1*5. That's 3 plus 5, which is 8.
        Our final output vector Z_final is [6, 8].""") as tracker:
            final_res = MathTex("=", "\\begin{bmatrix} 6 \\\\ 8 \\end{bmatrix}", color=BLACK).scale(0.7).next_to(wo_eq, RIGHT)
            final_res[1].set_color(GREEN)
            self.play(Write(final_res))
            self.wait(2)

        with self.voiceover(text="""And there we have it.
        The input token 'cat' entered as a 2D vector, was split and processed independently across multiple attention heads to capture diverse linguistic nuances, the results were concatenated, and then projected back into a 2D vector.
        This final output vector is now heavily contextualized by the parallel attention heads and is ready to be passed to the next component in the Transformer, typically a residual connection followed by Layer Normalization.""") as tracker:
            # Arrow out
            out_arrow = Arrow(start=final_res.get_bottom(), end=final_res.get_bottom() + DOWN * 1, color=GREEN)
            out_text = Text("To Add & Norm", color=BLACK).scale(0.5).next_to(out_arrow, DOWN)
            self.play(GrowArrow(out_arrow), Write(out_text))
            self.wait(3)

        # --- Backpropagation ---
        with self.voiceover(text="""What we have traced so far is the Forward Pass.
        Information flows forward, and calculations yield an output.
        But how does the model actually learn these crucial weight matrices: W_Q, W_K, W_V, and W_O?
        It learns through backpropagation.""") as tracker:
            self.play(
                FadeOut(concat_eq), FadeOut(wo_eq), FadeOut(final_res),
                FadeOut(out_arrow), FadeOut(out_text), FadeOut(token_boxes), FadeOut(title)
            )
            backprop_title = Text("Backpropagation in Multi-Head Attention", color=BLACK).scale(1.2).to_edge(UP)
            self.play(Write(backprop_title))
            self.wait(2)

        with self.voiceover(text="""During training, the final prediction of the network is compared against the true target, and a loss function calculates the error.
        This error, or gradient, must then flow backward through the network to update the weights.
        Let's visualize this reverse data flow.
        In the forward pass, data flows like this.""") as tracker:
            # Simplified Forward diagram
            box_concat = Rectangle(width=2, height=1, color=BLACK)
            text_concat = Text("Concat", color=BLACK).scale(0.6).move_to(box_concat)
            group_concat = VGroup(box_concat, text_concat).shift(UP * 0.5)

            box_wo = Rectangle(width=2, height=1, color=PURPLE)
            text_wo = Text("Linear (W_O)", color=BLACK).scale(0.6).move_to(box_wo)
            group_wo = VGroup(box_wo, text_wo).next_to(group_concat, DOWN, buff=1)

            box_loss = Rectangle(width=2, height=1, color=BLACK)
            text_loss = Text("Loss / Output", color=BLACK).scale(0.6).move_to(box_loss)
            group_loss = VGroup(box_loss, text_loss).next_to(group_wo, DOWN, buff=1)

            fwd_arrow1 = Arrow(start=group_concat.get_bottom(), end=group_wo.get_top(), color=GREEN)
            fwd_arrow2 = Arrow(start=group_wo.get_bottom(), end=group_loss.get_top(), color=GREEN)

            self.play(FadeIn(group_concat), FadeIn(group_wo), FadeIn(group_loss))
            self.play(GrowArrow(fwd_arrow1), GrowArrow(fwd_arrow2))
            self.wait(2)

        with self.voiceover(text="""Now, during backpropagation, the gradient of the loss with respect to the output, denoted as dL/dZ_final, flows backward.
        Let's draw this backward flow with dashed red arrows.""") as tracker:
            bwd_arrow1 = DashedLine(start=group_loss.get_top() + RIGHT * 0.5, end=group_wo.get_bottom() + RIGHT * 0.5, color=RED).add_tip()
            bwd_arrow2 = DashedLine(start=group_wo.get_top() + RIGHT * 0.5, end=group_concat.get_bottom() + RIGHT * 0.5, color=RED).add_tip()

            grad_text1 = MathTex("\\frac{\\partial L}{\\partial Z_{\\text{final}}}", color=RED).scale(0.6).next_to(bwd_arrow1, RIGHT)

            self.play(Create(bwd_arrow1), Write(grad_text1))
            self.wait(2)

        with self.voiceover(text="""When the gradient reaches the final linear projection layer, it is used to calculate two crucial things.
        First, it calculates how to update the W_O weight matrix.
        Using the chain rule of calculus, the gradient for W_O is the product of the incoming gradient and the forward pass input to that layer, which was Z_concat.
        The weights are updated, taking a small step in the opposite direction of the gradient to minimize the loss.""") as tracker:
            grad_wo = MathTex("\\frac{\\partial L}{\\partial W_O} = \\frac{\\partial L}{\\partial Z_{\\text{final}}} \\cdot Z_{\\text{concat}}^T", color=RED).scale(0.8)
            grad_wo.next_to(group_wo, LEFT, buff=0.5)

            self.play(Write(grad_wo))
            self.play(box_wo.animate.set_fill(PURPLE, opacity=0.3)) # Indicate update
            self.wait(2)

        with self.voiceover(text="""Second, it calculates the gradient to pass backward to the previous layer.
        This gradient is the product of the incoming gradient and the transpose of the W_O weight matrix.
        This new gradient, dL/dZ_concat, then flows back up through the concatenation operation.""") as tracker:
            grad_text2 = MathTex("\\frac{\\partial L}{\\partial Z_{\\text{concat}}}", color=RED).scale(0.6).next_to(bwd_arrow2, RIGHT)
            self.play(Create(bwd_arrow2), Write(grad_text2))
            self.wait(2)

        with self.voiceover(text="""Because concatenation is just a stacking operation, passing the gradient backward is simple.
        We just split the incoming gradient vector back into pieces and route each piece to the corresponding attention head.
        From there, the gradients flow backward through the scaled dot-product attention, ultimately arriving to update the W_Q, W_K, and W_V weight matrices in each individual head.
        This intricate dance of forward computation and backward error correction is what allows the multiple heads to learn complex, complementary features without explicit manual programming.""") as tracker:
            grad_concat_split = MathTex("\\to \\text{Split gradient for Head 1 and Head 2}", color=RED).scale(0.6).next_to(group_concat, LEFT, buff=0.5)
            self.play(Write(grad_concat_split))
            self.wait(4)

        # --- Key Interview Insight ---
        with self.voiceover(text="""To wrap up, let's focus on a Key Interview Insight.
        Interviewers love to test if you truly understand the dimension mathematics of Multi-Head Attention.""") as tracker:
            self.play(
                FadeOut(group_concat), FadeOut(group_wo), FadeOut(group_loss),
                FadeOut(fwd_arrow1), FadeOut(fwd_arrow2), FadeOut(bwd_arrow1), FadeOut(bwd_arrow2),
                FadeOut(grad_text1), FadeOut(grad_wo), FadeOut(grad_text2), FadeOut(grad_concat_split), FadeOut(backprop_title)
            )
            insight_title = Text("Key Interview Insight", color=BLACK).scale(1.2).to_edge(UP)
            self.play(Write(insight_title))
            self.wait(1)

        with self.voiceover(text="""Here is the gotcha: A common misconception is that Multi-Head Attention requires significantly more computation than Single-Head Attention.
        This is typically false in standard architectures.
        Why? Because as we increase the number of heads, we simultaneously decrease the dimension of each head.
        Specifically, the dimension of each head is usually d_model divided by the number of heads.
        For example, if d_model is 512 and we have 8 heads, each head operates on a dimension of 64.""") as tracker:
            gotcha_box = Rectangle(width=10, height=4, color=RED).shift(UP * 0.5)
            insight_text1 = Text("Gotcha: More heads = More computation?", color=BLACK).scale(0.7).move_to(gotcha_box.get_top() + DOWN * 0.5)
            insight_text2 = Text("Truth: Usually FALSE in standard Transformers.", color=GREEN).scale(0.7).next_to(insight_text1, DOWN, buff=0.5)

            math_text = MathTex("d_{\\text{head}} = \\frac{d_{\\text{model}}}{h}", color=BLACK).scale(1.2).next_to(insight_text2, DOWN, buff=0.5)

            self.play(Create(gotcha_box), Write(insight_text1))
            self.wait(1)
            self.play(Write(insight_text2))
            self.wait(1)
            self.play(Write(math_text))
            self.wait(2)

        with self.voiceover(text="""Therefore, the total computational cost of the matrix multiplications for Q, K, and V across all parallel heads is roughly equal to the cost of a single-head attention mechanism operating on the full dimension.
        The multi-head approach gives us the benefit of learning diverse representations essentially 'for free' in terms of computational complexity, though it does introduce some memory overhead.
        Remember this dimension splitting trick, as it often comes up when discussing model optimization and scaling.""") as tracker:
            insight_text3 = Text("Total computation is roughly equivalent to single-head.", color=BLACK).scale(0.6).next_to(math_text, DOWN, buff=0.5)
            self.play(Write(insight_text3))
            self.wait(4)

        with self.voiceover(text="""Thank you for joining this deep dive into Multi-Head Attention data flow.
        Mastering these internals is a significant step towards becoming a senior AI engineer.
        See you in the next session.""") as tracker:
            self.play(FadeOut(gotcha_box), FadeOut(insight_text1), FadeOut(insight_text2), FadeOut(math_text), FadeOut(insight_text3), FadeOut(insight_title))
            self.wait(2)
