from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class EncoderVsDecoderScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # Part 1: Input Setup
        with self.voiceover(text="Welcome to this deep dive session. Today, we are undertaking a comprehensive architectural analysis of the Transformer model, specifically contrasting the Encoder mechanism utilized by BERT with the Decoder mechanism that powers the GPT family of models. We will trace the exact data flow of a single token through the intricate self-attention pathways. This requires a rigorous understanding of linear algebra and matrix transformations.") as tracker:
            title = Text("Encoder vs Decoder", color=BLACK).scale(1.5).to_edge(UP)
            subtitle = Text("A Deep Dive into Self-Attention", color=DARK_GRAY).scale(0.8).next_to(title, DOWN)
            self.play(Write(title), FadeIn(subtitle))
            self.wait(2)

        with self.voiceover(text="To ground our analysis in a practical example, let us consider a fundamental three-token sequence: 'The', 'cat', and 'sat'. In a real-world scenario, these words are first mapped to their corresponding numerical token IDs via a tokenizer, and subsequently projected into a high-dimensional continuous embedding space.") as tracker:
            tokens = VGroup(
                Text("The", color=BLACK),
                Text("cat", color=BLACK),
                Text("sat", color=BLACK)
            ).arrange(RIGHT, buff=2).shift(UP * 2)

            token_boxes = VGroup()
            for token in tokens:
                box = SurroundingRectangle(token, color=BLACK, fill_opacity=0.1)
                token_boxes.add(VGroup(box, token))

            self.play(FadeOut(title), FadeOut(subtitle))
            self.play(FadeIn(token_boxes))
            self.wait(2)

        with self.voiceover(text="Our primary focus for this detailed walkthrough will be the central token, 'cat'. We will observe precisely how its internal representation is mathematically manipulated and enriched by its surrounding context during a single forward pass. I have highlighted our active token in blue to clearly track its progression.") as tracker:
            cat_box = token_boxes[1][0]
            self.play(cat_box.animate.set_color(BLUE).set_fill(BLUE, opacity=0.2))
            self.wait(2)

        with self.voiceover(text="Prior to entering the self-attention mechanism, this token is represented as a dense, initial embedding vector. Let us define this input vector, which we will denote as 'x'. For the sake of visual clarity in this demonstration, we will assume a reduced dimensionality of d_model equals three. Therefore, 'x' is a three-by-one column vector with the hypothetical values one, zero, and negative one.") as tracker:
            embedding_vector = MathTex(r"x = \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix}", color=BLACK).next_to(token_boxes[1], DOWN, buff=1)
            arrow_down = Arrow(token_boxes[1].get_bottom(), embedding_vector.get_top(), color=BLACK)
            self.play(GrowArrow(arrow_down), FadeIn(embedding_vector))
            self.wait(2)

        # Part 2: Linear Projections
        with self.voiceover(text="The foundational step in self-attention involves generating three distinct representations for each token: the Query vector 'q', the Key vector 'k', and the Value vector 'v'. This is achieved through three independent linear transformations. We multiply the input vector 'x' by three separate, learnable weight matrices: W_Q, W_K, and W_V. These weight matrices are the core parameters optimized during the training phase.") as tracker:
            self.play(FadeOut(arrow_down), embedding_vector.animate.shift(LEFT * 5))

            eq_q = MathTex(r"q = W_Q \, x", color=BLACK).shift(UP * 0.5)
            eq_k = MathTex(r"k = W_K \, x", color=BLACK)
            eq_v = MathTex(r"v = W_V \, x", color=BLACK).shift(DOWN * 0.5)

            eqs = VGroup(eq_q, eq_k, eq_v).next_to(embedding_vector, RIGHT, buff=1.5)
            self.play(Write(eqs))
            self.wait(2)

        with self.voiceover(text="Let us execute the matrix multiplication to derive the Query vector 'q'. We arrange the operation visually: the input vector 'x' is positioned on the left, and it acts upon the Query weight matrix W_Q, positioned on the right. Notice that W_Q is a three-by-three matrix. The resulting dot product yields our new Query vector.") as tracker:
            self.play(FadeOut(eqs))

            q_calc = VGroup(
                MathTex(r"x^T", color=BLACK),
                MathTex(r"\begin{bmatrix} 1 & 0 & -1 \end{bmatrix}", color=BLACK),
                MathTex(r"\times", color=BLACK),
                MathTex(r"W_Q", color=PURPLE),
                MathTex(r"\begin{bmatrix} 2 & 0 & 1 \\ -1 & 1 & 0 \\ 0 & 2 & 1 \end{bmatrix}", color=PURPLE),
                MathTex(r"=", color=BLACK),
                MathTex(r"q^T", color=GREEN),
                MathTex(r"\begin{bmatrix} 2 & -2 & 0 \end{bmatrix}", color=GREEN)
            )

            # Layout: x_T * W_Q = q_T
            q_calc[0].shift(LEFT * 4 + UP * 1)
            q_calc[1].next_to(q_calc[0], DOWN)

            q_calc[2].next_to(q_calc[1], RIGHT)

            q_calc[3].next_to(q_calc[2], RIGHT).shift(UP*1.5 + RIGHT*0.5)
            q_calc[4].next_to(q_calc[3], DOWN)

            q_calc[5].next_to(q_calc[4], RIGHT)

            q_calc[6].next_to(q_calc[5], RIGHT).shift(UP*1)
            q_calc[7].next_to(q_calc[6], DOWN)

            self.play(FadeIn(q_calc))
            self.wait(2)

        with self.voiceover(text="We meticulously calculate each element of the resulting vector. For the first element, we compute the dot product of 'x' with the first column of W_Q: one times two, plus zero times negative one, plus negative one times zero, equaling two. The second element is one times zero, plus zero times one, plus negative one times two, equating to negative two. The final element is one times one, plus zero times zero, plus negative one times one, resulting in zero. This meticulously constructed Query vector 'q' represents what the token 'cat' is actively seeking to find in the surrounding context sequence.") as tracker:
            self.wait(4)

        with self.voiceover(text="Simultaneously, identical operations occur using the W_K and W_V matrices to compute the Key vector 'k' and the Value vector 'v' for the token 'cat', as well as for all other tokens in the sequence. The Key vector represents the intrinsic properties the token offers to others, while the Value vector holds the actual semantic content that will be aggregated.") as tracker:
            self.play(FadeOut(q_calc), FadeOut(embedding_vector))
            self.wait(2)

        # Part 3: Scaled Dot-Product Attention & Masking (The Core Difference)
        with self.voiceover(text="We now arrive at the critical juncture that differentiates a bidirectional Encoder like BERT from an autoregressive Decoder like GPT: the calculation of attention scores. We determine how strongly the query of 'cat' aligns with the keys of all available tokens via a scaled dot product.") as tracker:
            attn_title = Text("Scaled Dot-Product Attention", color=BLACK).to_edge(UP)
            self.play(Write(attn_title))
            self.wait(2)

        with self.voiceover(text="In the BERT Encoder architecture, the attention mechanism is fully bidirectional. The token 'cat' possesses the unrestricted ability to 'look' at the entire sequence. It computes interaction scores with the preceding token 'The', itself, and crucially, the subsequent token 'sat'. This holistic view allows BERT to build deeply contextualized representations informed by both past and future linguistic structure.") as tracker:
            # Distinct points for self-attention curved arrow to avoid NaN error
            start_pt = token_boxes[1].get_top() + LEFT * 0.2
            end_pt = token_boxes[1].get_top() + RIGHT * 0.2

            arrows_bert = VGroup(
                Arrow(token_boxes[1].get_left(), token_boxes[0].get_right(), color=GREEN),
                Arrow(token_boxes[1].get_right(), token_boxes[2].get_left(), color=GREEN),
                CurvedArrow(start_pt, end_pt, angle=-TAU/1.5, color=GREEN)
            )
            self.play(FadeIn(arrows_bert))
            self.wait(2)

        with self.voiceover(text="Conversely, in the GPT Decoder architecture, the primary objective is autoregressive generation—predicting the next word. Permitting a token to look ahead at future tokens would constitute a violation of causality, effectively allowing the model to cheat during training. Therefore, we introduce Masked Self-Attention. The token 'cat' is strictly permitted to attend only to itself and preceding tokens.") as tracker:
            self.play(FadeOut(arrows_bert))

            arrows_gpt = VGroup(
                Arrow(token_boxes[1].get_left(), token_boxes[0].get_right(), color=GREEN),
                CurvedArrow(start_pt, end_pt, angle=-TAU/1.5, color=GREEN)
            )

            cross = Cross(token_boxes[2], stroke_color=RED, stroke_width=6)
            mask_box = Rectangle(width=2, height=1.5, color=RED, fill_color=RED, fill_opacity=0.3).move_to(token_boxes[2])

            self.play(FadeIn(arrows_gpt), FadeIn(cross), FadeIn(mask_box))
            self.wait(2)

        with self.voiceover(text="Mathematically, this causal masking is enforced by modifying the raw attention scores before applying the softmax normalization. The raw scores are computed by multiplying the matrix of Queries, Q, by the transpose of the matrix of Keys, K, and scaling by the square root of the dimensionality, d_k.") as tracker:
            eq_attn_base = MathTex(
                r"\text{Scores} = \frac{Q K^T}{\sqrt{d_k}}",
                color=BLACK
            ).shift(DOWN * 1)
            self.play(Write(eq_attn_base))
            self.wait(2)

        with self.voiceover(text="To implement the mask, we construct an upper triangular masking matrix, M. For any position where a token attempts to attend to a future token, the corresponding entry in M is set to negative infinity. When this matrix M is added to the raw scores, the subsequent softmax operation—which involves exponentiating these values—will drive the attention weights for future tokens mathematically to exactly zero.") as tracker:
            self.play(FadeOut(eq_attn_base))

            eq_attn_mask = MathTex(
                r"\text{Attention Weights} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M \right)",
                color=BLACK
            ).shift(DOWN * 1)

            mask_matrix = MathTex(
                r"M = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}",
                color=RED
            ).next_to(eq_attn_mask, DOWN, buff=0.5)

            self.play(Write(eq_attn_mask), FadeIn(mask_matrix))
            self.wait(3)

        with self.voiceover(text="These normalized attention weights, which now strictly adhere to causality, are then multiplied by the matrix of Value vectors, V. This aggregation process synthesizes a refined, context-aware representation for the token 'cat'. This newly updated vector encapsulates its localized meaning, ready to be passed forward through the subsequent Feed-Forward Network and residual connections.") as tracker:
            self.play(FadeOut(eq_attn_mask), FadeOut(mask_matrix), FadeOut(attn_title), FadeOut(arrows_gpt), FadeOut(cross), FadeOut(mask_box))

            updated_vector = MathTex(r"z = \begin{bmatrix} 0.5 \\ -0.2 \\ 0.8 \end{bmatrix}", color=GREEN).next_to(token_boxes[1], DOWN, buff=1)
            arrow_down2 = Arrow(token_boxes[1].get_bottom(), updated_vector.get_top(), color=GREEN)
            self.play(GrowArrow(arrow_down2), FadeIn(updated_vector))
            self.wait(2)

        # Part 4: Backpropagation & Gradients
        with self.voiceover(text="Having established the forward pass, we must now examine the learning process. During backpropagation, a scalar loss value, L, is computed at the final output layer, quantifying the model's prediction error. The objective is to distribute this error backward to update the learnable parameters. We will visualize this gradient flow using dashed red vectors.") as tracker:
            self.play(FadeOut(arrow_down2), FadeOut(updated_vector))

            loss_node = MathTex("L", color=RED).shift(DOWN * 3)
            grad_arrow1 = DashedLine(loss_node.get_top(), token_boxes[1].get_bottom(), color=RED).add_tip()

            self.play(FadeIn(loss_node), GrowArrow(grad_arrow1))
            self.wait(2)

        with self.voiceover(text="To update the initial weight matrices—W_Q, W_K, and W_V—the gradient must systematically traverse the chain rule of calculus through every operation of the self-attention mechanism. This includes propagating through the final output projection matrix W_O, the attention aggregation step, the complex Jacobian of the softmax function, and finally the scaled dot-product itself.") as tracker:
            self.wait(2)

        with self.voiceover(text="Let us derive the explicit gradient for the Query weight matrix, W_Q. The total derivative of the Loss with respect to W_Q is an expansion of the chain rule. It equals the partial derivative of the Loss with respect to the output 'z', multiplied by the partial derivative of 'z' with respect to the attention scores, multiplied by the derivative of the scores with respect to the query vector 'q', and ultimately multiplied by the derivative of 'q' with respect to W_Q.") as tracker:
            self.play(FadeOut(grad_arrow1), FadeOut(loss_node), token_boxes.animate.shift(UP * 1.5))

            chain_rule = MathTex(
                r"\frac{\partial L}{\partial W_Q} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial S} \cdot \frac{\partial S}{\partial q} \cdot \frac{\partial q}{\partial W_Q}",
                color=RED
            ).shift(UP * 1)
            self.play(Write(chain_rule))
            self.wait(3)

        with self.voiceover(text="This mathematical formulation reveals the intricate dependencies. The error signal for W_Q is heavily influenced by the Key vectors of all attending tokens, as the derivative of the scores with respect to 'q' is fundamentally the Key matrix, K. Furthermore, the gradient must navigate the softmax derivative, which often leads to the infamous vanishing gradient problem if the raw attention scores exhibit extreme magnitudes, saturating the softmax function.") as tracker:
            derivation_1 = MathTex(
                r"\text{Where } \frac{\partial S}{\partial q} \propto K^T \text{ and } \frac{\partial q}{\partial W_Q} = x^T",
                color=BLACK
            ).next_to(chain_rule, DOWN, buff=0.7)

            derivation_2 = MathTex(
                r"\therefore \frac{\partial L}{\partial W_Q} = \left( \frac{\partial L}{\partial q} \right) x^T",
                color=RED
            ).next_to(derivation_1, DOWN, buff=0.7)

            self.play(Write(derivation_1))
            self.wait(2)
            self.play(Write(derivation_2))
            self.wait(3)

        # Part 5: Interview Callout
        with self.voiceover(text="We conclude this comprehensive technical deep dive by emphasizing the most critical concept to articulate during an advanced AI engineering interview regarding these architectures.") as tracker:
            self.play(FadeOut(chain_rule), FadeOut(derivation_1), FadeOut(derivation_2), FadeOut(token_boxes))
            self.wait(1)

        with self.voiceover(text="The superficial answer distinguishing BERT and GPT is simply stating that 'BERT is an Encoder and GPT is a Decoder'. However, the rigorous, distinguishing technical answer centers on the causal mask. You must explicitly define how the masking matrix forces future attention weights to zero prior to the softmax operation, thereby enforcing the autoregressive property critical for generative tasks.") as tracker:
            callout_box = Rectangle(width=12, height=4.5, color=BLACK, fill_color=WHITE, fill_opacity=1)
            gotcha_title = Text("Critical Interview Insight", color=RED).next_to(callout_box.get_top(), DOWN, buff=0.3)
            gotcha_text = Text(
                "Do not merely state 'GPT is a Decoder, BERT is an Encoder'.\n\n"
                "Explain the mechanistic change: The inclusion of a Causal Mask matrix (M)\n"
                "prior to the Softmax layer in self-attention, forcing future attention\n"
                "weights mathematically to exactly zero, enabling autoregression.",
                color=BLACK, font_size=24, line_spacing=1.2
            ).next_to(gotcha_title, DOWN, buff=0.5)

            self.play(FadeIn(callout_box), Write(gotcha_title), Write(gotcha_text))
            self.wait(5)

        with self.voiceover(text="Mastery of these low-level architectural nuances—understanding not just what the models do, but precisely how the underlying linear algebra and calculus execute it—is the hallmark of a senior AI engineer. Thank you for your rigorous attention during this session.") as tracker:
            self.wait(4)
            self.play(FadeOut(callout_box), FadeOut(gotcha_title), FadeOut(gotcha_text))
