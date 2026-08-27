from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np

class DecoderStackAnimation(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # 1. Introduction and setup
        with self.voiceover(text="Welcome to Day 51 of our AI and Machine Learning interview prep series. Today, we are taking a deep, technical dive into the Decoder Stack of a Transformer model, specifically focusing on the data flow through Masked Self-Attention and Cross-Attention. Let's trace how a token travels through this stage step by step. We will use the simple example sentence: 'The cat sat'. We start with three input tokens.") as tracker:
            tokens_text = ["The", "cat", "sat"]
            token_boxes = VGroup()

            for i, text in enumerate(tokens_text):
                box = Rectangle(width=2.5, height=1.5, color=BLACK).set_fill(WHITE, opacity=1)
                label = Text(text, color=BLACK).move_to(box.get_center())
                idx_label = MathTex(f"i={i}", color=BLACK).scale(0.6).next_to(box, DOWN)
                token_group = VGroup(box, label, idx_label)
                token_boxes.add(token_group)

            token_boxes.arrange(RIGHT, buff=0.5).to_edge(UP, buff=1)
            self.play(FadeIn(token_boxes))

            # Highlight the active token "cat"
            self.play(token_boxes[1][0].animate.set_fill(BLUE, opacity=0.3))

        self.wait(1)

        with self.voiceover(text="In this detailed examination, we will focus entirely on the active token 'cat', which is at index 1. We will observe exactly what mathematical operations are applied to it as it passes through the decoder block. The decoder block consists of two main attention mechanisms: Masked Self-Attention, followed by Cross-Attention with the encoder's output. Let's begin with Masked Self-Attention.") as tracker:
            self.play(token_boxes[0].animate.set_opacity(0.3), token_boxes[2].animate.set_opacity(0.3))

            title_masked = Text("Masked Self-Attention", color=BLACK).scale(0.8).to_edge(LEFT, buff=1).shift(UP * 1)
            self.play(Write(title_masked))
            self.wait(1)

        # 2. Masked Self-Attention
        with self.voiceover(text="In Masked Self-Attention, the fundamental rule is that a token can only attend to itself and tokens that precede it. It cannot look ahead into the future. This ensures the autoregressive property of the decoder during text generation. For our token 'cat' at index 1, it can compute attention scores with 'The' at index 0, and itself at index 1. However, the connection to 'sat' at index 2 will be completely masked out.") as tracker:
            self.wait(2)

        with self.voiceover(text="First, the embedding for 'cat' is multiplied by three learned weight matrices: W_Q, W_K, and W_V. This produces three distinct vectors: the Query vector q, the Key vector k, and the Value vector v. Let's animate this process with some simplified 2-dimensional representations to make the math concrete.") as tracker:
            # Show vectors for 'cat'
            math_qkv = MathTex(
                r"x_{cat} \cdot W_Q = q_{cat} = [1.2, 0.5]",
                r"x_{cat} \cdot W_K = k_{cat} = [0.8, -0.2]",
                r"x_{cat} \cdot W_V = v_{cat} = [2.0, 1.1]"
            , color=BLACK).arrange(DOWN, buff=0.5).scale(0.8).next_to(title_masked, DOWN, buff=1)

            math_qkv[0][0:2].set_color(BLUE)
            math_qkv[0][3:5].set_color(PURPLE)
            math_qkv[0][6:10].set_color(GREEN)

            math_qkv[1][0:2].set_color(BLUE)
            math_qkv[1][3:5].set_color(PURPLE)
            math_qkv[1][6:10].set_color(GREEN)

            math_qkv[2][0:2].set_color(BLUE)
            math_qkv[2][3:5].set_color(PURPLE)
            math_qkv[2][6:10].set_color(GREEN)

            self.play(Write(math_qkv))
            self.wait(2)

        with self.voiceover(text="Now, we calculate the raw attention scores by taking the dot product of the query vector for 'cat' with the key vectors of all tokens up to the current position. We also compute the dot product with future tokens just to show how the mask will eliminate them shortly.") as tracker:
            self.play(FadeOut(math_qkv))

            # Show dot products
            dot_prod_math = MathTex(
                r"score_{cat \to The} = q_{cat} \cdot k_{The} = 1.8",
                r"score_{cat \to cat} = q_{cat} \cdot k_{cat} = 0.86",
                r"score_{cat \to sat} = q_{cat} \cdot k_{sat} = -0.5"
            , color=BLACK).arrange(DOWN, buff=0.5).scale(0.8).next_to(title_masked, DOWN, buff=1)

            self.play(Write(dot_prod_math))
            self.wait(2)

        with self.voiceover(text="Here comes the crucial masking step. Because 'sat' is a future token relative to 'cat', we apply a mask. The mask sets the attention score for any future token to negative infinity. This is mathematically necessary so that the subsequent softmax operation will squash this score precisely to zero.") as tracker:
            mask_box = SurroundingRectangle(dot_prod_math[2], color=RED, buff=0.2)
            self.play(Write(mask_box))

            masked_score = MathTex(r"score_{cat \to sat} = -\infty", color=RED).scale(0.8).move_to(dot_prod_math[2].get_center())
            self.play(Transform(dot_prod_math[2], masked_score))
            self.wait(2)

        with self.voiceover(text="Next, we apply the softmax function over these masked scores. The softmax operation exponentiates the scores and normalizes them so they sum to one. As expected, e to the power of negative infinity is zero, meaning 'cat' pays absolutely zero attention to the future token 'sat'.") as tracker:
            self.play(FadeOut(mask_box))

            softmax_math = MathTex(
                r"\text{Softmax}([1.8, 0.86, -\infty])",
                r"= \left[ \frac{e^{1.8}}{Z}, \frac{e^{0.86}}{Z}, \frac{e^{-\infty}}{Z} \right]",
                r"= [0.72, 0.28, 0.0]"
            , color=BLACK).arrange(DOWN, buff=0.3).scale(0.8).next_to(title_masked, DOWN, buff=1)

            self.play(FadeOut(dot_prod_math), Write(softmax_math))
            self.wait(2)

        with self.voiceover(text="Finally, we compute the output of the masked self-attention layer for 'cat' by taking a weighted sum of the Value vectors, using our newly computed softmax probabilities as the weights. Notice how the value vector for 'sat' is multiplied by zero, effectively removing it from the calculation.") as tracker:
            self.play(FadeOut(softmax_math))

            weighted_sum = MathTex(
                r"z_{cat} = 0.72 \cdot v_{The} + 0.28 \cdot v_{cat} + 0.0 \cdot v_{sat}",
                r"z_{cat} = [1.8, 0.7] + [0.56, 0.3] + [0, 0]",
                r"z_{cat} = [2.36, 1.0]"
            , color=BLACK).arrange(DOWN, buff=0.4).scale(0.8).next_to(title_masked, DOWN, buff=1)

            self.play(Write(weighted_sum))
            self.wait(2)

            # Highlight output
            out_box = SurroundingRectangle(weighted_sum[2], color=GREEN, buff=0.1)
            self.play(Write(out_box))
            self.wait(1)

        # 3. Cross-Attention
        with self.voiceover(text="The vector z_cat, which is the output of the masked self-attention mechanism, now flows upwards into the Cross-Attention layer. Let's transition to the next phase.") as tracker:
            self.play(FadeOut(title_masked), FadeOut(weighted_sum), FadeOut(out_box))

            title_cross = Text("Cross-Attention", color=BLACK).scale(0.8).to_edge(LEFT, buff=1).shift(UP * 1)
            self.play(Write(title_cross))

            arrow_up = Arrow(start=DOWN*2, end=UP*0.5, color=GREEN).next_to(title_cross, DOWN, buff=1).shift(LEFT*2)
            label_z = MathTex(r"z_{cat}", color=GREEN).next_to(arrow_up, DOWN)
            self.play(GrowArrow(arrow_up), Write(label_z))
            self.wait(1)

        with self.voiceover(text="In Cross-Attention, the architecture blends information from both the decoder and the encoder. The Query vector, Q, is derived from the previous layer in the decoder. In our case, that is z_cat. However, the Key and Value vectors, K and V, come directly from the final output of the Encoder stack. This is how the model incorporates the context of the original input sequence.") as tracker:
            q_eq = MathTex(r"q_{cross} = z_{cat} \cdot W_{Q}^{cross}", color=BLUE).scale(0.8).next_to(arrow_up, RIGHT, buff=1)
            k_eq = MathTex(r"K_{enc}, V_{enc} \text{ from Encoder}", color=PURPLE).scale(0.8).next_to(q_eq, DOWN, buff=0.5)

            self.play(Write(q_eq), Write(k_eq))
            self.wait(2)

        with self.voiceover(text="The mathematical steps here mirror self-attention exactly, but without any masking. The query q_cross computes dot products with all the encoder keys to get attention scores. These scores are passed through softmax to get attention weights. Then, we take a weighted sum of the encoder values.") as tracker:
            self.play(FadeOut(q_eq), FadeOut(k_eq), FadeOut(arrow_up), FadeOut(label_z))

            cross_math = MathTex(
                r"\text{Scores} = q_{cross} \cdot K_{enc}^T",
                r"\text{Weights} = \text{Softmax}(\text{Scores})",
                r"\text{Output}_{cross} = \text{Weights} \cdot V_{enc}"
            , color=BLACK).arrange(DOWN, buff=0.5).scale(0.8).next_to(title_cross, DOWN, buff=1)

            self.play(Write(cross_math))
            self.wait(2)

        with self.voiceover(text="This output vector then passes through a Feed-Forward Network and layer normalization, resulting in the final contextualized representation for the token 'cat' at this decoder layer.") as tracker:
            ffn_math = MathTex(
                r"\text{Final}_{cat} = \text{LayerNorm}(\text{FFN}(\text{Output}_{cross}) + \text{Output}_{cross})",
                color=BLACK
            ).scale(0.7).next_to(cross_math, DOWN, buff=0.7)

            self.play(Write(ffn_math))
            self.wait(2)

        # 4. Forward vs Reverse Pass
        with self.voiceover(text="So far, we have completely traced the forward pass. Data flows from the embedding, through Masked Self-Attention, into Cross-Attention, and finally out through the Feed-Forward Network. Let's visualize this macroscopic flow with solid arrows.") as tracker:
            self.play(FadeOut(title_cross), FadeOut(cross_math), FadeOut(ffn_math))

            forward_title = Text("Forward Pass vs Backpropagation", color=BLACK).scale(0.8).to_edge(LEFT, buff=1).shift(UP * 1)
            self.play(Write(forward_title))

            node_emb = Text("Embedding", color=BLACK).scale(0.6).shift(LEFT*4 + DOWN*1)
            node_msa = Text("Masked Attn", color=BLACK).scale(0.6).shift(LEFT*1 + DOWN*1)
            node_cross = Text("Cross Attn", color=BLACK).scale(0.6).shift(RIGHT*2 + DOWN*1)
            node_ffn = Text("FFN", color=BLACK).scale(0.6).shift(RIGHT*5 + DOWN*1)

            self.play(Write(node_emb), Write(node_msa), Write(node_cross), Write(node_ffn))

            a1 = Arrow(node_emb.get_right(), node_msa.get_left(), buff=0.1, color=GREEN)
            a2 = Arrow(node_msa.get_right(), node_cross.get_left(), buff=0.1, color=GREEN)
            a3 = Arrow(node_cross.get_right(), node_ffn.get_left(), buff=0.1, color=GREEN)

            self.play(GrowArrow(a1), GrowArrow(a2), GrowArrow(a3))
            self.wait(2)

        with self.voiceover(text="During the training phase, after the model generates a prediction and computes a loss, we perform backpropagation. The gradient of the loss, often referred to as the error signal, flows backward through the network. We will visualize this reverse flow with dashed red arrows.") as tracker:
            b3 = DashedLine(node_ffn.get_top() + LEFT*0.2, node_cross.get_top() + RIGHT*0.2, color=RED).add_tip()
            b2 = DashedLine(node_cross.get_top() + LEFT*0.2, node_msa.get_top() + RIGHT*0.2, color=RED).add_tip()
            b1 = DashedLine(node_msa.get_top() + LEFT*0.2, node_emb.get_top() + RIGHT*0.2, color=RED).add_tip()

            self.play(FadeIn(b3), FadeIn(b2), FadeIn(b1))
            self.wait(2)

        with self.voiceover(text="As the gradient signal propagates backwards, it is used to calculate the gradients for all the learnable parameters in each layer. Using the chain rule of calculus, we compute the partial derivatives of the loss with respect to the weight matrices.") as tracker:
            grad_math = MathTex(
                r"\frac{\partial L}{\partial W_Q} = \text{Input}^T \cdot \frac{\partial L}{\partial Q}",
                r"\frac{\partial L}{\partial W_K} = \text{Input}^T \cdot \frac{\partial L}{\partial K}",
                r"\frac{\partial L}{\partial W_V} = \text{Input}^T \cdot \frac{\partial L}{\partial V}"
            , color=BLACK).arrange(DOWN, buff=0.3).scale(0.7).next_to(forward_title, DOWN, buff=0.5)

            self.play(Write(grad_math))
            self.wait(2)

        with self.voiceover(text="Once these gradients are calculated, an optimizer like Adam or SGD updates the matrices W_Q, W_K, W_V, the output projection matrix W_O, and the FFN weights. This meticulous adjustment is what allows the transformer to learn complex language patterns over time.") as tracker:
            self.wait(2)

        # 5. Key Interview Gotcha
        with self.voiceover(text="To conclude our deep dive, let us highlight a critical concept that frequently trips up candidates in machine learning engineering interviews. It is essential that you understand and can confidently explain this distinction.") as tracker:
            self.play(
                FadeOut(forward_title), FadeOut(grad_math),
                FadeOut(node_emb), FadeOut(node_msa), FadeOut(node_cross), FadeOut(node_ffn),
                FadeOut(a1), FadeOut(a2), FadeOut(a3),
                FadeOut(b1), FadeOut(b2), FadeOut(b3),
                FadeOut(token_boxes)
            )

        with self.voiceover(text="The most common gotcha regarding the Decoder stack involves the Cross-Attention layer. You must remember the exact sources of the Query, Key, and Value vectors.") as tracker:
            box = Rectangle(width=10, height=4, color=RED, fill_opacity=0.1)
            gotcha_title = Text("Interview Gotcha: Cross-Attention Sources", color=RED).scale(0.8).next_to(box.get_top(), DOWN, buff=0.3)

            q_text = MathTex(r"\text{Query (Q)} \leftarrow \text{Comes from Decoder (Self-Attention Output)}", color=BLACK).scale(0.7).next_to(gotcha_title, DOWN, buff=0.5)
            kv_text = MathTex(r"\text{Key (K), Value (V)} \leftarrow \text{Come from Encoder (Final Output)}", color=BLACK).scale(0.7).next_to(q_text, DOWN, buff=0.3)

            gotcha_group = VGroup(box, gotcha_title, q_text, kv_text)
            self.play(FadeIn(gotcha_group))
            self.wait(2)

        with self.voiceover(text="A classic trick question is to ask where the Queries come from in cross-attention. Candidates often mistakenly say they come from the encoder, or that all three come from the decoder. The correct answer is that the Query is generated by the decoder's current state, effectively asking the encoder 'what information do you have that is relevant to me right now?', while the Keys and Values are provided by the encoder's complete context. Understanding this fundamental routing of information is key to mastering the Transformer architecture.") as tracker:
            self.wait(3)

        with self.voiceover(text="Thank you for watching Day 51. Be sure to review these matrix operations and gradient flows, as they form the backbone of modern Generative AI. Good luck with your interview preparation.") as tracker:
            self.play(FadeOut(gotcha_group))
            self.wait(2)
