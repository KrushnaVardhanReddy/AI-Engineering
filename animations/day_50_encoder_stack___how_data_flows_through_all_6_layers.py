from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np

class TransformerEncoderStackFlow(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Scene 1: Introduction and Token Setup
        with self.voiceover(text="""
            Welcome to Day 50 of our comprehensive AI Engineering Mastery series. Today, we are embarking on a deep, technical dive into the architecture of the Transformer Encoder Stack. Specifically, we will be meticulously tracing how data flows through all six layers of a standard Transformer encoder during both the forward and backward passes. We will trace the exact path of the simple example sentence 'The cat sat' to understand the mechanics at a highly granular level, examining every mathematical operation along the way. We begin our journey with our input consisting of three initial tokens.
            Before these tokens even enter the first encoder layer, they have already been converted into high-dimensional continuous vector representations known as embeddings. Furthermore, because the Transformer processes all tokens in parallel rather than sequentially like a Recurrent Neural Network, these embeddings have been combined with positional encodings. This crucial step ensures that the model retains information about the relative and absolute positions of the words in the sentence.
            Let us visualize these three tokens as labeled boxes representing our input vectors. For the duration of this analysis, we will highlight the active token we are tracking, 'cat', in a distinct blue color to differentiate it as it propagates through the complex network of transformations.
            This entire process is fundamentally critical to understanding how multi-head self-attention mechanisms and position-wise feed-forward networks process information in parallel while simultaneously maintaining and enriching sequential context. We will walk through the exact mathematical operations step by step, utilizing concrete example numbers and matrices so you can visually and quantitatively see exactly how the query, key, and value matrices interact with one another. We will detail how the raw attention scores are calculated, how they are normalized using the softmax function to form a probability distribution, and how the final contextualized output vector is generated.
            Later in the animation, we will also explore the reverse process: how the gradient of the loss flows backward during backpropagation to update the numerous parameters and weights within the network. Let's begin by thoroughly setting up and examining our input tokens on the whiteboard.
        """) as tracker:
            title = Text("Transformer Encoder: Data Flow", color=BLACK, font_size=48).to_edge(UP)
            self.play(Write(title))

            tokens = VGroup(
                Text("The", color=BLACK),
                Text("cat", color=BLUE, weight=BOLD),
                Text("sat", color=BLACK)
            ).arrange(RIGHT, buff=1.5)

            boxes = VGroup(*[
                SurroundingRectangle(t, color=BLUE if i==1 else BLACK, buff=0.3)
                for i, t in enumerate(tokens)
            ])

            token_group = VGroup(tokens, boxes).move_to(ORIGIN)
            self.play(FadeIn(token_group))
            self.wait(2)
            self.play(token_group.animate.shift(UP * 2 + LEFT * 3).scale(0.7))
            self.wait(2)

        # Scene 2: Forward Pass - Linear Projections for Q, K, V
        with self.voiceover(text="""
            In the forward pass, the first major computational block inside any encoder layer is the multi-head self-attention mechanism. For our designated token 'cat', its input vector, which we will mathematically refer to as X sub cat, arrives at this specific stage. To participate in self-attention, this input vector must be linearly projected into three distinct semantic spaces. This is achieved by multiplying it by three learned weight matrices: W_Q for the queries, W_K for the keys, and W_V for the values.
            Let's dissect the query calculation first. Here, the token 'cat' is represented by a highly simplified, three-dimensional vector for illustrative purposes: 1.0, 0.5, and 0.2. It is multiplied by the Query weight matrix W_Q, which in our example is a three by three matrix, to produce the new query vector q. On the screen, we can clearly see the input vector on the left, the weight matrix in the middle, and the resulting query vector on the right.
            The matrix multiplication involves taking the dot product of our row input vector with each individual column of the weight matrix. Let's painstakingly trace this fundamental calculation. For the first output element of the query vector, we multiply the first input element 1.0 by the first matrix element 0.1, add the product of 0.5 and 0.3, and add the product of 0.2 and 0.0. This yields 0.1 plus 0.15 plus 0, which exactly equals 0.25.
            Similarly, we calculate the second element by computing the dot product with the second column: 1.0 times 0.4, plus 0.5 times 0.2, plus 0.2 times 0.5, resulting in 0.4 plus 0.1 plus 0.1, giving us 0.60. For the third element, the dot product with the third column yields 1.0 times 0.2, plus 0.5 times 0.4, plus 0.2 times 0.0, totaling 0.40.
            This newly computed vector, q, essentially represents a question: it embodies what the token 'cat' is conceptually 'looking for' in the other tokens within the sequence to establish context. Concurrently, the network performs the exact same type of matrix multiplication operations using the W_K and W_V matrices to generate the key and value vectors. The key vector acts as an identifier, representing what the token 'cat' intrinsically 'contains', while the value vector holds the actual semantic payload or content that will eventually be aggregated. Let's carefully observe the full operation for the query matrix multiplication on the board.
        """) as tracker:

            q_eq = MathTex(
                r"\mathbf{X}_{\text{cat}} \times \mathbf{W}_Q = \mathbf{q}_{\text{cat}}",
                color=BLACK
            ).next_to(token_group, DOWN, buff=1.5)
            self.play(Write(q_eq))
            self.wait(2)

            x_mat = MathTex(
                r"\begin{bmatrix} 1.0 & 0.5 & 0.2 \end{bmatrix}",
                color=BLUE
            )
            times = MathTex(r"\times", color=BLACK)
            wq_mat = MathTex(
                r"\begin{bmatrix} 0.1 & 0.4 & 0.2 \\ 0.3 & 0.2 & 0.4 \\ 0.0 & 0.5 & 0.0 \end{bmatrix}",
                color=PURPLE
            )
            equals = MathTex(r"=", color=BLACK)
            q_mat = MathTex(
                r"\begin{bmatrix} 0.25 & 0.60 & 0.40 \end{bmatrix}",
                color=GREEN
            )

            q_calc = VGroup(x_mat, times, wq_mat, equals, q_mat).arrange(RIGHT, buff=0.3).next_to(q_eq, DOWN, buff=0.5)

            self.play(FadeIn(x_mat))
            self.play(FadeIn(times), FadeIn(wq_mat))
            self.wait(2)
            self.play(FadeIn(equals), FadeIn(q_mat))
            self.wait(3)

        # Scene 3: Self-Attention Scores and Softmax
        with self.voiceover(text="""
            Now that we have successfully computed the query, key, and value vectors for all the tokens in our sequence, the next critical phase is to compute the self-attention scores. The token 'cat' must now compare its query vector against the key vectors of all tokens in the sentence, including itself, the preceding word 'The', and the succeeding word 'sat'.
            This comparison is mathematically executed by calculating the dot product between the query vector of 'cat' and the transposed key vectors of the respective other tokens. This dot product essentially measures the alignment or similarity between the query and the keys. A higher dot product indicates a stronger relevance or match.
            To maintain numerical stability, especially with high-dimensional vectors, these raw dot products are subsequently scaled down by dividing them by the square root of the dimension of the key vectors. Let's assume, for the sake of our demonstration, that the resulting scaled raw dot products are 1.2 for the token 'The', a much higher 3.5 for the token 'cat' itself, and 0.8 for the token 'sat'.
            Next, we apply the softmax function to these raw, unnormalized scores. The purpose of the softmax function is to convert these arbitrary scores into a valid probability distribution, ensuring that all the resulting weights are positive and sum exactly to 1.0.
            The softmax function operates by exponentiating each individual score and then dividing it by the sum of all the exponentiated scores. Let's walk through this softmax calculation. Euler's number E raised to the power of 3.5 is significantly larger than E raised to the power of 1.2 or E raised to the power of 0.8. The exponential function amplifies the differences between the scores, making the largest score dominate.
            When we perform the normalization step by dividing by the sum, we obtain our final attention weights: approximately 0.08 for 'The', a dominant 0.86 for 'cat', and 0.06 for 'sat'.
            Notice how the token strongly attends to itself, a common phenomenon in self-attention, but it also crucially picks up some contextual information from the surrounding words based on their relevance. These normalized weights dictate exactly how much of each respective token's value vector we should dynamically include in the final aggregated output representation for our focal token, 'cat'.
        """) as tracker:
            self.play(FadeOut(q_eq), FadeOut(q_calc))

            scores_title = Text("Self-Attention: Dot Product & Softmax", color=BLACK, font_size=36).next_to(token_group, DOWN, buff=1.0)
            self.play(Write(scores_title))

            raw_scores = MathTex(
                r"\text{Scores} = \begin{bmatrix} 1.2 & 3.5 & 0.8 \end{bmatrix}",
                color=BLACK
            ).next_to(scores_title, DOWN, buff=0.5)
            self.play(FadeIn(raw_scores))
            self.wait(2)

            softmax_eq = MathTex(
                r"\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}",
                color=BLACK
            ).next_to(raw_scores, DOWN, buff=0.5)
            self.play(Write(softmax_eq))
            self.wait(2)

            attn_weights = MathTex(
                r"\text{Weights} = \begin{bmatrix} 0.08 & 0.86 & 0.06 \end{bmatrix}",
                color=GREEN
            ).next_to(softmax_eq, DOWN, buff=0.5)
            self.play(FadeIn(attn_weights))
            self.wait(3)

        # Scene 4: Weighted Sum of Values and Output
        with self.voiceover(text="""
            Armed with the normalized attention weights, the final step within the self-attention sub-layer is to compute the weighted sum of the value vectors. We methodically multiply the value vector of the token 'The' by its corresponding weight of 0.08. We multiply the value vector of 'cat' by its heavy weight of 0.86. Finally, we multiply the value vector of 'sat' by its assigned weight of 0.06.
            We then perform an element-wise addition of these three scaled vectors. This aggregation process produces a brand new, context-aware output vector for the token 'cat', which we will denote as Z sub cat. This dense vector now encapsulates profoundly richer information; it contains not just the original semantic meaning of 'cat', but also dynamically integrated, relevant contextual nuances synthesized from the entire surrounding sentence, precisely weighted by the learned attention mechanism.
            However, the processing within a single encoder layer is not yet complete. After the multi-head attention block, this intermediate output vector is passed through an Add and Norm layer. This layer implements a residual connection—adding the original input vector X back to the attention output Z—followed by a layer normalization operation. This architectural detail is vital for preventing the vanishing gradient problem and stabilizing the learning process in deep networks.
            Subsequently, the normalized vector flows into a fully connected Feed-Forward Network, which independently applies two linear transformations separated by a non-linear activation function, typically a ReLU or a GELU. This adds essential non-linearity to the model, allowing it to learn more complex representations. The output of the Feed-Forward Network goes through another Add and Norm layer.
            This entire intricate sequence of operations constitutes just one single layer. In our standard architecture, this exact identical process is repeated sequentially across all six layers of the encoder stack, with each layer continuously refining and abstracting the representations. Let's visualize this final contextualized output vector flowing out as a definitive green arrow, advancing to the next sequential stage of the hierarchical stack, with the dynamically updated numerical values vividly highlighted in green.
        """) as tracker:
            self.play(FadeOut(raw_scores), FadeOut(softmax_eq), FadeOut(attn_weights), FadeOut(scores_title))

            context_title = Text("Weighted Sum & Output", color=BLACK, font_size=36).next_to(token_group, DOWN, buff=1.0)
            self.play(Write(context_title))

            weighted_sum = MathTex(
                r"\mathbf{Z}_{\text{cat}} = 0.08\mathbf{v}_{\text{The}} + 0.86\mathbf{v}_{\text{cat}} + 0.06\mathbf{v}_{\text{sat}}",
                color=BLACK
            ).next_to(context_title, DOWN, buff=0.5)

            final_out = MathTex(
                r"\mathbf{Z}_{\text{cat}} = \begin{bmatrix} 0.85 & 1.20 & 0.30 \end{bmatrix}",
                color=GREEN
            ).next_to(weighted_sum, DOWN, buff=0.5)

            self.play(Write(weighted_sum))
            self.wait(2)
            self.play(FadeIn(final_out))
            self.wait(3)

            arrow_out = Arrow(start=UP, end=DOWN, color=GREEN).next_to(final_out, DOWN)
            next_layer = Text("To Layer 2", color=BLACK, font_size=24).next_to(arrow_out, DOWN)
            self.play(FadeIn(arrow_out), Write(next_layer))
            self.wait(3)

        # Scene 5: Backpropagation and Gradient Flow
        with self.voiceover(text="""
            Understanding the forward pass is only half the battle. During the training phase, after the forward pass computes the final predictions and evaluates the loss function against the ground truth, the error signal must flow in reverse. This gradient flows back through the entire network architecture to systematically update the millions of weights and parameters. This iterative optimization process is known as backpropagation. Let's carefully trace this reverse flow of information.
            We will depict the gradient flowing backward using prominent dashed red arrows to contrast with the forward data flow. The gradient of the scalar loss function with respect to the output vector, which we will mathematically denote as the partial derivative of L with respect to Z, travels backward through the upper Add and Norm layer, descends through the Feed-Forward Network, passes through the lower Add and Norm layer, and finally permeates into the multi-head attention mechanism.
            Relying heavily on the chain rule of calculus, this incoming gradient is systematically utilized to compute the specific local gradients for every single learnable weight matrix: the query matrix W_Q, the key matrix W_K, the value matrix W_V, the final linear output projection matrix W_O, and all the dense weights within the Feed-Forward Network.
            Let's examine a formal, multi-step gradient derivation for the query matrix W_Q. The gradient of the total loss with respect to the matrix W_Q is mathematically equivalent to the matrix product of the transposed input vector X and the gradient of the loss with respect to the intermediate query vector q. Once this precise gradient is calculated, we perform a gradient descent step. We update the original W_Q matrix by subtracting a fraction of this gradient, determined by the chosen learning rate, denoted by the Greek letter eta.
            We explicitly highlight the newly updated weights and mathematical terms in red on the screen to visually emphasize that they are actively being optimized and adjusted. It is crucial to remember that this complex reverse flow of gradients happens simultaneously and continuously across all tokens in the batch and across all six layers of the encoder stack, meticulously and iteratively tuning the parameters to progressively minimize the overall loss.
        """) as tracker:
            self.play(
                FadeOut(context_title), FadeOut(weighted_sum), FadeOut(final_out), FadeOut(arrow_out), FadeOut(next_layer)
            )

            backprop_title = Text("Backward Pass (Backpropagation)", color=RED, font_size=36).next_to(token_group, DOWN, buff=1.0)
            self.play(Write(backprop_title))

            dashed_arrow = DashedLine(start=DOWN*2, end=UP*0.5, color=RED).add_tip().next_to(backprop_title, DOWN, buff=0.5)
            grad_label = MathTex(r"\frac{\partial \mathcal{L}}{\partial \mathbf{Z}}", color=RED).next_to(dashed_arrow, RIGHT)
            self.play(FadeIn(dashed_arrow), FadeIn(grad_label))
            self.wait(2)

            grad_eq = MathTex(
                r"\frac{\partial \mathcal{L}}{\partial \mathbf{W}_Q} = \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{q}}",
                color=BLACK
            ).next_to(dashed_arrow, DOWN, buff=0.5)
            self.play(Write(grad_eq))
            self.wait(2)

            update_eq = MathTex(
                r"\mathbf{W}_Q \leftarrow \mathbf{W}_Q - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}_Q}",
                color=RED
            ).next_to(grad_eq, DOWN, buff=0.5)
            self.play(FadeIn(update_eq))
            self.wait(3)

            updated_weights = Text("Weights Updated: W_Q, W_K, W_V, W_O, FFN", color=PURPLE, font_size=28).next_to(update_eq, DOWN, buff=0.5)
            self.play(Write(updated_weights))
            self.wait(3)

        # Scene 6: Key Interview Insight
        with self.voiceover(text="""
            Before we conclude this intensive session, let's highlight a paramount key interview insight that frequently trips up even experienced candidates. A very common and deceptively simple question posed in rigorous AI engineering system design interviews is: 'Why is it strictly necessary to inject positional encodings into the input embeddings before they enter the encoder stack?'
            The classic 'gotcha' here is failing to recognize and clearly articulate that the standard dot-product self-attention mechanism, by its very mathematical nature, is entirely permutation equivariant. If you neglect to explicitly add positional encodings to the input embeddings at the very beginning of the pipeline, the model has absolutely no inherent concept of sequential word order or relative positioning.
            To illustrate, without positional encodings, the network would process the sentence 'The cat sat' identically to the nonsensical permutation 'sat The cat'. Both would yield the exact same set of un-ordered contextualized output embeddings. This completely destroys the underlying semantic meaning and syntax of natural language.
            Always remember to emphasize that the data flowing through the complex layers must have deterministic positional information injected right at the inception of the forward pass. This critical addition allows the self-attention mechanism to intelligently factor in the distance and sequential order of words while computing those crucial attention weights we calculated earlier. Grasping this fundamental architectural nuance of the data flow is absolutely essential for successfully designing, debugging, and explaining modern Transformer-based large language models. Thank you for your dedication in joining this extensive deep dive into the inner workings of the Encoder Stack. Keep mastering these foundational concepts.
        """) as tracker:
            self.play(
                FadeOut(backprop_title), FadeOut(dashed_arrow), FadeOut(grad_label), FadeOut(grad_eq), FadeOut(update_eq), FadeOut(updated_weights)
            )

            insight_box = Rectangle(width=11, height=5, color=RED, fill_color=WHITE, fill_opacity=1).move_to(DOWN*1.5)
            insight_title = Text("Key Interview Insight", color=RED, font_size=36, weight=BOLD).move_to(insight_box.get_top() + DOWN*0.6)

            insight_text1 = Text("Attention is fundamentally permutation equivariant.", color=BLACK, font_size=28).next_to(insight_title, DOWN, buff=0.5)
            insight_text2 = Text("Without positional encodings added at the input layer,", color=BLACK, font_size=28).next_to(insight_text1, DOWN, buff=0.3)
            insight_text3 = Text("the model has zero concept of sequential word order.", color=BLACK, font_size=28).next_to(insight_text2, DOWN, buff=0.3)
            insight_text4 = Text("'The cat sat' == 'sat The cat'", color=BLUE, font_size=32).next_to(insight_text3, DOWN, buff=0.4)

            self.play(FadeIn(insight_box), Write(insight_title))
            self.wait(1)
            self.play(Write(insight_text1))
            self.wait(2)
            self.play(Write(insight_text2))
            self.wait(2)
            self.play(Write(insight_text3))
            self.wait(2)
            self.play(Write(insight_text4))
            self.wait(5)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(2)
