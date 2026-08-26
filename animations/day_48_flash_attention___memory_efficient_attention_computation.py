from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class FlashAttentionScene(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Section 1: Intro & Input Setup
        with self.voiceover(text="""Welcome to Day 48 of our AI and Machine Learning interview preparation series.
            Today, we are going deep into Flash Attention and the memory efficient computation of attention mechanisms.
            We will trace the exact data flow of a real token passing through a transformer's attention layer.
            Understanding the step-by-step matrix operations is crucial for answering advanced interview questions about model performance, memory bandwidth limits, and optimization techniques.
            Flash Attention fundamentally changes how these computations are ordered to minimize slow memory reads and writes.
            We start with a simple sequence of three tokens. Let's imagine our input sentence is 'The cat sat'.""") as tracker:

            tokens_text = VGroup(
                Text("The", color=BLACK, font_size=36),
                Text("cat", color=BLACK, font_size=36),
                Text("sat", color=BLACK, font_size=36)
            ).arrange(RIGHT, buff=1.0).move_to(UP*2.5)

            boxes = VGroup()
            for text in tokens_text:
                box = SurroundingRectangle(text, color=BLACK, buff=0.2)
                boxes.add(box)

            input_group = VGroup(tokens_text, boxes)
            self.play(FadeIn(input_group))
            self.wait(1)

        with self.voiceover(text="""For our data flow deep dive, we will focus entirely on the active token, which is 'cat'.
            In self-attention, we need to see how 'cat' relates to 'The' and 'sat', but also how it computes its own representation.
            Let's highlight our active token in blue.""") as tracker:

            highlight_box = SurroundingRectangle(tokens_text[1], color=BLUE, fill_opacity=0.3, buff=0.2)
            self.play(Write(highlight_box))

            # Extract just the cat box and text to move them down
            active_token = VGroup(tokens_text[1].copy(), boxes[1].copy(), highlight_box.copy())
            self.play(FadeOut(input_group), FadeOut(highlight_box))
            self.play(active_token.animate.move_to(UP*3))
            self.wait(1)

        # Section 2: Forward Pass Data Flow - Q, K, V Generation
        with self.voiceover(text="""In the forward pass, the first step is projecting our token's input embedding into Query, Key, and Value vectors.
            Let's assume our embedding for 'cat' is a simple one-dimensional vector with three values: 1, 0, and 2.
            We represent this as a 1 by 3 row vector.
            This input vector arrives at the attention stage and must be multiplied by three learned weight matrices: W Q, W K, and W V.""") as tracker:

            emb_vector = MathTex(r"\begin{bmatrix} 1 & 0 & 2 \end{bmatrix}", color=BLACK).next_to(active_token, DOWN, buff=0.5)
            emb_label = Text("Input Embedding (x)", color=BLACK, font_size=24).next_to(emb_vector, LEFT, buff=0.5)

            input_arrow = Arrow(active_token.get_bottom(), emb_vector.get_top(), color=BLACK)
            self.play(Write(input_arrow), FadeIn(emb_vector), FadeIn(emb_label))
            self.wait(2)

        with self.voiceover(text="""Let's visualize the computation for the Query vector.
            We multiply our input vector x by the Query weight matrix, W Q.
            Suppose W Q is a 3 by 3 matrix containing small integer weights.
            The first column is 1, 0, 1. The second column is 0, 1, 0. The third column is 1, 1, 0.
            This matrix multiplication projects our 1 by 3 input into a new 1 by 3 Query vector.""") as tracker:

            wq_matrix = MathTex(
                r"W_Q = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \end{bmatrix}",
                color=PURPLE
            ).move_to(LEFT*3 + UP*0)

            q_calc = MathTex(
                r"q = \begin{bmatrix} 1 & 0 & 2 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \end{bmatrix}",
                color=BLACK
            ).next_to(wq_matrix, DOWN, buff=0.5)

            self.play(FadeIn(wq_matrix))
            self.play(Write(q_calc))
            self.wait(2)

        with self.voiceover(text="""Let's do the math cell by cell.
            For the first element of the Query vector, we take the dot product of the input vector with the first column of W Q.
            That is 1 times 1, plus 0 times 0, plus 2 times 1. The result is 3.
            For the second element, we use the second column. 1 times 0, plus 0 times 1, plus 2 times 1. The result is 2.
            For the third element, we use the third column. 1 times 1, plus 0 times 0, plus 2 times 0. The result is 1.
            So, our resulting Query vector for the token 'cat' is 3, 2, 1.""") as tracker:

            q_result = MathTex(r"= \begin{bmatrix} 3 & 2 & 1 \end{bmatrix}", color=GREEN).next_to(q_calc, RIGHT)
            self.play(Write(q_result))
            self.wait(2)

        with self.voiceover(text="""Similarly, the token's Key and Value vectors are computed using their respective weight matrices, W K and W V.
            In a real transformer, all tokens in the sequence compute their Queries, Keys, and Values simultaneously in large matrix operations.
            For simplicity in our example, let's assume the Key and Value vectors for 'cat' have already been computed.
            Let's say the Key vector k is 1, 2, 0, and the Value vector v is 0, 1, 1.
            Remember, in Flash Attention, the key innovation is avoiding writing these intermediate Q, K, and V matrices out to slow GPU High Bandwidth Memory, or HBM.
            Instead, they are kept in the fast, on-chip SRAM.""") as tracker:

            self.play(FadeOut(wq_matrix), FadeOut(q_calc), q_result.animate.move_to(LEFT*3 + UP*0))
            q_label = MathTex(r"q = \begin{bmatrix} 3 & 2 & 1 \end{bmatrix}", color=GREEN).move_to(LEFT*3 + UP*0)
            self.play(Transform(q_result, q_label))

            k_label = MathTex(r"k = \begin{bmatrix} 1 & 2 & 0 \end{bmatrix}", color=BLACK).next_to(q_label, DOWN, buff=0.5)
            v_label = MathTex(r"v = \begin{bmatrix} 0 & 1 & 1 \end{bmatrix}", color=BLACK).next_to(k_label, DOWN, buff=0.5)

            self.play(FadeIn(k_label), FadeIn(v_label))
            self.wait(2)

        # Section 3: Attention Score & Softmax
        with self.voiceover(text="""The next step in the data flow is calculating the attention scores.
            This determines how much focus the active token 'cat' should place on every other token, including itself.
            We calculate this by taking the dot product of our Query vector with the Key vectors of all tokens.
            Let's assume the Key matrix for the entire sequence 'The cat sat' is structured such that the columns represent the tokens.
            The first column is the key for 'The', the second for 'cat', and the third for 'sat'.
            Let the Key transpose matrix be: first column 0, 1, 1; second column 1, 2, 0; third column 1, 0, 1.""") as tracker:

            self.play(FadeOut(emb_vector), FadeOut(emb_label), FadeOut(input_arrow))

            kt_matrix = MathTex(
                r"K^T = \begin{bmatrix} 0 & 1 & 1 \\ 1 & 2 & 0 \\ 1 & 0 & 1 \end{bmatrix}",
                color=BLACK
            ).move_to(RIGHT*2 + UP*1.5)

            self.play(FadeIn(kt_matrix))
            self.wait(2)

        with self.voiceover(text="""We multiply our Query row vector for 'cat', which is 3, 2, 1, by this K transpose matrix.
            The dot product with the first column (for 'The') is 3 times 0, plus 2 times 1, plus 1 times 1, which equals 3.
            The dot product with the second column (for 'cat') is 3 times 1, plus 2 times 2, plus 1 times 0, which equals 7.
            The dot product with the third column (for 'sat') is 3 times 1, plus 2 times 0, plus 1 times 1, which equals 4.
            So, our raw attention scores, also known as logits, are 3, 7, and 4.""") as tracker:

            score_calc = MathTex(
                r"Scores = q K^T = \begin{bmatrix} 3 & 2 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 & 1 \\ 1 & 2 & 0 \\ 1 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 3 & 7 & 4 \end{bmatrix}",
                color=BLACK
            ).move_to(DOWN*1.5)
            self.play(Write(score_calc))
            self.wait(2)

        with self.voiceover(text="""Standard attention scales these scores by the square root of the dimension, but we will skip scaling here for simplicity.
            The crucial and memory-intensive part is the Softmax operation applied to these scores.
            Softmax turns these raw scores into probabilities that sum to 1.
            To compute Softmax, we exponentiate each score. Let's approximate e to the power of 3 as 20, e to the power of 7 as 1096, and e to the power of 4 as 54.""") as tracker:

            exp_calc = MathTex(
                r"e^{Scores} \approx \begin{bmatrix} 20 & 1096 & 54 \end{bmatrix}", color=BLACK
            ).next_to(score_calc, DOWN, buff=0.5)
            self.play(Write(exp_calc))
            self.wait(2)

        with self.voiceover(text="""Then, we calculate the denominator by summing these exponentials.
            20 plus 1096 plus 54 equals 1170.
            Standard attention calculates this entire matrix for all tokens and writes it to HBM, which is extremely slow.
            Flash Attention uses a technique called tiling to compute these softmax denominators incrementally in blocks without materializing the full matrix.
            Finally, we divide each exponential by the sum to get our attention weights.""") as tracker:

            denom_calc = MathTex(r"\text{Sum} = 20 + 1096 + 54 = 1170", color=BLACK).next_to(exp_calc, DOWN, buff=0.5)
            self.play(Write(denom_calc))
            self.wait(2)

            softmax_result = MathTex(
                r"\text{Softmax} \approx \begin{bmatrix} 0.02 & 0.93 & 0.05 \end{bmatrix}", color=GREEN
            ).next_to(denom_calc, DOWN, buff=0.5)
            self.play(Write(softmax_result))
            self.wait(2)

        with self.voiceover(text="""Notice how the active token 'cat' attends very heavily to itself, with a 93% weight, and pays very little attention to 'The' and 'sat'.
            The final step of the forward pass is to multiply these softmax weights by the Value matrix V to compute the output vector for this token.
            We won't do the full math here, but the output vector is a weighted sum of all Value vectors in the sequence, and it flows out to the Feed Forward Network stage.""") as tracker:

            self.play(FadeOut(score_calc), FadeOut(exp_calc), FadeOut(denom_calc))
            self.play(softmax_result.animate.move_to(DOWN*1))

            out_arrow = Arrow(softmax_result.get_bottom(), DOWN*3, color=GREEN)
            out_label = Text("To Next Stage (FFN)", color=GREEN, font_size=24).next_to(out_arrow, RIGHT)
            self.play(Write(out_arrow), FadeIn(out_label))
            self.wait(2)

        # Section 4: Backward Pass & Gradients
        with self.voiceover(text="""That completes the forward pass. Now, let's switch gears and look at the backward pass during training.
            In backpropagation, the error signal, or gradient of the loss with respect to the output, flows backward through the network.
            This tells us how we need to update our weight matrices, like W Q, to minimize the loss.
            Let's clear the board and trace the gradients flowing backwards in red.""") as tracker:

            self.play(
                FadeOut(q_result), FadeOut(q_label), FadeOut(k_label), FadeOut(v_label),
                FadeOut(kt_matrix), FadeOut(softmax_result), FadeOut(out_arrow), FadeOut(out_label),
                FadeOut(active_token)
            )
            self.wait(1)

        with self.voiceover(text="""During backprop, the gradient flows back from the loss L.
            We represent the gradient of the loss with respect to the output of the attention layer as dL over dOut.
            This gradient vector flows backward along dashed red arrows.
            Ultimately, these gradients are used to update several distinct sets of learned parameters within this layer.""") as tracker:

            loss_node = Text("Loss (L)", color=RED).move_to(RIGHT*3 + DOWN*2)
            grad_out = MathTex(r"\frac{\partial L}{\partial \text{Out}}", color=RED).next_to(loss_node, UP, buff=1)
            back_arrow1 = DashedLine(loss_node.get_top(), grad_out.get_bottom(), color=RED).add_tip()

            self.play(FadeIn(loss_node))
            self.play(Write(back_arrow1), FadeIn(grad_out))

            # List of all updated weights
            updated_weights = VGroup(
                Text("Weights Updated:", color=BLACK, font_size=28),
                MathTex("W_Q", color=PURPLE),
                MathTex("W_K", color=PURPLE),
                MathTex("W_V", color=PURPLE),
                MathTex("W_O \\text{ (Output)}", color=PURPLE),
                MathTex("\\text{FFN Weights}", color=PURPLE)
            ).arrange(DOWN, aligned_edge=LEFT).move_to(RIGHT*4 + UP*1.5)

            self.play(FadeIn(updated_weights))

            self.wait(2)

        with self.voiceover(text="""To update our Query weight matrix W Q, we need the gradient of the loss with respect to W Q.
            Using the chain rule, this involves multiple steps.
            First, the gradient flows through the Softmax and the Value matrix multiplication to reach the raw attention scores.
            Then, it flows back to the Query vector q.""") as tracker:

            grad_q = MathTex(r"\frac{\partial L}{\partial q}", color=RED).move_to(LEFT*2 + UP*0)
            back_arrow2 = DashedLine(grad_out.get_left(), grad_q.get_right(), color=RED).add_tip()

            self.play(Write(back_arrow2), FadeIn(grad_q))
            self.wait(2)

        with self.voiceover(text="""Finally, from the Query vector q, the gradient flows back to the Query weight matrix W Q.
            Mathematically, the gradient dL over d W Q is equal to the input vector x transpose, multiplied by the gradient dL over dq.
            Let's write out this full chain rule derivation using math text.""") as tracker:

            grad_wq = MathTex(r"\frac{\partial L}{\partial W_Q} = x^T \frac{\partial L}{\partial q}", color=RED).move_to(LEFT*2 + UP*2)
            back_arrow3 = DashedLine(grad_q.get_top(), grad_wq.get_bottom(), color=RED).add_tip()

            self.play(Write(back_arrow3), FadeIn(grad_wq))
            self.wait(2)

        with self.voiceover(text="""To be completely rigorous, the full derivation incorporates the intermediate steps we discussed.
            The gradient of the loss with respect to W Q equals the gradient with respect to the Output,
            times the derivative of the Output with respect to the Softmax probabilities,
            times the derivative of Softmax with respect to the Scores,
            times the derivative of the Scores with respect to q,
            and finally times the derivative of q with respect to W Q.
            This identical chain rule logic applies when deriving the updates for the other weights we labeled, like the Key matrix W K, Value matrix W V, the attention Output matrix W O, and the dense layers in the Feed Forward Network.""") as tracker:

            full_chain = MathTex(
                r"\frac{\partial L}{\partial W_Q} = \frac{\partial L}{\partial \text{Out}} \frac{\partial \text{Out}}{\partial \text{Softmax}} \frac{\partial \text{Softmax}}{\partial \text{Scores}} \frac{\partial \text{Scores}}{\partial q} \frac{\partial q}{\partial W_Q}",
                color=RED, font_size=36
            ).move_to(DOWN*3)

            self.play(Write(full_chain))
            self.wait(3)

        with self.voiceover(text="""In standard attention, calculating these backward pass gradients requires storing the huge attention score and softmax matrices from the forward pass in memory.
            This is because the derivative of Softmax depends on the Softmax outputs.
            If the sequence length is long, for example 8000 or 32000 tokens, these intermediate matrices grow quadratically and take up gigabytes of VRAM.
            When you run out of VRAM, the training process crashes with an out-of-memory error.
            Flash Attention solves this memory explosion brilliantly. Instead of storing the massive intermediate matrices in HBM, it recomputes them on the fly during the backward pass inside the fast on-chip SRAM.
            This completely avoids reading and writing to the slow High Bandwidth Memory entirely. The final mathematical gradients for our W_Q, W_K, W_V, W_O, and FFN weights are identical, but the memory bottleneck is entirely bypassed, allowing us to train on much longer sequences.""") as tracker:

            flash_callout = SurroundingRectangle(full_chain, color=BLUE, buff=0.2)
            recompute_text = Text("Flash Attention: Recompute in SRAM instead of storing in HBM!", color=BLUE, font_size=24).next_to(flash_callout, UP)

            self.play(Write(flash_callout), FadeIn(recompute_text))
            self.wait(3)

        # Section 5: Interview Callout
        with self.voiceover(text="""To wrap up, let's highlight a key interview insight.
            A very common 'gotcha' question in Machine Learning systems interviews is:
            'Why does Flash Attention make training faster even though it does more math operations by recomputing values?'
            Intuitively, doing more math operations should take more time. But in reality, it's significantly faster.""") as tracker:

            self.play(
                FadeOut(loss_node), FadeOut(grad_out), FadeOut(back_arrow1),
                FadeOut(grad_q), FadeOut(back_arrow2), FadeOut(grad_wq), FadeOut(back_arrow3),
                FadeOut(full_chain), FadeOut(flash_callout), FadeOut(recompute_text), FadeOut(updated_weights)
            )

            box = Rectangle(width=10, height=4, color=BLACK)
            title = Text("Key Interview Insight", color=RED).move_to(box.get_top() + DOWN*0.5)
            self.play(Write(box), FadeIn(title))
            self.wait(1)

        with self.voiceover(text="""The answer comes down to Memory Bandwidth limits.
            Modern GPU architectures, like Nvidia's A100 or H100, have immense computational power, calculating trillions of floating-point operations per second.
            However, they are bottlenecked by how fast they can move data from the main High Bandwidth Memory to those compute cores.
            They are not bottlenecked by the compute cores themselves.
            Attention computations are memory-bound operations. By fusing operations and keeping the data in the very fast, on-chip SRAM, Flash Attention drastically reduces the expensive HBM reads and writes.
            So, even though it performs extra arithmetic to recompute the forward pass during backprop, skipping the slow memory transfers makes the overall wall-clock training time significantly faster.
            Always remember this fundamental rule of systems engineering: memory access is often orders of magnitude more expensive than raw arithmetic computation.
            Thank you for watching Day 48 of the AI Engineering Mastery series. We hope this deep dive into Flash Attention helps you ace your next technical interview. See you next time.""") as tracker:

            insight1 = Text("Problem: GPUs are bottlenecked by memory bandwidth (HBM), not compute.", font_size=24, color=BLACK).next_to(title, DOWN, buff=0.5)
            insight2 = Text("Flash Attention avoids slow HBM read/writes by fusing operations in SRAM.", font_size=24, color=BLACK).next_to(insight1, DOWN, buff=0.3)
            insight3 = Text("Result: Faster wall-clock time despite recomputing values in the backward pass.", font_size=24, color=GREEN).next_to(insight2, DOWN, buff=0.3)

            self.play(FadeIn(insight1))
            self.wait(1)
            self.play(FadeIn(insight2))
            self.wait(1)
            self.play(FadeIn(insight3))
            self.wait(3)

            self.play(FadeOut(box), FadeOut(title), FadeOut(insight1), FadeOut(insight2), FadeOut(insight3))
            self.wait(1)
