import numpy as np
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class SelfAttentionQKVComputation(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # We need a robust 10-12 minute video. That means a script of around 1500+ words.
        # It's an interview deep dive, so we will walk through each step slowly, with concrete numbers.

        with self.voiceover(text="""
            Welcome to Day 45 of our AI Engineering Mastery series.
            Today, we are diving deep into the core mechanism of the Transformer architecture: Self-Attention.
            Specifically, we are going to look at exactly how the Query, Key, and Value matrices are computed from a single token,
            and how these computations flow through both the forward pass and the backpropagation phase.
            This is a very common topic in advanced machine learning and AI engineering interviews,
            so we will trace a real token through this stage step by step, examining the actual mathematical operations.
            Let's begin by setting up our input sequence.
        """) as tracker:
            self.wait(2)

        # 1. Input Setup
        tokens = ["The", "cat", "sat"]
        token_boxes = VGroup(*[
            VGroup(
                Rectangle(width=2, height=1, color=BLACK, fill_color=WHITE, fill_opacity=1),
                Text(word, color=BLACK).scale(0.8)
            ) for word in tokens
        ]).arrange(RIGHT, buff=0.5).to_edge(UP, buff=1)

        with self.voiceover(text="""
            We start with three tokens: 'The', 'cat', and 'sat'.
            In a real model, these words would first be converted into token IDs, and then into dense embedding vectors.
            For this deep dive, we will focus our attention entirely on the second token, 'cat'.
            Let's highlight it in blue to track its journey.
        """) as tracker:
            self.play(FadeIn(token_boxes, shift=UP))
            self.wait(1)
            self.play(
                token_boxes[1][0].animate.set_fill(BLUE, opacity=0.3),
                token_boxes[1][0].animate.set_color(BLUE)
            )
            self.wait(2)

        cat_embedding_text = MathTex(
            r"X_{\text{cat}} = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}",
            color=BLACK
        ).next_to(token_boxes[1], DOWN, buff=1.5)

        arrival_arrow = Arrow(token_boxes[1].get_bottom(), cat_embedding_text.get_top(), buff=0.1, color=BLACK)

        with self.voiceover(text="""
            After the embedding layer, our token 'cat' is represented as a dense numerical vector arriving into the current stage.
            For the sake of this visualization, let's assume a simplified embedding dimension of two.
            So, the input vector for 'cat', which we will call X sub cat, is a 2-dimensional vector containing the values 1.0 and 0.5.
            In a real large language model, this vector would typically have thousands of dimensions, such as 4096 in Llama 3,
            but the mathematical operations remain exactly the same.
        """) as tracker:
            self.play(Write(arrival_arrow))
            self.play(Write(cat_embedding_text))
            self.wait(2)

        # Move X_cat to left to start matrix multiplications
        with self.voiceover(text="""
            Now, let's clear the screen and focus on how this single input vector X sub cat is transformed.
            The fundamental concept of self-attention is that every token needs to figure out its role in the sequence.
            To do this, it needs to ask questions about other tokens, answer questions from other tokens,
            and offer its own information to be used.
            These three distinct roles are captured by three distinct vectors: the Query, the Key, and the Value.
        """) as tracker:
            self.play(FadeOut(token_boxes), FadeOut(arrival_arrow), cat_embedding_text.animate.to_edge(LEFT).shift(UP*2))
            self.wait(2)

        # 2. Compute Query
        with self.voiceover(text="""
            In the forward pass... we compute the Query vector first. The Query represents what the token 'cat' is looking for.
            Perhaps it is looking for an adjective that modifies it, or a verb that it is the subject of.
            To calculate the Query vector, we multiply our input vector X sub cat by a learned weight matrix, W sub Q.
        """) as tracker:
            self.wait(1)

        fwd_arrow = Arrow(cat_embedding_text.get_right(), cat_embedding_text.get_right() + RIGHT * 1, buff=0.1, color=BLACK)
        self.play(Write(fwd_arrow))

        wq_matrix = MathTex(
            r"W_Q = \begin{bmatrix} 2.0 & 0.0 \\ 1.0 & -1.0 \end{bmatrix}",
            color=PURPLE
        ).next_to(fwd_arrow, RIGHT, buff=0.1)

        with self.voiceover(text="""
            Here is our Query weight matrix, W sub Q.
            Notice that it is a 2 by 2 matrix. In this simplified example, the projection dimension is the same as the embedding dimension.
            This matrix contains learnable parameters. Its job is to linearly project the input embedding into the Query space.
            Let's see this matrix multiplication in action.
        """) as tracker:
            self.play(Write(wq_matrix))
            self.wait(2)

        query_calc_eq = MathTex(
            r"q_{\text{cat}} = W_Q X_{\text{cat}}",
            color=BLACK
        ).next_to(wq_matrix, RIGHT, buff=1.5)

        query_calc_expand = MathTex(
            r"= \begin{bmatrix} 2.0 & 0.0 \\ 1.0 & -1.0 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}",
            color=BLACK
        ).next_to(query_calc_eq, DOWN, aligned_edge=LEFT)

        with self.voiceover(text="""
            The formula for the query vector q sub cat is simply the matrix W sub Q multiplied by the vector X sub cat.
            Let's expand this out.
            To perform this matrix-vector multiplication, we take the dot product of each row of the weight matrix
            with the input vector.
        """) as tracker:
            self.play(Write(query_calc_eq))
            self.wait(1)
            self.play(Write(query_calc_expand))
            self.wait(2)

        # Compute values
        q_val_1 = 2.0 * 1.0 + 0.0 * 0.5
        q_val_2 = 1.0 * 1.0 + (-1.0) * 0.5

        query_result = MathTex(
            r"= \begin{bmatrix} (2.0 \times 1.0) + (0.0 \times 0.5) \\ (1.0 \times 1.0) + (-1.0 \times 0.5) \end{bmatrix}",
            color=BLACK
        ).next_to(query_calc_expand, DOWN, aligned_edge=LEFT)

        with self.voiceover(text="""
            For the first element of our query vector, we multiply the first row of W sub Q by X sub cat.
            That is 2.0 times 1.0, plus 0.0 times 0.5.
            For the second element, we take the second row.
            That is 1.0 times 1.0, plus negative 1.0 times 0.5.
        """) as tracker:
            self.play(Write(query_result))
            self.wait(2)

        query_final = MathTex(
            r"= \begin{bmatrix} 2.0 + 0.0 \\ 1.0 - 0.5 \end{bmatrix} = \begin{bmatrix} 2.0 \\ 0.5 \end{bmatrix}",
            color=GREEN
        ).next_to(query_result, DOWN, aligned_edge=LEFT)

        out_arrow = Arrow(query_final.get_right(), query_final.get_right() + RIGHT*1.5, buff=0.1, color=BLACK)

        with self.voiceover(text="""
            Doing the arithmetic, we get 2.0 plus 0.0 for the top element, which is 2.0.
            And 1.0 minus 0.5 for the bottom element, which is 0.5.
            So, our final query vector for the token 'cat' is the vector containing 2.0 and 0.5.
            We will highlight this resulting query vector in green, as it is the output of our forward pass for this step,
            flowing out to the next stage.
        """) as tracker:
            self.play(Write(query_final))
            self.play(Write(out_arrow))
            self.wait(2)

        # Transition to Key and Value
        with self.voiceover(text="""
            We have now successfully projected our token into the Query space.
            The process for computing the Key and Value vectors is exactly the same,
            but using different, independent learned weight matrices.
            Let's clear the detailed calculation and look at the computations for Keys and Values.
        """) as tracker:
            self.play(
                FadeOut(wq_matrix),
                FadeOut(fwd_arrow),
                FadeOut(query_calc_eq),
                FadeOut(query_calc_expand),
                FadeOut(query_result),
                FadeOut(out_arrow),
                query_final.animate.to_edge(UP).shift(RIGHT*4)
            )
            self.wait(2)

        q_label = MathTex(r"q_{\text{cat}}", color=GREEN).next_to(query_final, LEFT)
        self.play(Write(q_label))

        fwd_arrow_k = Arrow(cat_embedding_text.get_bottom(), cat_embedding_text.get_bottom() + DOWN * 1.5, buff=0.1, color=BLACK)
        self.play(Write(fwd_arrow_k))

        wk_matrix = MathTex(
            r"W_K = \begin{bmatrix} 0.5 & 1.0 \\ 0.0 & 2.0 \end{bmatrix}",
            color=PURPLE
        ).next_to(fwd_arrow_k, DOWN, buff=0.1, aligned_edge=LEFT)

        key_eq = MathTex(
            r"k_{\text{cat}} = W_K X_{\text{cat}}",
            color=BLACK
        ).next_to(wk_matrix, RIGHT, buff=1)

        key_final = MathTex(
            r"= \begin{bmatrix} (0.5)(1.0) + (1.0)(0.5) \\ (0.0)(1.0) + (2.0)(0.5) \end{bmatrix} = \begin{bmatrix} 1.0 \\ 1.0 \end{bmatrix}",
            color=GREEN
        ).next_to(key_eq, RIGHT, buff=0.5)

        with self.voiceover(text="""
            The Key vector represents what the token 'cat' contains, its identity.
            Other tokens will compare their queries against this key to see if 'cat' is relevant to them.
            We use a different matrix, W sub K. Let's say W sub K is 0.5, 1.0 in the first row, and 0.0, 2.0 in the second.
            Multiplying W sub K by X sub cat, we calculate 0.5 times 1.0 plus 1.0 times 0.5, which is 1.0 for the top element.
            For the bottom element, 0.0 times 1.0 plus 2.0 times 0.5, which is also 1.0.
            Our Key vector k sub cat is 1.0 and 1.0.
        """) as tracker:
            self.play(Write(wk_matrix))
            self.wait(1)
            self.play(Write(key_eq))
            self.wait(1)
            self.play(Write(key_final))
            self.wait(2)

        wv_matrix = MathTex(
            r"W_V = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}",
            color=PURPLE
        ).next_to(wk_matrix, DOWN, buff=1.5, aligned_edge=LEFT)

        val_eq = MathTex(
            r"v_{\text{cat}} = W_V X_{\text{cat}}",
            color=BLACK
        ).next_to(wv_matrix, RIGHT, buff=1)

        val_final = MathTex(
            r"= \begin{bmatrix} (1.0)(1.0) + (0.0)(0.5) \\ (0.0)(1.0) + (1.0)(0.5) \end{bmatrix} = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}",
            color=GREEN
        ).next_to(val_eq, RIGHT, buff=0.5)

        fwd_arrow_v = Arrow(wk_matrix.get_bottom(), wk_matrix.get_bottom() + DOWN * 1.5, buff=0.1, color=BLACK)
        self.play(Write(fwd_arrow_v))

        with self.voiceover(text="""
            Finally, the Value vector represents the actual information the token 'cat' will contribute to the sequence
            if another token decides it is relevant based on the query-key interaction.
            We use the Value weight matrix, W sub V. Here we'll use the identity matrix for simplicity.
            Multiplying the identity matrix by X sub cat simply gives us back X sub cat.
            So our value vector v sub cat is 1.0 and 0.5.
        """) as tracker:
            self.play(wv_matrix.animate.next_to(fwd_arrow_v, DOWN, buff=0.1, aligned_edge=LEFT))
            self.wait(1)
            self.play(val_eq.animate.next_to(wv_matrix, RIGHT, buff=1))
            self.play(val_final.animate.next_to(val_eq, RIGHT, buff=0.5))
            self.wait(2)

        with self.voiceover(text="""
            To summarize the forward pass for this stage: a single token embedding comes in,
            and by multiplying it with three distinct learned weight matrices—W sub Q, W sub K, and W sub V—
            we project it into three distinct representational spaces.
            This gives us the query, key, and value vectors that will then be used in the scaled dot-product attention mechanism.
            Notice that all of these calculations are independent and can be done in parallel for every token in the sequence.
        """) as tracker:
            self.wait(3)

        # Clear to show Backprop
        with self.voiceover(text="""
            Now, let's explore the reverse direction. During training, how do these weight matrices actually learn?
            This brings us to backpropagation. Let's clear the screen and look at how gradients flow backwards
            through these exact same operations.
        """) as tracker:
            self.play(
                FadeOut(cat_embedding_text), FadeOut(q_label), FadeOut(query_final),
                FadeOut(wk_matrix), FadeOut(key_eq), FadeOut(key_final), FadeOut(fwd_arrow_k), FadeOut(fwd_arrow_v),
                FadeOut(wv_matrix), FadeOut(val_eq), FadeOut(val_final)
            )
            self.wait(2)

        # Backpropagation Section
        with self.voiceover(text="""
            In the forward pass, we computed q equals W sub Q times X.
            During backprop, the gradient flows back... from the subsequent layers, representing the error.
            Let's call the loss function L. The gradient of the loss with respect to the query vector is denoted as
            partial L over partial q.
            Our goal is to figure out how to update the weight matrix W sub Q, and also what gradient to pass back down to X.
        """) as tracker:
            self.wait(2)

        fwd_eq_again = MathTex(
            r"\text{Forward: } q = W_Q X",
            color=BLACK
        ).to_edge(UP).shift(DOWN*0.5)

        backprop_arrow = DashedLine(fwd_eq_again.get_bottom() + DOWN*2, fwd_eq_again.get_bottom(), buff=0.1, color=RED).add_tip()

        incoming_grad = MathTex(
            r"\text{Incoming Gradient: } \frac{\partial L}{\partial q} = \begin{bmatrix} \delta_{q1} \\ \delta_{q2} \end{bmatrix}",
            color=RED
        ).next_to(fwd_eq_again, DOWN, buff=1).shift(RIGHT * 3)

        with self.voiceover(text="""
            Here is our forward equation again, q equals W sub Q times X.
            And here is our dashed red arrow showing the incoming gradient flowing back from the attention softmax layer, which we will call partial L over partial q.
            It's a vector of the same dimension as q, containing error signals delta q1 and delta q2.
            To update W sub Q, we need the gradient of the loss with respect to W sub Q.
            We use the chain rule of calculus for matrices.
        """) as tracker:
            self.play(Write(fwd_eq_again))
            self.wait(1)
            self.play(Write(backprop_arrow))
            self.play(Write(incoming_grad))
            self.wait(2)

        grad_wq = MathTex(
            r"\text{Gradient for } W_Q: \quad \frac{\partial L}{\partial W_Q} = \frac{\partial L}{\partial q} \times X^T",
            color=RED
        ).next_to(backprop_arrow, DOWN, buff=1)

        grad_wq_expand = MathTex(
            r"= \begin{bmatrix} \delta_{q1} \\ \delta_{q2} \end{bmatrix} \begin{bmatrix} X_1 & X_2 \end{bmatrix}",
            color=RED
        ).next_to(grad_wq, DOWN, aligned_edge=LEFT)

        grad_wq_final = MathTex(
            r"= \begin{bmatrix} \delta_{q1} X_1 & \delta_{q1} X_2 \\ \delta_{q2} X_1 & \delta_{q2} X_2 \end{bmatrix}",
            color=RED
        ).next_to(grad_wq_expand, DOWN, aligned_edge=LEFT)

        with self.voiceover(text="""
            The gradient for the weight matrix W sub Q is the outer product of the incoming gradient vector
            and the transpose of the input vector X.
            Let's write that out. We have our column vector of deltas multiplied by the row vector of X inputs.
            This outer product results in a matrix of exactly the same shape as W sub Q.
            Every element in the weight matrix gets an update that is proportional to the error signal for its output dimension,
            multiplied by the activation of its corresponding input dimension.
            This gradient will then be used by an optimizer, like Adam, to adjust W sub Q.
        """) as tracker:
            self.play(Write(grad_wq))
            self.wait(1)
            self.play(Write(grad_wq_expand))
            self.wait(1)
            self.play(Write(grad_wq_final))
            self.wait(2)

        with self.voiceover(text="""
            But we are not done yet. We also need to compute the gradient with respect to the input X,
            so we can pass the error signal further backward down the network, to the preceding layers.
        """) as tracker:
            self.play(
                FadeOut(grad_wq),
                FadeOut(grad_wq_expand),
                FadeOut(grad_wq_final)
            )
            self.wait(2)

        grad_x = MathTex(
            r"\text{Gradient for } X: \quad \frac{\partial L}{\partial X} = W_Q^T \times \frac{\partial L}{\partial q}",
            color=RED
        ).next_to(incoming_grad, DOWN, buff=1)

        grad_x_explain = Text(
            "The error is projected backward using the transpose of the weights.",
            color=BLACK, font_size=24
        ).next_to(grad_x, DOWN, buff=0.5)

        with self.voiceover(text="""
            The gradient for the input vector X is computed by multiplying the transpose of the weight matrix W sub Q
            by the incoming gradient vector.
            Essentially, the error signal is projected backward through the transpose of the weights.
            It's a beautiful symmetry: the forward pass multiplies X by W, and the backward pass multiplies the error by W transpose.
        """) as tracker:
            self.play(Write(grad_x))
            self.wait(1)
            self.play(Write(grad_x_explain))
            self.wait(2)

        with self.voiceover(text="""
            Of course, X doesn't just contribute to the query vector.
            It is also multiplied by W sub K to create the key, and W sub V to create the value.
            Therefore, the total gradient flowing back to X is the sum of the gradients from all three of these pathways.
        """) as tracker:
            self.wait(1)

        total_grad_x = MathTex(
            r"\frac{\partial L}{\partial X}_{\text{total}} = \left(W_Q^T \frac{\partial L}{\partial q}\right) + \left(W_K^T \frac{\partial L}{\partial k}\right) + \left(W_V^T \frac{\partial L}{\partial v}\right)",
            color=RED
        ).next_to(grad_x_explain, DOWN, buff=1)

        with self.voiceover(text="""
            The total gradient for X is the sum of the gradient from the query path, the gradient from the key path,
            and the gradient from the value path.
            This accumulated gradient is what gets passed down to the layer below, which might be another attention block,
            a feed-forward network, or the embedding layer itself.
            This demonstrates how the error signal disperses through multiple parallel branches of computation,
            and then recombines when flowing backwards.
        """) as tracker:
            self.play(Write(total_grad_x))
            self.wait(3)

        # 5. Key Interview Insight
        with self.voiceover(text="""
            Before we conclude, let's highlight a very common interview "gotcha" related to this topic.
            Interviewers love to ask questions that test if you understand the dimensions and computational complexity of this specific step.
        """) as tracker:
            self.play(
                FadeOut(fwd_eq_again), FadeOut(incoming_grad),
                FadeOut(grad_x), FadeOut(grad_x_explain), FadeOut(total_grad_x)
            )
            self.wait(2)

        insight_box = Rectangle(width=10, height=4, color=RED, fill_color=WHITE, fill_opacity=1)
        insight_title = Text("Key Interview Insight", color=RED).to_edge(UP, buff=1.5)

        insight_text = VGroup(
            Text("Gotcha: Are Q, K, and V matrices calculated sequentially?", color=BLACK, font_size=28),
            Text("Answer: NO. In practice, W_Q, W_K, and W_V are often", color=BLACK, font_size=28),
            Text("concatenated into a single large weight matrix.", color=BLACK, font_size=28),
            MathTex(r"W_{QKV} = [W_Q, W_K, W_V]", color=PURPLE),
            Text("This allows a single matrix multiplication to compute", color=BLACK, font_size=28),
            Text("Q, K, and V simultaneously, vastly improving GPU utilization.", color=BLACK, font_size=28)
        ).arrange(DOWN, buff=0.3).move_to(insight_box.get_center())

        with self.voiceover(text="""
            A classic interview question is: Are the Query, Key, and Value matrices calculated sequentially, one after another?
            The answer is firmly no.
            While conceptually we think of them as three separate steps—and we animated them that way for clarity—
            in actual production implementations like PyTorch or vLLM, they are not separate.
            The weight matrices W sub Q, W sub K, and W sub V are almost always concatenated along the output dimension
            into a single, large weight matrix, often referred to as W sub Q K V.
        """) as tracker:
            self.play(Write(insight_box), Write(insight_title))
            self.wait(1)
            self.play(Write(insight_text[0:3]))
            self.wait(2)

        with self.voiceover(text="""
            By concatenating them, we can perform a single, massive matrix multiplication for all three projections at once.
            This allows a single matrix multiplication to compute Q, K, and V simultaneously.
            Why do we do this? GPUs are incredibly efficient at large matrix multiplications, but they suffer from kernel launch overhead
            if you give them many small operations sequentially.
            Fusing operations maximizes GPU utilization and memory bandwidth, which is critical for high-performance AI engineering.
            Understanding this bridge between mathematical theory and hardware-level execution is what separates
            a good candidate from a great one.
        """) as tracker:
            self.play(Write(insight_text[3:]))
            self.wait(3)

        with self.voiceover(text="""
            That concludes our deep dive into the computation of Queries, Keys, and Values.
            We have traced a token through its forward projection using concrete numbers,
            derived the backpropagation gradients step by step,
            and uncovered a crucial optimization used in modern LLM inference engines.
            Take some time to review these equations, and we will see you tomorrow for the next step in the Transformer architecture.
        """) as tracker:
            self.wait(4)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(1)
