from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class TransformerTrainingScene(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Scene 1: Introduction and Token Setup
        with self.voiceover(text="Welcome back to our deep dive on Transformer Training. Today we focus on the forward pass, loss calculation, and backpropagation. Let's imagine we are training a language model, and our input sequence is 'The cat sat'. We start with three tokens representing this sequence. We'll trace the flow of a single token through a transformer layer. Transformers have revolutionized natural language processing by processing all tokens in a sequence simultaneously rather than sequentially. This deep dive will illuminate the exact mathematical operations that power this architecture, from initial embeddings to the final gradient updates that allow the model to learn.") as tracker:
            title = Text("Transformer Training: Forward Pass & Backprop", color=BLACK).scale(0.8)
            title.to_edge(UP)
            self.play(Write(title))
            self.wait(2)

            # Setup tokens
            tokens_group = VGroup()
            words = ["The", "cat", "sat"]
            for word in words:
                box = Rectangle(width=1.5, height=1, color=BLACK)
                t = Text(word, color=BLACK).scale(0.8)
                t.move_to(box.get_center())
                tokens_group.add(VGroup(box, t))

            tokens_group.arrange(RIGHT, buff=0.5).move_to(ORIGIN)
            self.play(FadeIn(tokens_group))
            self.wait(2)

        with self.voiceover(text="Throughout this journey, we'll focus specifically on the word 'cat' to trace exactly what happens to it. Let's highlight the active token in blue. We'll follow this token as it gets transformed through the self-attention mechanism and feed-forward networks, experiencing matrix multiplications, loss calculation, and finally, backpropagation, which is the heart of training. By focusing on a single token, we can demystify the complex tensor operations happening under the hood.") as tracker:
            active_token = tokens_group[1]
            active_box_copy = Rectangle(width=1.5, height=1, color=BLUE, fill_color=BLUE, fill_opacity=0.3)
            active_box_copy.move_to(active_token[0].get_center())
            self.play(Transform(active_token[0], active_box_copy))

            # Move tokens up to make space for next steps
            self.play(
                tokens_group.animate.scale(0.7).to_edge(UL, buff=0.5),
                FadeOut(title)
            )
            self.wait(2)

        # Scene 2: Input Embedding and Linear Projections
        with self.voiceover(text="Before attention, each token is converted into an embedding vector. For our token 'cat', let's say its embedding is a 3-dimensional vector x. In a real model, this vector might have hundreds or thousands of dimensions, but for our visual example, we'll use a simple 3D vector. This input vector arrives at the self-attention layer. To compute attention, this vector must be projected into three distinct spaces: Query, Key, and Value. We do this by multiplying the input vector by three learned weight matrices: W_Q, W_K, and W_V.") as tracker:
            x_label = MathTex("x_{\\text{cat}} =", color=BLACK).scale(0.8)
            x_vec = Matrix([["1.0"], ["0.5"], ["0.2"]]).scale(0.7)
            x_vec.get_entries().set_color(BLACK)
            x_vec.get_brackets().set_color(BLACK)
            x_group = VGroup(x_label, x_vec).arrange(RIGHT, buff=0.2)
            x_group.next_to(tokens_group, DOWN, buff=1.0).align_to(tokens_group, LEFT)

            self.play(Write(x_group))
            self.wait(2)

            arrow_in = Arrow(start=tokens_group[1].get_bottom(), end=x_group.get_top(), color=BLACK)
            self.play(FadeIn(arrow_in))
            self.wait(1)

        with self.voiceover(text="Let's look at the Query projection. Here, the token 'cat' represented by vector x, is multiplied by the Query weight matrix W_Q to produce the query vector q. The query represents what this specific token is looking for in other parts of the sequence. Notice how each element of the resulting vector is a dot product of the input vector and a row of the weight matrix. Let's say W_Q is a 3x3 matrix.") as tracker:
            wq_label = MathTex("W_Q =", color=PURPLE).scale(0.8)
            wq_mat = Matrix([["0.1", "0.2", "0.3"], ["0.4", "0.5", "0.6"], ["0.7", "0.8", "0.9"]]).scale(0.7)
            wq_mat.get_entries().set_color(PURPLE)
            wq_mat.get_brackets().set_color(PURPLE)
            wq_group = VGroup(wq_label, wq_mat).arrange(RIGHT, buff=0.2)
            wq_group.move_to(ORIGIN)

            q_calc = MathTex("q = W_Q \\cdot x_{\\text{cat}}", color=BLACK).scale(0.8)
            q_calc.next_to(wq_group, UP, buff=0.5)

            self.play(Write(q_calc), Write(wq_group))
            self.wait(2)

        with self.voiceover(text="Performing this multiplication step-by-step. The first element of q is calculated as 0.1 times 1.0, plus 0.2 times 0.5, plus 0.3 times 0.2, which equals 0.26. The second element is 0.4 times 1.0 plus 0.5 times 0.5 plus 0.6 times 0.2 equals 0.77. The third is 0.7 times 1.0 plus 0.8 times 0.5 plus 0.9 times 0.2 equals 1.28. This explicit calculation gives us our final query vector q. These dense mathematical operations happen billions of times per second during training.") as tracker:
            q_res_label = MathTex("q =", color=BLACK).scale(0.8)
            q_res_mat = Matrix([["0.26"], ["0.77"], ["1.28"]]).scale(0.7)
            q_res_mat.get_entries().set_color(BLACK)
            q_res_mat.get_brackets().set_color(BLACK)
            q_res_group = VGroup(q_res_label, q_res_mat).arrange(RIGHT, buff=0.2)
            q_res_group.next_to(wq_group, RIGHT, buff=1.0)

            copy_x = x_group.copy()
            self.play(Transform(copy_x, q_res_group), FadeIn(q_res_group))
            self.wait(2)

        with self.voiceover(text="A similar process happens simultaneously to compute the Key vector k and Value vector v, using their respective weight matrices W_K and W_V. Keys represent what a token contains, and Values hold the actual information to be aggregated. These weight matrices are the foundational parameters that our neural network is updating during the training process.") as tracker:
            kv_text = MathTex("k = W_K \\cdot x, \\quad v = W_V \\cdot x", color=BLACK).scale(0.8)
            kv_text.next_to(q_res_group, DOWN, buff=0.5)
            self.play(Write(kv_text))
            self.wait(2)

            self.play(FadeOut(VGroup(wq_group, q_calc, q_res_group, kv_text, arrow_in, x_group, copy_x)))
            self.wait(1)

        # Scene 3: Self-Attention Calculation
        with self.voiceover(text="Next in the forward pass, we calculate the attention scores. We take the query vector of 'cat' and compute its dot product with the key vectors of all tokens in the sequence, including itself. This dot product acts as a similarity metric, telling us how much focus 'cat' should place on 'The', 'cat', and 'sat'. Let's say these raw alignment scores evaluate to 2.0, 5.0, and 1.5 respectively.") as tracker:
            scores_label = MathTex("\\text{Raw Scores} = q_{\\text{cat}} \\cdot K^T =", color=BLACK).scale(0.8)
            scores_mat = Matrix([["2.0", "5.0", "1.5"]]).scale(0.7)
            scores_mat.get_entries().set_color(BLACK)
            scores_mat.get_brackets().set_color(BLACK)
            scores_group = VGroup(scores_label, scores_mat).arrange(RIGHT, buff=0.2)
            scores_group.move_to(UP * 1.5)

            self.play(Write(scores_group))
            self.wait(2)

        with self.voiceover(text="To convert these raw scores into a clean probability distribution that sums to one, we apply the Softmax function. In practice, we first scale the scores down by the square root of the key dimension to prevent the softmax gradients from vanishing, a crucial step for stable training. The Softmax function then exponentiates each scaled score and divides by the sum of all exponentials across the sequence length.") as tracker:
            softmax_eq = MathTex("\\text{Softmax}(x_i) = \\frac{e^{x_i}}{\\sum e^{x_j}}", color=BLACK).scale(0.8)
            softmax_eq.next_to(scores_group, DOWN, buff=0.5)
            self.play(Write(softmax_eq))
            self.wait(2)

        with self.voiceover(text="Applying softmax to our raw scores 2.0, 5.0, and 1.5 yields our final normalized attention weights. Because of the exponential nature of the softmax function, the relatively high score of 5.0 completely dominates the distribution. The resulting probabilities calculate out to roughly 0.04, 0.93, and 0.03. Notice how almost all attention for 'cat' is placed on its own token in this specific example, which is common in early layers.") as tracker:
            attn_label = MathTex("\\text{Attention Weights} =", color=BLACK).scale(0.8)
            attn_mat = Matrix([["0.04", "0.93", "0.03"]]).scale(0.7)
            attn_mat.get_entries().set_color(BLACK)
            attn_mat.get_brackets().set_color(BLACK)
            attn_group = VGroup(attn_label, attn_mat).arrange(RIGHT, buff=0.2)
            attn_group.next_to(softmax_eq, DOWN, buff=0.5)

            self.play(Write(attn_group))
            self.wait(2)

        with self.voiceover(text="Finally, we multiply these attention weights by the corresponding Value vectors of all tokens and sum them up. This weighted sum produces the context-aware output vector z for 'cat'. This vector now encodes not just the word 'cat', but its contextual relationship with the surrounding words. In the forward pass, this output flows directly to the next stage. We'll show the forward pass completing in green.") as tracker:
            out_calc = MathTex("z_{\\text{cat}} = \\sum \\text{weights}_i \\cdot v_i", color=BLACK).scale(0.8)
            out_calc.next_to(attn_group, DOWN, buff=0.5)
            self.play(Write(out_calc))

            final_out_box = Rectangle(width=2, height=1, color=GREEN)
            final_out_text = Text("Output z", color=GREEN).scale(0.6)
            final_out_group = VGroup(final_out_box, final_out_text).move_to(out_calc.get_center())
            final_out_text.move_to(final_out_box.get_center())

            arrow_out = Arrow(start=out_calc.get_bottom(), end=out_calc.get_bottom() + DOWN * 1.5, color=GREEN)

            self.play(Transform(out_calc, final_out_group), FadeIn(arrow_out))
            self.wait(2)

            self.play(FadeOut(VGroup(scores_group, softmax_eq, attn_group, final_out_group, arrow_out, out_calc)))
            self.wait(1)

        # Scene 4: Loss and Backpropagation Overview
        with self.voiceover(text="In the forward pass... the signal travels all the way through the multi-layer perceptron feed-forward networks, experiencing non-linear activations, before eventually generating a final logits prediction over the vocabulary. This prediction is compared against the true target token using a loss function, typically Cross-Entropy Loss for language modeling. Let's denote this final computed scalar loss value simply as L.") as tracker:
            ffn_box = Rectangle(width=3, height=1.5, color=BLACK)
            ffn_text = Text("Feed-Forward Net & Output", color=BLACK).scale(0.5)
            ffn_text.move_to(ffn_box.get_center())
            ffn_group = VGroup(ffn_box, ffn_text).move_to(UP * 1)

            loss_box = Rectangle(width=2, height=1, color=RED)
            loss_text = MathTex("Loss = L", color=RED).scale(0.8)
            loss_text.move_to(loss_box.get_center())
            loss_group = VGroup(loss_box, loss_text).next_to(ffn_group, DOWN, buff=1.0)

            arrow_fwd = Arrow(start=ffn_group.get_bottom(), end=loss_group.get_top(), color=GREEN)

            self.play(FadeIn(ffn_group), FadeIn(arrow_fwd), FadeIn(loss_group))
            self.wait(2)

        with self.voiceover(text="During backprop, the gradient flows backwards from the loss. We must compute the derivative of the loss with respect to every single parameter in our massive model. This error signal propagates backward step-by-step through the network using the fundamental chain rule of calculus. Let's represent this backward flow of gradients with a dashed red arrow pointing backwards through the layers.") as tracker:
            arrow_back = DashedLine(start=loss_group.get_left(), end=ffn_group.get_left() + LEFT * 0.5, color=RED).add_tip()

            self.play(FadeIn(arrow_back))
            self.wait(2)

            self.play(FadeOut(VGroup(ffn_group, loss_group, arrow_fwd, arrow_back)))
            self.wait(1)

        # Scene 5: Backpropagation Deep Dive (Math)
        with self.voiceover(text="Let's dive deep into the math of backpropagation for our original Query weight matrix W_Q. To properly update W_Q, we need the full gradient of the Loss L with respect to W_Q. According to the chain rule, this is the derivative of the Loss with respect to the intermediate query vector q, multiplied by the local derivative of q with respect to W_Q.") as tracker:
            chain_rule = MathTex("\\frac{\\partial L}{\\partial W_Q} = \\frac{\\partial L}{\\partial q} \\cdot \\frac{\\partial q}{\\partial W_Q}", color=BLACK)
            chain_rule.move_to(UP * 1.5)
            self.play(Write(chain_rule))
            self.wait(2)

        with self.voiceover(text="We know from our earlier explicit calculation that the query q is just the linear product of W_Q and the input embedding vector x. Therefore, the local derivative, the partial of q with respect to W_Q, evaluates simply to the input vector x transposed. This is a fundamental property of matrix calculus that appears constantly when calculating neural network gradients.") as tracker:
            q_eq = MathTex("q = W_Q \\cdot x \\implies \\frac{\\partial q}{\\partial W_Q} = x^T", color=BLACK)
            q_eq.next_to(chain_rule, DOWN, buff=0.8)
            self.play(Write(q_eq))
            self.wait(2)

        with self.voiceover(text="Substituting this local derivative back into our chain rule equation, we obtain the final gradient matrix for W_Q. It is the outer product of the incoming error gradient, partial L over partial q, and the input vector x transposed. This resulting red matrix tells us exactly the direction and magnitude to adjust every single parameter inside W_Q in order to minimize our training loss.") as tracker:
            grad_wq = MathTex("\\frac{\\partial L}{\\partial W_Q} = \\frac{\\partial L}{\\partial q} \\otimes x^T", color=RED)
            grad_wq.next_to(q_eq, DOWN, buff=0.8)
            self.play(Write(grad_wq))
            self.wait(2)

            # Highlight weight matrices that get updated
            update_text = Text("Weights Updated: W_Q, W_K, W_V, W_O, FFN weights", color=PURPLE).scale(0.5)
            update_text.to_edge(DOWN)
            self.play(Write(update_text))
            self.wait(2)

            self.play(FadeOut(VGroup(chain_rule, q_eq, grad_wq, update_text)))
            self.wait(1)

        # Scene 6: Parameter Update
        with self.voiceover(text="Once we have isolated the gradient for all matrices, we actually update the weights using an optimizer, typically Adam or AdamW in modern Transformers. However, for a simple stochastic gradient descent step, the new W_Q is simply the old W_Q minus the learning rate alpha times the calculated gradient. We highlight the newly updated, better-performing weights in green.") as tracker:
            update_eq = MathTex("W_Q^{(new)} = W_Q^{(old)} - \\alpha \\frac{\\partial L}{\\partial W_Q}", color=BLACK)
            update_eq.move_to(ORIGIN)
            self.play(Write(update_eq))
            self.wait(2)

            # Show changed cells in GREEN
            updated_mat = Matrix([["0.09", "0.21", "0.29"], ["0.38", "0.49", "0.62"], ["0.71", "0.78", "0.88"]]).scale(0.7)
            updated_mat.get_entries().set_color(GREEN)
            updated_mat.get_brackets().set_color(BLACK)
            updated_mat.next_to(update_eq, DOWN, buff=1.0)

            self.play(FadeIn(updated_mat))
            self.wait(2)

            self.play(FadeOut(VGroup(update_eq, updated_mat)))
            self.wait(1)

        # Scene 7: Interview Gotcha Callout Box
        with self.voiceover(text="To wrap up, here is a key interview gotcha for this topic. Interviewers often ask about the complexity of the attention mechanism. The most common pitfall is forgetting that while the forward pass computes scaling quadratically with sequence length due to the N by N attention matrix, the actual memory bottleneck during backprop is often storing the massive intermediate activations for the huge feed-forward layers, not just the attention maps. Always remember to holistically discuss both compute and memory scaling bottlenecks when discussing transformer training.") as tracker:
            callout_box = Rectangle(width=10, height=3, color=RED)
            callout_title = Text("Key Interview Gotcha", color=RED).scale(0.8)
            callout_title.next_to(callout_box.get_top(), DOWN, buff=0.2)

            gotcha_text = Text(
                "Don't just say 'Attention is O(N^2)'.\n"
                "Remember that during Backprop, storing\n"
                "activations for the Feed-Forward Network\n"
                "(FFN) often dominates memory usage!",
                color=BLACK,
                t2c={"O(N^2)": BLUE, "Backprop": RED, "Feed-Forward Network": PURPLE}
            ).scale(0.6)
            gotcha_text.move_to(callout_box.get_center()).shift(DOWN * 0.2)

            callout_group = VGroup(callout_box, callout_title, gotcha_text)
            self.play(FadeIn(callout_group))
            self.wait(4)

        with self.voiceover(text="Understanding this precise data flow through the forward pass, how the final loss is calculated, and exactly how gradients flow back step-by-step to intelligently update the query, key, and value weights is crucial for mastering AI engineering and passing senior technical interviews. Thank you for watching this comprehensive deep dive on transformer training.") as tracker:
            self.play(FadeOut(callout_group), FadeOut(tokens_group))
            final_text = Text("Transformer Training Mastered", color=BLACK)
            self.play(Write(final_text))
            self.wait(2)
            self.play(FadeOut(final_text))
