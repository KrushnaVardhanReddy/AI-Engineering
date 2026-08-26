from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class FeedForwardLayerScene(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Introduction
        with self.voiceover(text="Welcome to this deep dive into the Feed-Forward Layer inside the Transformer block. This is a crucial component that adds non-linearity and capacity to the model.") as tracker:
            title = Text("Feed-Forward Layer", color=BLACK).scale(1.2).to_edge(UP)
            self.play(FadeIn(title))
            self.wait(1)

        with self.voiceover(text="After the self-attention mechanism mixes information across different tokens, the Feed-Forward Network processes each token's representation completely independently.") as tracker:
            self.wait(1)

        # Input Setup
        with self.voiceover(text="Let's start with an example sequence of three tokens: 'The', 'cat', and 'sat'.") as tracker:
            tokens = VGroup(
                Text("The", color=BLACK),
                Text("cat", color=BLUE),
                Text("sat", color=BLACK)
            ).arrange(RIGHT, buff=1.0)

            boxes = VGroup(*[
                SurroundingRectangle(t, color=BLACK, fill_color=WHITE, fill_opacity=1, buff=0.3)
                for t in tokens
            ])

            # Change box color for "cat" to BLUE
            boxes[1].set_color(BLUE)

            token_group = VGroup(*[VGroup(b, t) for b, t in zip(boxes, tokens)])
            token_group.next_to(title, DOWN, buff=1)

            self.play(FadeIn(token_group))
            self.wait(1)

        with self.voiceover(text="We will trace the exact data flow of a single token. Here, we highlight the token 'cat' in blue. Let's see how its vector representation travels through the Feed-Forward Layer stage.") as tracker:
            self.play(token_group[1].animate.scale(1.2))
            self.play(token_group[1].animate.scale(1/1.2))
            self.wait(2)

        with self.voiceover(text="Let's isolate the token 'cat' and its corresponding vector. The other tokens are processed in exactly the same way, in parallel, but there is no cross-token interaction within this layer.") as tracker:
            self.play(FadeOut(VGroup(token_group[0], token_group[2])))
            self.play(token_group[1].animate.move_to(LEFT * 4 + UP * 1))
            self.wait(2)

        # Data Flow - Step 1: Input Vector
        with self.voiceover(text="We represent our token 'cat' as a 2-dimensional column vector for simplicity. In reality, this dimension, often called d_model, might be 512 or larger.") as tracker:
            x_label = MathTex(r"x = ", color=BLACK).next_to(token_group[1], DOWN, buff=0.5)
            x_vector = Matrix([["2.0"], ["-1.0"]]).set_color(BLACK).next_to(x_label, RIGHT)
            x_group = VGroup(x_label, x_vector)

            self.play(Write(x_group))
            self.wait(2)

        with self.voiceover(text="The Feed-Forward Layer consists of two linear transformations with a non-linear activation function in between. Let's look at the first linear transformation.") as tracker:
            self.wait(1)

        # Data Flow - Step 2: W_1 Multiplication
        with self.voiceover(text="First, our input vector x is multiplied by a weight matrix W_1. This matrix typically expands the dimensionality of our vector by a factor of 4. We will expand our 2D vector into a 3D vector for this demonstration.") as tracker:
            w1_label = MathTex(r"W_1 = ", color=PURPLE).move_to(RIGHT * 1 + UP * 1)
            w1_matrix = Matrix([
                ["1.0", "0.5"],
                ["-1.0", "2.0"],
                ["0.0", "-1.0"]
            ]).set_color(PURPLE).next_to(w1_label, RIGHT)
            w1_group = VGroup(w1_label, w1_matrix)

            self.play(FadeIn(w1_group))
            self.wait(2)

        with self.voiceover(text="We compute the intermediate hidden representation, often called h. This is equal to W_1 multiplied by x, plus a bias vector b_1, which we will assume is zero here.") as tracker:
            calc_1 = MathTex(r"h = W_1 x", color=BLACK).move_to(RIGHT * 2 + DOWN * 2)
            self.play(Write(calc_1))
            self.wait(2)

        with self.voiceover(text="Let's walk through the matrix multiplication step by step. We take the dot product of each row of W_1 with our input vector x.") as tracker:
            self.wait(1)

        with self.voiceover(text="For the first dimension of h, we multiply 1.0 by 2.0, which is 2.0, and add 0.5 times -1.0, which is -0.5. The result is 1.5.") as tracker:
            h_matrix_raw = [["1.5"], ["-4.0"], ["1.0"]]
            h_matrix = Matrix(h_matrix_raw).set_color(GREEN).next_to(calc_1, RIGHT)

            h_rect_0 = SurroundingRectangle(h_matrix.get_entries()[0], color=GREEN)
            w1_rect_0 = SurroundingRectangle(w1_matrix.get_rows()[0], color=PURPLE)
            x_rect = SurroundingRectangle(x_vector.get_columns()[0], color=BLACK)

            self.play(FadeIn(w1_rect_0), FadeIn(x_rect))
            self.play(Write(h_matrix.get_brackets()), Write(h_matrix.get_entries()[0]))
            self.play(FadeIn(h_rect_0))
            self.wait(2)
            self.play(FadeOut(w1_rect_0), FadeOut(h_rect_0))

        with self.voiceover(text="For the second dimension, we multiply -1.0 by 2.0, yielding -2.0, and add 2.0 times -1.0, which is -2.0. The sum is -4.0.") as tracker:
            w1_rect_1 = SurroundingRectangle(w1_matrix.get_rows()[1], color=PURPLE)
            h_rect_1 = SurroundingRectangle(h_matrix.get_entries()[1], color=GREEN)

            self.play(FadeIn(w1_rect_1))
            self.play(Write(h_matrix.get_entries()[1]))
            self.play(FadeIn(h_rect_1))
            self.wait(2)
            self.play(FadeOut(w1_rect_1), FadeOut(h_rect_1))

        with self.voiceover(text="Finally, for the third dimension, 0.0 times 2.0 is 0.0, plus -1.0 times -1.0, which is 1.0.") as tracker:
            w1_rect_2 = SurroundingRectangle(w1_matrix.get_rows()[2], color=PURPLE)
            h_rect_2 = SurroundingRectangle(h_matrix.get_entries()[2], color=GREEN)

            self.play(FadeIn(w1_rect_2))
            self.play(Write(h_matrix.get_entries()[2]))
            self.play(FadeIn(h_rect_2))
            self.wait(2)
            self.play(FadeOut(w1_rect_2), FadeOut(h_rect_2), FadeOut(x_rect))

        with self.voiceover(text="Notice how the resulting hidden vector is highlighted in green. This represents the expanded intermediate state, capturing a richer set of features.") as tracker:
            self.wait(2)

        # Data Flow - Step 3: Activation Function (ReLU)
        with self.voiceover(text="The next step is applying a non-linear activation function. This is critical; without it, the two linear transformations would simply collapse into a single linear transformation, offering no additional expressivity.") as tracker:
            self.play(FadeOut(w1_group), FadeOut(x_group), FadeOut(calc_1))
            self.play(h_matrix.animate.move_to(LEFT * 3))
            self.wait(2)

        with self.voiceover(text="Traditionally, Transformers use the ReLU or GELU activation function. We will use ReLU, which stands for Rectified Linear Unit. It simply replaces all negative values with zero.") as tracker:
            relu_eq = MathTex(r"h_{act} = \max(0, h)", color=BLACK).move_to(UP * 2)
            self.play(Write(relu_eq))
            self.wait(2)

        with self.voiceover(text="Let's apply ReLU to our intermediate vector h. The value 1.5 is positive, so it remains 1.5.") as tracker:
            h_act_matrix = Matrix([["1.5"], ["0.0"], ["1.0"]]).set_color(GREEN).move_to(RIGHT * 3)
            arrow_relu = Arrow(h_matrix.get_right(), h_act_matrix.get_left(), color=BLACK)

            self.play(FadeIn(arrow_relu))
            self.play(Write(h_act_matrix.get_brackets()), Write(h_act_matrix.get_entries()[0]))
            self.wait(2)

        with self.voiceover(text="The second value, -4.0, is negative. The ReLU function replaces it with 0.0.") as tracker:
            self.play(Write(h_act_matrix.get_entries()[1]))
            h_act_rect = SurroundingRectangle(h_act_matrix.get_entries()[1], color=RED)
            self.play(FadeIn(h_act_rect))
            self.wait(2)
            self.play(FadeOut(h_act_rect))

        with self.voiceover(text="The third value, 1.0, is positive, so it stays exactly the same.") as tracker:
            self.play(Write(h_act_matrix.get_entries()[2]))
            self.wait(2)

        with self.voiceover(text="Now we have our activated hidden state, completing the first half of the Feed-Forward Network.") as tracker:
            self.wait(2)

        # Data Flow - Step 4: W_2 Multiplication
        with self.voiceover(text="Now, we apply the second linear transformation using weight matrix W_2. This projects our vector back to its original dimension, d_model.") as tracker:
            self.play(FadeOut(h_matrix), FadeOut(arrow_relu), FadeOut(relu_eq))
            self.play(h_act_matrix.animate.move_to(LEFT * 4))

            w2_label = MathTex(r"W_2 = ", color=PURPLE).move_to(UP * 2 + LEFT * 1)
            w2_matrix = Matrix([
                ["1.0", "0.0", "-1.0"],
                ["0.5", "1.0", "1.0"]
            ]).set_color(PURPLE).next_to(w2_label, RIGHT)
            w2_group = VGroup(w2_label, w2_matrix)

            self.play(FadeIn(w2_group))
            self.wait(2)

        with self.voiceover(text="We calculate the final output vector y by multiplying W_2 and our activated hidden state.") as tracker:
            calc_2 = MathTex(r"y = W_2 h_{act}", color=BLACK).move_to(DOWN * 2 + LEFT * 1)
            self.play(Write(calc_2))
            self.wait(2)

        with self.voiceover(text="Let's compute the dot product again. For the first dimension, we multiply 1.0 by 1.5, add 0.0 times 0.0, and add -1.0 times 1.0. The result is 0.5.") as tracker:
            y_matrix = Matrix([["0.5"], ["1.75"]]).set_color(GREEN).next_to(calc_2, RIGHT)

            h_act_rect_x = SurroundingRectangle(h_act_matrix.get_columns()[0], color=BLACK)
            w2_rect_0 = SurroundingRectangle(w2_matrix.get_rows()[0], color=PURPLE)

            self.play(FadeIn(h_act_rect_x), FadeIn(w2_rect_0))
            self.play(Write(y_matrix.get_brackets()), Write(y_matrix.get_entries()[0]))
            self.wait(2)
            self.play(FadeOut(w2_rect_0))

        with self.voiceover(text="For the second dimension, we multiply 0.5 by 1.5 which is 0.75, add 1.0 times 0.0, and add 1.0 times 1.0. The final result is 1.75.") as tracker:
            w2_rect_1 = SurroundingRectangle(w2_matrix.get_rows()[1], color=PURPLE)

            self.play(FadeIn(w2_rect_1))
            self.play(Write(y_matrix.get_entries()[1]))
            self.wait(2)
            self.play(FadeOut(w2_rect_1), FadeOut(h_act_rect_x))

        with self.voiceover(text="This output vector y is the final representation of the token 'cat' as it exits the Feed-Forward Layer, ready for a residual connection and layer normalization.") as tracker:
            y_rect = SurroundingRectangle(y_matrix, color=GREEN)
            self.play(FadeIn(y_rect))
            self.wait(2)
            self.play(FadeOut(y_rect))

        # Forward vs Backward Pass
        with self.voiceover(text="What we just saw is the forward pass. Data flows from the input token, through the weight matrices, to produce an output.") as tracker:
            self.play(FadeOut(w2_group), FadeOut(calc_2), FadeOut(h_act_matrix), FadeOut(y_matrix))

            ffn_box = Rectangle(width=4, height=3, color=BLACK, fill_color=WHITE, fill_opacity=1)
            ffn_text = Text("Feed-Forward Layer", color=BLACK).scale(0.6).move_to(ffn_box.get_center())
            ffn_group = VGroup(ffn_box, ffn_text)

            in_arrow = Arrow(LEFT * 4, ffn_box.get_left(), color=BLACK)
            out_arrow = Arrow(ffn_box.get_right(), RIGHT * 4, color=BLACK)

            in_label = MathTex("x", color=BLACK).next_to(in_arrow, UP)
            out_label = MathTex("y", color=BLACK).next_to(out_arrow, UP)

            self.play(FadeIn(ffn_group), FadeIn(in_arrow), FadeIn(in_label), FadeIn(out_arrow), FadeIn(out_label))
            self.wait(2)

        with self.voiceover(text="During the training phase, after a loss is calculated at the end of the network, we must perform backpropagation to update the weights W_1 and W_2.") as tracker:
            self.wait(1)

        with self.voiceover(text="In backpropagation, the gradient of the loss with respect to the output, denoted as partial L over partial y, flows backward into the layer.") as tracker:
            grad_out_arrow = DashedLine(RIGHT * 4 + DOWN * 0.5, ffn_box.get_right() + DOWN * 0.5, color=RED).add_tip()
            grad_out_label = MathTex(r"\frac{\partial L}{\partial y}", color=RED).next_to(grad_out_arrow, DOWN)

            self.play(Write(grad_out_arrow), Write(grad_out_label))
            self.wait(2)

        with self.voiceover(text="Using the chain rule, this gradient is multiplied by the transpose of W_2, and passes backward through the derivative of the ReLU function, creating the gradient for the intermediate state.") as tracker:
            chain_rule_1 = MathTex(r"\frac{\partial L}{\partial h_{act}} = W_2^T \frac{\partial L}{\partial y}", color=RED).move_to(UP * 2)
            self.play(Write(chain_rule_1))
            self.wait(2)

        with self.voiceover(text="Then, it is multiplied by the transpose of W_1 to continue flowing back down to the previous layers.") as tracker:
            grad_in_arrow = DashedLine(ffn_box.get_left() + DOWN * 0.5, LEFT * 4 + DOWN * 0.5, color=RED).add_tip()
            grad_in_label = MathTex(r"\frac{\partial L}{\partial x}", color=RED).next_to(grad_in_arrow, DOWN)
            self.play(Write(grad_in_arrow), Write(grad_in_label))
            self.wait(2)

        with self.voiceover(text="Simultaneously, we calculate the gradients with respect to the parameters W_1 and W_2 themselves, so we can update them using an optimizer like Adam.") as tracker:
            grad_w2 = MathTex(r"\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial y} h_{act}^T", color=PURPLE).move_to(DOWN * 2)
            self.play(Write(grad_w2))
            self.wait(2)

        with self.voiceover(text="These parameter updates are what allow the Feed-Forward Layer to learn complex representations over time.") as tracker:
            self.wait(2)

        # Key Interview Gotcha
        with self.voiceover(text="Now, let's discuss a crucial concept that frequently comes up in machine learning interviews regarding the Feed-Forward Layer.") as tracker:
            self.play(FadeOut(VGroup(ffn_group, in_arrow, out_arrow, in_label, out_label, grad_out_arrow, grad_out_label, chain_rule_1, grad_in_arrow, grad_in_label, grad_w2, token_group[1])))
            self.wait(1)

        with self.voiceover(text="The most common interview gotcha is about how the Feed-Forward Layer handles tokens compared to the Self-Attention layer.") as tracker:
            gotcha_box = Rectangle(width=10, height=4, color=RED, fill_color=WHITE, fill_opacity=1)
            gotcha_title = Text("Key Interview Gotcha!", color=RED, weight=BOLD).scale(0.8).next_to(gotcha_box.get_top(), DOWN, buff=0.2)

            gotcha_text_1 = Text("Self-Attention mixes information ACROSS tokens.", color=BLACK).scale(0.5).next_to(gotcha_title, DOWN, buff=0.4)
            gotcha_text_2 = Text("Feed-Forward Layer processes each token INDEPENDENTLY.", color=BLUE).scale(0.5).next_to(gotcha_text_1, DOWN, buff=0.3)
            gotcha_text_3 = Text("Why? It acts as a vast 'key-value' memory bank to store", color=BLACK).scale(0.5).next_to(gotcha_text_2, DOWN, buff=0.3)
            gotcha_text_4 = Text("learned knowledge, while attention routes that knowledge.", color=BLACK).scale(0.5).next_to(gotcha_text_3, DOWN, buff=0.1)

            gotcha_group = VGroup(gotcha_box, gotcha_title, gotcha_text_1, gotcha_text_2, gotcha_text_3, gotcha_text_4)

            self.play(FadeIn(gotcha_group))
            self.wait(3)

        with self.voiceover(text="Remember, Self-Attention allows tokens to look at each other. But the Feed-Forward Layer is a position-wise operation. It applies the exact same W_1 and W_2 matrices to every token individually.") as tracker:
            self.wait(2)

        with self.voiceover(text="This intermediate expansion to a huge dimensionality allows the network to store facts and concepts, acting somewhat like a massive key-value memory bank. Attention routes the information, and the Feed-Forward Layer processes and refines the knowledge.") as tracker:
            self.wait(3)

        with self.voiceover(text="Understanding this separation of responsibilities is key to mastering Transformer architecture. Thank you for watching.") as tracker:
            self.play(FadeOut(gotcha_group))
            self.play(FadeOut(title))
            self.wait(2)
