from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class RegularizationScene(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # ==========================================
        # 1. WHAT IS IT?
        # ==========================================
        title = Tex(r"\textbf{Regularization: L1 (Lasso) vs L2 (Ridge)}", color=BLACK).scale(1.2)
        title.to_edge(UP)

        with self.voiceover(text="Welcome to day 21 of the AI engineering interview prep. Today we are looking at a fundamental machine learning concept known as Regularization, and specifically comparing two of its most popular forms: L1 Lasso, and L2 Ridge. These techniques are absolutely essential to understand for any machine learning interview, as they dictate how we handle complex models and prevent them from overfitting to our training data.") as tracker:
            self.play(Write(title))
            self.wait(2.0)

        def_text = Tex(
            r"Regularization is a technique that ", r"\textbf{penalizes complex models}\\",
            r"to ", r"\textbf{prevent overfitting}", r" on the training data.",
            color=BLACK
        ).scale(0.8)
        def_text.next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="So, what exactly is regularization? In simple terms, regularization is a technique that penalizes complex models to prevent them from overfitting. By introducing a penalty for complexity, we force our models to learn simpler, more generalizable patterns that will perform better on unseen data.") as tracker:
            self.play(FadeIn(def_text[0:2], shift=UP*0.2))
            self.wait(1.5)
            self.play(FadeIn(def_text[2:], shift=UP*0.2))
            self.wait(2.0)

        # Show loss function equation step by step using TransformMatchingTex
        eq_loss_1 = MathTex(r"\text{Loss}", r"=", r"\text{Error}(Y, \hat{Y})", color=BLACK).scale(0.8)
        eq_loss_1.next_to(def_text, DOWN, buff=0.8)

        with self.voiceover(text="Let's look at this mathematically. Originally, our objective function is simply to minimize the error or loss between our actual values Y and predicted values Y-hat.") as tracker:
            self.play(Write(eq_loss_1))
            self.wait(1.5)

        eq_loss_2 = MathTex(r"\text{Loss}", r"=", r"\text{Error}(Y, \hat{Y})", r"+", r"\text{Penalty}(w)", color=BLACK).scale(0.8)
        eq_loss_2.next_to(def_text, DOWN, buff=0.8)

        with self.voiceover(text="When we introduce regularization, we add a penalty term to this loss function. This penalty is based on the model's weights, w. The more complex the model, the larger the weights, and thus the higher the penalty.") as tracker:
            self.play(TransformMatchingTex(eq_loss_1, eq_loss_2))
            self.wait(1.5)

        eq_loss_3 = MathTex(r"\text{Loss}", r"=", r"\text{Error}(Y, \hat{Y})", r"+", r"\lambda \cdot \text{Penalty}(w)", color=BLACK).scale(0.8)
        eq_loss_3.next_to(def_text, DOWN, buff=0.8)

        with self.voiceover(text="Finally, we control the strength of this penalty using a hyperparameter known as lambda. If lambda is zero, we have no regularization. As lambda increases, the penalty for complexity grows stronger, forcing the model's weights to shrink.") as tracker:
            self.play(TransformMatchingTex(eq_loss_2, eq_loss_3))
            self.wait(2.0)

        self.play(FadeOut(def_text), FadeOut(eq_loss_3))

        # ==========================================
        # 2. WHY DO WE NEED IT? (Before vs After)
        # ==========================================
        why_title = Tex(r"\textbf{Why do we need it?}", color=BLUE).scale(1.1)
        why_title.next_to(title, DOWN, buff=0.3)

        axes1 = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": BLACK, "include_numbers": False},
        ).shift(LEFT*3 + DOWN*1)

        axes2 = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": BLACK, "include_numbers": False},
        ).shift(RIGHT*3 + DOWN*1)

        points = [
            (0.5, 1.2), (1.5, 1.5), (2.0, 2.8), (3.0, 2.5), (4.0, 4.2)
        ]

        dots1 = VGroup(*[Dot(axes1.c2p(x, y), color=BLACK) for x, y in points])
        dots2 = VGroup(*[Dot(axes2.c2p(x, y), color=BLACK) for x, y in points])

        label1 = Tex(r"\textbf{Before:} Overfitting (No Penalty)", color=RED).scale(0.7).next_to(axes1, UP)
        label2 = Tex(r"\textbf{After:} Generalized (With Penalty)", color=GREEN).scale(0.7).next_to(axes2, UP)

        with self.voiceover(text="Why do we need regularization? Let's visualize a scenario where we are trying to fit a curve to some data points.") as tracker:
            self.play(Write(why_title))
            self.play(FadeIn(axes1), FadeIn(dots1))
            self.wait(1.5)

        with self.voiceover(text="Without any penalty, a highly complex model might try to perfectly intersect every single data point. This results in a very wiggly, overfitted curve. While its training error is near zero, it has memorized the noise in the data and will likely perform terribly on new, unseen data.") as tracker:
            overfit_curve = axes1.plot(
                lambda x: 0.2*x**3 - 1.2*x**2 + 2.8*x + 0.1 if x < 4.5 else 4,
                color=RED,
                x_range=[0, 4.5]
            )
            self.play(Write(label1), Write(overfit_curve), run_time=2)
            self.wait(2.0)

        with self.voiceover(text="Now, let's see what happens after we apply a regularization penalty. By constraining the model's weights, we prevent it from forming these extreme wiggles.") as tracker:
            self.play(FadeIn(axes2), FadeIn(dots2))
            self.wait(1.5)

        with self.voiceover(text="The model is forced to learn a much simpler, smoother relationship, such as this straight line. This generalized model captures the true underlying trend of the data, rather than the noise, leading to far better performance in the real world.") as tracker:
            generalized_curve = axes2.plot(
                lambda x: 0.8*x + 0.5,
                color=GREEN,
                x_range=[0, 4.5]
            )
            self.play(Write(label2), Write(generalized_curve), run_time=2)
            self.wait(2.0)

        self.play(FadeOut(VGroup(why_title, axes1, dots1, label1, overfit_curve, axes2, dots2, label2, generalized_curve)))

        # ==========================================
        # L1 vs L2 Details
        # ==========================================
        l1_title = Tex(r"\textbf{L1 Regularization (Lasso)}", color=PURPLE).scale(1.1).shift(UP*2.5)
        l2_title = Tex(r"\textbf{L2 Regularization (Ridge)}", color=BLUE).scale(1.1).shift(DOWN*0.5)

        l1_eq_1 = MathTex(r"\text{Penalty}", r"=", r"\text{Magnitude of weights}", color=BLACK).scale(0.9).next_to(l1_title, DOWN)
        l2_eq_1 = MathTex(r"\text{Penalty}", r"=", r"\text{Square of weights}", color=BLACK).scale(0.9).next_to(l2_title, DOWN)

        l1_eq_2 = MathTex(r"\text{Penalty}", r"=", r"\sum |w_i|", color=BLACK).scale(0.9).next_to(l1_title, DOWN)
        l2_eq_2 = MathTex(r"\text{Penalty}", r"=", r"\sum w_i^2", color=BLACK).scale(0.9).next_to(l2_title, DOWN)

        l1_desc = Tex(r"Drives weights to exactly zero (Feature Selection).", color=BLACK).scale(0.7).next_to(l1_eq_2, DOWN)
        l2_desc = Tex(r"Drives weights close to zero, but rarely exactly zero.", color=BLACK).scale(0.7).next_to(l2_eq_2, DOWN)

        with self.voiceover(text="Now let's dive into the core differences between our two main types of regularization.") as tracker:
            self.play(Write(l1_title))
            self.play(Write(l1_eq_1))
            self.wait(1.5)

        with self.voiceover(text="L1 regularization, commonly known as Lasso, calculates the penalty by taking the sum of the absolute values of the magnitude of the model's coefficients.") as tracker:
            self.play(TransformMatchingTex(l1_eq_1, l1_eq_2))
            self.wait(1.5)

        with self.voiceover(text="This absolute value penalty has a very unique mathematical property: it tends to drive the weights of less important features down to exactly zero. Because of this, L1 inherently performs feature selection, leaving you with a sparse, simple model that only uses the most critical variables.") as tracker:
            self.play(FadeIn(l1_desc))
            self.wait(2.0)

        with self.voiceover(text="On the other hand, we have L2 regularization, also known as Ridge regression.") as tracker:
            self.play(Write(l2_title))
            self.play(Write(l2_eq_1))
            self.wait(1.5)

        with self.voiceover(text="L2 calculates its penalty by taking the sum of the squared magnitudes of the coefficients. By squaring the weights, L2 severely penalizes very large weights much more than small ones.") as tracker:
            self.play(TransformMatchingTex(l2_eq_1, l2_eq_2))
            self.wait(1.5)

        with self.voiceover(text="As a result, L2 encourages the model to distribute the weight across all features evenly. It shrinks all weights to be very small, but it rarely ever drives them to exactly zero. You keep all your features, but prevent any single one from dominating.") as tracker:
            self.play(FadeIn(l2_desc))
            self.wait(2.0)

        self.play(FadeOut(VGroup(l1_title, l1_eq_2, l1_desc, l2_title, l2_eq_2, l2_desc)))

        # ==========================================
        # 3. USE CASES
        # ==========================================
        uc_title = Tex(r"\textbf{Real-World Use Cases}", color=BLUE).scale(1.1)
        uc_title.next_to(title, DOWN, buff=0.5)

        uc1 = Tex(r"\textbf{ChatGPT (OpenAI):} Uses forms of L2 regularization (weight decay)\\", r"during transformer training to prevent huge parameter explosions.", color=BLACK).scale(0.7)
        uc1.next_to(uc_title, DOWN, buff=1.0)

        uc2 = Tex(r"\textbf{Zillow (Pricing Models):} Uses L1 (Lasso) in sparse datasets\\", r"to select only the most important features (like square footage) out of thousands of variables.", color=BLACK).scale(0.7)
        uc2.next_to(uc1, DOWN, buff=0.8)

        with self.voiceover(text="So, where do we see these used in industry? Let's look at two prominent real world examples.") as tracker:
            self.play(Write(uc_title))
            self.wait(1.5)

        with self.voiceover(text="First, consider massive language models like ChatGPT built by OpenAI. During the training of these giant transformers, they utilize a technique called weight decay, which is mathematically equivalent to L2 regularization. This is crucial because it prevents the billions of parameters from exploding to extreme values, keeping the neural network stable during its learning process.") as tracker:
            self.play(FadeIn(uc1, shift=LEFT*0.5))
            self.wait(2.0)

        with self.voiceover(text="Conversely, look at a company like Zillow, predicting housing prices. They might start with thousands of potential features for a house, many of which are useless. By applying L1 Lasso regression, they can automatically drive the weights of irrelevant features to zero. This selects only the most important indicators, like square footage and zip code, resulting in a cleaner, faster, and highly interpretable model.") as tracker:
            self.play(FadeIn(uc2, shift=LEFT*0.5))
            self.wait(2.0)

        self.play(FadeOut(VGroup(uc_title, uc1, uc2)))

        # ==========================================
        # 4. KEY INTERVIEW INSIGHT
        # ==========================================
        insight_box = Rectangle(width=11, height=4.5, color=RED, fill_color=RED, fill_opacity=0.1)
        insight_title = Tex(r"\textbf{Key Interview Insight}", color=RED).scale(1.1)
        insight_title.next_to(insight_box.get_top(), DOWN, buff=0.3)

        insight_text1 = Tex(r"Interviewers will ask: \textit{'When do you choose L1 vs L2?'}", color=BLACK).scale(0.8)
        insight_text2 = Tex(r"$\rightarrow$ \textbf{Choose L1 (Lasso)} if you suspect many features are irrelevant \\ and you want a sparse, interpretable model.", color=BLACK).scale(0.7)
        insight_text3 = Tex(r"$\rightarrow$ \textbf{Choose L2 (Ridge)} if you believe most features contribute \\ and you just want to prevent extreme weight values.", color=BLACK).scale(0.7)

        insight_text1.next_to(insight_title, DOWN, buff=0.4)
        insight_text2.next_to(insight_text1, DOWN, buff=0.4)
        insight_text3.next_to(insight_text2, DOWN, buff=0.4)

        with self.voiceover(text="Finally, let's synthesize this into our key interview insight. The absolute most common question an interviewer will ask you on this topic is simply: When do you choose L1 versus L2?") as tracker:
            self.play(FadeIn(insight_box), Write(insight_title))
            self.play(FadeIn(insight_text1))
            self.wait(2.0)

        with self.voiceover(text="Your answer should be crisp and confident. You should state that you choose L1 Lasso when you have high dimensionality and suspect that a large portion of your features are irrelevant. L1 will give you a sparse model that is highly interpretable because it throws away the noise.") as tracker:
            self.play(FadeIn(insight_text2, shift=UP*0.2))
            self.wait(2.0)

        with self.voiceover(text="On the other hand, you should state that you choose L2 Ridge when you believe that the majority of your features carry at least some useful signal. L2 will keep all your features, but shrink them to prevent any single feature from overpowering the model, which is especially useful when features are highly correlated.") as tracker:
            self.play(FadeIn(insight_text3, shift=UP*0.2))
            self.wait(2.0)

        with self.voiceover(text="Mastering this trade-off between feature selection in L1 and uniform shrinkage in L2 is an absolute must for machine learning system design interviews. Thank you for watching, and see you in the next lesson!") as tracker:
            self.wait(3.0)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(1)
