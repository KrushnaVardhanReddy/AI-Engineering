from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np

class AUCROCCurveExplained(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # 1. Title
        title = Text("Day 28: AUC-ROC Curve Explained", color=BLACK, font_size=48)
        with self.voiceover(text="Welcome to Day 28 of our AI Engineering Mastery series. Today, we are diving deep into one of the most critical evaluation metrics for classification models: The AUC-ROC Curve.") as tracker:
            self.play(FadeIn(title))
            self.wait(1.5)

        with self.voiceover(text="By the end of this video, you will not only understand what this metric is, but also exactly why we need it, real world use cases, and the most common interview gotcha that trips up many engineers.") as tracker:
            self.wait(1.5)

        self.play(FadeOut(title))

        # 2. What is it?
        section_1 = Text("What is it?", color=BLUE, font_size=56).to_edge(UP)
        with self.voiceover(text="First, what exactly is it? The Receiver Operating Characteristic curve, or ROC curve, is a graphical representation that shows the diagnostic ability of a binary classifier system as its discrimination threshold is varied.") as tracker:
            self.play(Write(section_1))
            self.wait(1.5)

        # Draw axes
        axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            x_length=6,
            y_length=6,
            axis_config={"color": BLACK},
        ).shift(DOWN * 0.5)

        labels = axes.get_axis_labels(
            x_label=Text("False Positive Rate (FPR)", color=BLACK, font_size=24),
            y_label=Text("True Positive Rate (TPR)", color=BLACK, font_size=24)
        )

        with self.voiceover(text="It plots the True Positive Rate against the False Positive Rate at various threshold settings.") as tracker:
            self.play(FadeIn(axes), FadeIn(labels))
            self.wait(1.5)

        # ROC Curve
        def roc_func(x):
            return 1 - (1 - x)**3

        def diag_func(x):
            return x

        roc_graph = axes.plot(roc_func, color=BLUE)
        diag_graph = axes.plot(diag_func, color=RED)
        dashed_diag = DashedVMobject(diag_graph, num_dashes=20, color=RED)

        area = axes.get_area(roc_graph, x_range=(0, 1), color=BLUE, opacity=0.3)

        with self.voiceover(text="And AUC stands for Area Under the Curve, which measures the entire two-dimensional area underneath this ROC curve from zero, zero to one, one.") as tracker:
            self.play(FadeIn(roc_graph))
            self.play(FadeIn(area))
            self.wait(1.5)

        auc_text = Text("AUC = 0.85", color=BLACK, font_size=32).move_to(axes.c2p(0.6, 0.4))

        with self.voiceover(text="In simple terms, it provides an aggregate measure of performance across all possible classification thresholds.") as tracker:
            self.play(FadeIn(auc_text))
            self.wait(1.5)

        # Perfect model vs random
        perfect_points = [axes.c2p(0,0), axes.c2p(0,1), axes.c2p(1,1)]
        perfect_graph = VMobject(color=GREEN).set_points_as_corners(perfect_points)

        with self.voiceover(text="An AUC of one point zero means the model perfectly distinguishes between positive and negative classes,") as tracker:
            self.play(Transform(roc_graph, perfect_graph), FadeOut(area), FadeOut(auc_text))
            self.wait(1.5)

        with self.voiceover(text="while an AUC of zero point five means the model is no better than random guessing.") as tracker:
            self.play(Transform(roc_graph, dashed_diag))
            self.wait(1.5)

        self.play(FadeOut(VGroup(axes, labels, roc_graph, dashed_diag, section_1)))

        # 3. Why do we need it?
        section_2 = Text("Why do we need it?", color=RED, font_size=56).to_edge(UP)
        with self.voiceover(text="Why do we need it? Why can't we just use standard accuracy?") as tracker:
            self.play(Write(section_2))
            self.wait(1.5)

        # Imbalanced dataset example
        fraud_text = Text("Fraud Detection System", color=BLACK, font_size=40).shift(UP*2)
        transactions = Text("1,000 Transactions", color=BLACK, font_size=36).next_to(fraud_text, DOWN, buff=0.5)
        legit = Text("990 Legitimate (Negative)", color=GREEN, font_size=32).next_to(transactions, DOWN, buff=0.5).shift(LEFT*2)
        fraud = Text("10 Fraudulent (Positive)", color=RED, font_size=32).next_to(transactions, DOWN, buff=0.5).shift(RIGHT*2)

        with self.voiceover(text="Let us imagine we are building a fraud detection system for a bank. Suppose out of one thousand transactions, only ten are fraudulent, while nine hundred and ninety are legitimate.") as tracker:
            self.play(FadeIn(fraud_text), FadeIn(transactions))
            self.play(FadeIn(legit), FadeIn(fraud))
            self.wait(1.5)

        naive_model = Text("Naive Model: Predict 'Not Fraud' for all", color=BLACK, font_size=36).next_to(legit, DOWN, buff=1).shift(RIGHT*2)
        accuracy_eq = MathTex(r"\text{Accuracy}", "=", r"\frac{990}{1000}", "=", r"99\%", color=BLACK, font_size=48).next_to(naive_model, DOWN, buff=0.5)

        with self.voiceover(text="If we have a naive model that always predicts 'not fraud' for every single transaction, what is its accuracy? It correctly identifies all nine hundred and ninety legitimate transactions. Thus, the accuracy is ninety nine percent.") as tracker:
            self.play(FadeIn(naive_model))
            self.play(Write(accuracy_eq))
            self.wait(1.5)

        missed_fraud = Text("Missed ALL 10 fraudulent transactions!", color=RED, font_size=40, weight=BOLD).next_to(accuracy_eq, DOWN, buff=0.5)

        with self.voiceover(text="This looks amazing on paper! But is it actually useful? No. It entirely missed all ten fraudulent transactions, which is the exact problem we were hired to solve. This is known as the class imbalance problem.") as tracker:
            self.play(FadeIn(missed_fraud))
            self.wait(1.5)

        with self.voiceover(text="With imbalanced datasets, accuracy is a highly misleading metric. Before we used AUC ROC, we might think our model is perfect.") as tracker:
            self.wait(1.5)

        # Show ROC for naive model
        naive_roc_axes = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            x_length=3,
            y_length=3,
            axis_config={"color": BLACK},
        ).to_edge(RIGHT).shift(DOWN*1.5 + LEFT*0.5)

        def naive_roc_func(x):
            return x
        naive_roc_curve = naive_roc_axes.plot(naive_roc_func, color=RED)
        naive_auc = Text("AUC = 0.5", color=RED, font_size=24).next_to(naive_roc_axes, UP)

        with self.voiceover(text="But let's look at the ROC curve for this naive model. Its curve would simply follow the diagonal line, giving an area under the curve of exactly zero point five, exposing the fact that our model has zero true predictive power for the positive class.") as tracker:
            self.play(FadeIn(VGroup(naive_roc_axes, naive_roc_curve, naive_auc)))
            self.wait(1.5)

        with self.voiceover(text="The AUC ROC curve is entirely independent of the class distribution. It forces us to examine the tradeoff between sensitivity, which is catching the actual fraud, and specificity, which is avoiding false alarms.") as tracker:
            self.wait(1.5)

        self.play(FadeOut(VGroup(section_2, fraud_text, transactions, legit, fraud, naive_model, accuracy_eq, missed_fraud, naive_roc_axes, naive_roc_curve, naive_auc)))

        # 4. Mathematical Deep Dive
        section_3 = Text("Mathematical Deep Dive", color=PURPLE, font_size=56).to_edge(UP)
        with self.voiceover(text="Let's break down the math behind the curve so you can clearly visualize what is happening.") as tracker:
            self.play(Write(section_3))
            self.wait(1.5)

        tpr_eq_1 = MathTex(r"\text{TPR}", color=BLACK, font_size=48).shift(UP*1)
        tpr_eq_2 = MathTex(r"\text{TPR}", "=", r"\frac{\text{TP}}{\text{Actual Positives}}", color=BLACK, font_size=48).shift(UP*1)
        tpr_eq_3 = MathTex(r"\text{TPR}", "=", r"\frac{\text{TP}}{\text{TP} + \text{FN}}", color=BLACK, font_size=48).shift(UP*1)

        with self.voiceover(text="The True Positive Rate, also known as Recall or Sensitivity, is calculated as True Positives divided by the sum of True Positives and False Negatives.") as tracker:
            self.play(FadeIn(tpr_eq_1))
            self.play(TransformMatchingTex(tpr_eq_1, tpr_eq_2))
            self.play(TransformMatchingTex(tpr_eq_2, tpr_eq_3))
            tpr_eq_3[2].set_color(GREEN)
            self.wait(1.5)

        with self.voiceover(text="It tells us: out of all the actual positive cases, how many did we correctly identify?") as tracker:
            self.wait(1.5)

        fpr_eq_1 = MathTex(r"\text{FPR}", color=BLACK, font_size=48).shift(DOWN*1)
        fpr_eq_2 = MathTex(r"\text{FPR}", "=", r"\frac{\text{FP}}{\text{Actual Negatives}}", color=BLACK, font_size=48).shift(DOWN*1)
        fpr_eq_3 = MathTex(r"\text{FPR}", "=", r"\frac{\text{FP}}{\text{FP} + \text{TN}}", color=BLACK, font_size=48).shift(DOWN*1)

        with self.voiceover(text="On the horizontal axis, we have the False Positive Rate. This is calculated as False Positives divided by the sum of False Positives and True Negatives.") as tracker:
            self.play(FadeIn(fpr_eq_1))
            self.play(TransformMatchingTex(fpr_eq_1, fpr_eq_2))
            self.play(TransformMatchingTex(fpr_eq_2, fpr_eq_3))
            fpr_eq_3[2].set_color(RED)
            self.wait(1.5)

        with self.voiceover(text="It tells us: out of all the actual negative cases, how many did we incorrectly flag as positive?") as tracker:
            self.wait(1.5)

        with self.voiceover(text="By plotting these two metrics against each other at every possible threshold between zero and one, we generate a curve.") as tracker:
            self.wait(1.5)

        with self.voiceover(text="As we lower the threshold to catch more true positives, we inevitably increase the false positives. A perfect model bends all the way to the top left corner, maximizing true positives without increasing false positives, giving an area of one.") as tracker:
            self.wait(1.5)

        self.play(FadeOut(VGroup(section_3, tpr_eq_3, fpr_eq_3)))

        # 5. Use Cases
        section_4 = Text("Use Cases", color=GREEN, font_size=56).to_edge(UP)
        with self.voiceover(text="Now, let's explore some real-world use cases where AUC ROC is the gold standard for model evaluation.") as tracker:
            self.play(Write(section_4))
            self.wait(1.5)

        openai_box = Rectangle(width=5, height=3, color=BLACK).shift(LEFT*3.5 + DOWN*0.5)
        openai_title = Text("OpenAI / ChatGPT", color=BLUE, font_size=32, weight=BOLD).next_to(openai_box, UP)
        openai_desc = Paragraph(
            "Content Moderation",
            "Balancing safety vs.",
            "false positives",
            alignment="center",
            font_size=24,
            color=BLACK
        ).move_to(openai_box.get_center())

        with self.voiceover(text="First, consider ChatGPT and content moderation at OpenAI. When deciding if a user's prompt violates safety guidelines, OpenAI uses binary classifiers. They must balance strictness.") as tracker:
            self.play(FadeIn(VGroup(openai_box, openai_title, openai_desc)))
            self.wait(1.5)

        with self.voiceover(text="If the threshold is too low, innocent prompts are blocked, frustrating users. If too high, unsafe content slips through. By using the AUC ROC curve, engineers can evaluate the core capability of the safety model independently of the specific threshold they eventually choose to deploy.") as tracker:
            self.wait(1.5)

        spotify_box = Rectangle(width=5, height=3, color=BLACK).shift(RIGHT*3.5 + DOWN*0.5)
        spotify_title = Text("Spotify", color=GREEN, font_size=32, weight=BOLD).next_to(spotify_box, UP)
        spotify_desc = Paragraph(
            "Song Recommendation",
            "Highly imbalanced",
            "(Users save <1% of songs)",
            alignment="center",
            font_size=24,
            color=BLACK
        ).move_to(spotify_box.get_center())

        with self.voiceover(text="Second, let's look at Spotify. In their recommendation algorithms, they might want to predict if a user will save a specific song to their library. Since users only save a tiny fraction of the thousands of songs they are exposed to, the dataset is heavily imbalanced.") as tracker:
            self.play(FadeIn(VGroup(spotify_box, spotify_title, spotify_desc)))
            self.wait(1.5)

        with self.voiceover(text="Spotify engineers use AUC to evaluate whether their ranking models are genuinely learning user preferences, ensuring that a higher score truly corresponds to a higher probability of the user loving the song.") as tracker:
            self.wait(1.5)

        self.play(FadeOut(VGroup(section_4, openai_box, openai_title, openai_desc, spotify_box, spotify_title, spotify_desc)))

        # 6. Key Interview Insight
        section_5 = Text("Key Interview Insight", color=PURPLE, font_size=56).to_edge(UP)
        callout_box = SurroundingRectangle(section_5, color=PURPLE, buff=0.2, fill_color=PURPLE_A, fill_opacity=0.2)

        with self.voiceover(text="Finally, we arrive at the Key Interview Insight. If you are asked about AUC ROC in an interview, this is the most common gotcha that interviewers use to test your practical engineering experience.") as tracker:
            self.play(Write(section_5))
            self.play(FadeIn(callout_box))
            self.wait(1.5)

        q_text = Text("Question: 'Should you ALWAYS use ROC for imbalanced data?'", color=BLACK, font_size=28).shift(UP*1.5)
        a_text = Text("Answer: NO.", color=RED, font_size=48, weight=BOLD).next_to(q_text, DOWN, buff=0.5)

        with self.voiceover(text="The question goes like this: 'If your dataset is highly imbalanced, should you always use the ROC curve?' The answer is a resounding NO.") as tracker:
            self.play(FadeIn(q_text))
            self.play(FadeIn(a_text))
            self.wait(1.5)

        with self.voiceover(text="While ROC is better than accuracy, it can still be misleading in cases of severe class imbalance where you care mostly about the positive class, such as rare diseases or click-through rates.") as tracker:
            self.wait(1.5)

        # Tradeoff example
        tradeoff_eq = MathTex(r"\text{FPR}", "=", r"\frac{\text{FP}}{\text{FP} + \text{TN}}", color=BLACK, font_size=40).shift(DOWN*0.5)

        with self.voiceover(text="The key tradeoff is that the False Positive Rate includes True Negatives in its denominator.") as tracker:
            self.play(FadeIn(tradeoff_eq))
            self.wait(1.5)

        # Example numbers
        tn_val = MathTex(r"\text{TN} = 1,000,000", color=GREEN, font_size=36).next_to(tradeoff_eq, DOWN, buff=0.5).shift(LEFT*2)
        fp_val = MathTex(r"\text{FP} = 10,000", color=RED, font_size=36).next_to(tradeoff_eq, DOWN, buff=0.5).shift(RIGHT*2)
        fpr_result = MathTex(r"\text{FPR} \approx 1\%", color=RED, font_size=36).next_to(tradeoff_eq, DOWN, buff=1.5)

        with self.voiceover(text="If you have millions of negative examples, a massive increase in False Positives might barely move the False Positive Rate, making the ROC curve look artificially good.") as tracker:
            self.play(FadeIn(VGroup(tn_val, fp_val)))
            self.play(FadeIn(fpr_result))
            self.wait(1.5)

        pr_curve_text = Text("Alternative: Precision-Recall (PR) Curve", color=BLUE, font_size=40).next_to(fpr_result, DOWN, buff=1)

        with self.voiceover(text="In these extreme cases, you must explain to the interviewer that you would use the Precision-Recall curve instead, because Precision completely ignores True Negatives and focuses solely on the positive predictions.") as tracker:
            self.play(FadeIn(pr_curve_text))
            self.wait(1.5)

        with self.voiceover(text="Demonstrating that you know when NOT to use AUC ROC is what separates a junior engineer from a senior AI specialist.") as tracker:
            self.wait(1.5)

        self.play(FadeOut(VGroup(section_5, callout_box, q_text, a_text, tradeoff_eq, tn_val, fp_val, fpr_result, pr_curve_text)))

        # 7. Outro
        outro = Text("Happy Learning!", color=BLACK, font_size=56)
        with self.voiceover(text="Thank you for joining today's session on the AUC ROC Curve. This foundational knowledge is critical as we move into more advanced AI systems. Happy learning, and see you on Day 29.") as tracker:
            self.play(FadeIn(outro))
            self.wait(1.5)

        self.play(FadeOut(outro))
