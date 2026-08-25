from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class PrecisionRecallF1Scene(VoiceoverScene):
    def construct(self):
        # Initialize Speech Service
        self.set_speech_service(GTTSService())

        # Whiteboard aesthetic configuration
        self.camera.background_color = WHITE
        Text.set_default(color=BLACK)
        Tex.set_default(color=BLACK)
        MathTex.set_default(color=BLACK)

        # Helper configuration for colors
        ACCENT_PRIMARY = BLUE
        ACCENT_SECONDARY = RED
        ACCENT_SUCCESS = GREEN
        ACCENT_INFO = PURPLE

        # Introduction
        title = Text("Precision, Recall, and F1 Score", font_size=48, weight=BOLD).to_edge(UP)
        underline = Underline(title, color=ACCENT_PRIMARY)

        with self.voiceover(text="Welcome to day 27 of our AI/ML interview prep series. Today, we are tackling three foundational metrics in classification: Precision, Recall, and the F1 Score.") as tracker:
            self.play(Write(title))
            self.play(Create(underline))
            self.wait(1.5)

        # ---------------------------------------------------------
        # SECTION 1: What is it?
        # ---------------------------------------------------------
        section1_title = Text("1. What is it?", font_size=36, color=ACCENT_PRIMARY).to_corner(UL).shift(DOWN*1.5)

        with self.voiceover(text="First, let us start with definitions. What exactly are they?") as tracker:
            self.play(FadeIn(section1_title))
            self.wait(1.5)

        # Precision Definition
        precision_def = Text("Precision: Out of all positive predictions, how many were actually correct?", font_size=28).next_to(section1_title, DOWN, aligned_edge=LEFT, buff=0.5)
        precision_formula = MathTex(
            "\\text{Precision} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Positives}}"
        ).next_to(precision_def, DOWN, buff=0.5)

        with self.voiceover(text="Precision asks: out of all the instances our model predicted as positive, how many were truly positive? It focuses on the accuracy of our positive predictions.") as tracker:
            self.play(Write(precision_def))
            self.play(FadeIn(precision_formula))
            self.wait(1.5)

        precision_formula_2 = MathTex(
            "\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}"
        ).next_to(precision_def, DOWN, buff=0.5)

        with self.voiceover(text="We often abbreviate this as True Positives divided by True Positives plus False Positives.") as tracker:
            self.play(TransformMatchingTex(precision_formula, precision_formula_2))
            self.wait(1.5)

        # Recall Definition
        recall_def = Text("Recall: Out of all actual positives, how many did we find?", font_size=28).next_to(precision_formula_2, DOWN, aligned_edge=LEFT, buff=0.8)
        recall_formula = MathTex(
            "\\text{Recall} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Negatives}}"
        ).next_to(recall_def, DOWN, buff=0.5)

        with self.voiceover(text="Recall, on the other hand, asks: out of all the actual positive instances in our dataset, how many did our model successfully identify? It measures the model's ability to find all the positive cases.") as tracker:
            self.play(Write(recall_def))
            self.play(FadeIn(recall_formula))
            self.wait(1.5)

        recall_formula_2 = MathTex(
            "\\text{Recall} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}"
        ).next_to(recall_def, DOWN, buff=0.5)

        with self.voiceover(text="This is abbreviated as True Positives divided by True Positives plus False Negatives.") as tracker:
            self.play(TransformMatchingTex(recall_formula, recall_formula_2))
            self.wait(1.5)

        # F1 Score Definition
        f1_def = Text("F1 Score: The harmonic mean of Precision and Recall.", font_size=28).next_to(recall_formula_2, DOWN, aligned_edge=LEFT, buff=0.8)
        f1_formula = MathTex(
            "F1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}"
        ).next_to(f1_def, DOWN, buff=0.5)

        with self.voiceover(text="Finally, the F1 Score is the harmonic mean of Precision and Recall. It provides a single metric that balances both concerns, which is especially useful when your class distribution is imbalanced.") as tracker:
            self.play(Write(f1_def))
            self.play(FadeIn(f1_formula))
            self.wait(1.5)

        with self.voiceover(text="Let us clear the board to understand why we need these metrics instead of just looking at accuracy.") as tracker:
            self.play(
                FadeOut(VGroup(section1_title, precision_def, precision_formula_2, recall_def, recall_formula_2, f1_def, f1_formula))
            )
            self.wait(1)


        # ---------------------------------------------------------
        # SECTION 2: Why do we need it?
        # ---------------------------------------------------------
        section2_title = Text("2. Why do we need it?", font_size=36, color=ACCENT_PRIMARY).to_corner(UL).shift(DOWN*1.5)

        with self.voiceover(text="Why do we need precision and recall? Why not just use accuracy? Let us look at a classic problem: class imbalance.") as tracker:
            self.play(FadeIn(section2_title))
            self.wait(1.5)

        # Before (Without concept - Accuracy Illusion)
        scenario_text = Text("Scenario: Detecting a rare disease (1 in 100 people)", font_size=28).next_to(section2_title, DOWN, aligned_edge=LEFT, buff=0.5)

        # Grid of dots to represent 100 people
        dots = VGroup(*[Dot(color=BLACK, radius=0.1) for _ in range(100)])
        dots.arrange_in_grid(rows=5, cols=20, buff=0.2).next_to(scenario_text, DOWN, buff=0.5)

        # Make one dot red (the sick person)
        sick_person = dots[42]
        sick_person.set_color(ACCENT_SECONDARY)

        with self.voiceover(text="Imagine we are building a model to detect a rare disease that affects only one in one hundred people. The red dot represents the single sick person, while the black dots are healthy people.") as tracker:
            self.play(Write(scenario_text))
            self.play(FadeIn(dots))
            self.wait(1.5)

        dumb_model_text = Text("A \"Dumb\" Model predicts: EVERYONE IS HEALTHY", font_size=24, color=ACCENT_SECONDARY).next_to(dots, DOWN, buff=0.5)

        with self.voiceover(text="If we create a very lazy model that simply predicts everyone is healthy all the time, what happens?") as tracker:
            self.play(Write(dumb_model_text))
            self.wait(1.5)

        accuracy_text = MathTex("\\text{Accuracy} = \\frac{99}{100} = 99\\%").next_to(dumb_model_text, DOWN, buff=0.3)

        with self.voiceover(text="Since it correctly identifies the 99 healthy people, its accuracy is 99 percent! But it completely failed to find the one sick person.") as tracker:
            self.play(Write(accuracy_text))
            self.wait(1.5)

        # After (With concept)
        with self.voiceover(text="Accuracy gives us a false sense of security here. Let us see what recall tells us about this lazy model.") as tracker:
            self.wait(1)

        recall_eval = MathTex("\\text{Recall} = \\frac{0 \\text{ (Found Sick)}}{1 \\text{ (Total Sick)}} = 0\\%").next_to(accuracy_text, DOWN, buff=0.5)
        recall_eval.set_color(ACCENT_SECONDARY)

        with self.voiceover(text="The recall of this model is exactly zero percent, because out of the actual positive cases—the sick people—it found zero. This instantly reveals how useless the model is.") as tracker:
            self.play(Write(recall_eval))
            self.wait(1.5)

        with self.voiceover(text="Now let us look at some real world use cases where prioritizing one metric over the other is crucial.") as tracker:
            self.play(
                FadeOut(VGroup(section2_title, scenario_text, dots, dumb_model_text, accuracy_text, recall_eval))
            )
            self.wait(1)

        # ---------------------------------------------------------
        # SECTION 3: Use Cases
        # ---------------------------------------------------------
        section3_title = Text("3. Real-World Use Cases", font_size=36, color=ACCENT_PRIMARY).to_corner(UL).shift(DOWN*1.5)

        with self.voiceover(text="In the real world, companies often have to choose whether to prioritize precision or recall based on the business context.") as tracker:
            self.play(FadeIn(section3_title))
            self.wait(1.5)

        # Use Case 1: YouTube/Spotify (High Precision)
        uc1_title = Text("Use Case 1: YouTube / Spotify Recommendations", font_size=28, weight=BOLD).next_to(section3_title, DOWN, aligned_edge=LEFT, buff=0.5)
        uc1_desc = Text("Goal: High Precision.", font_size=24).next_to(uc1_title, DOWN, aligned_edge=LEFT, buff=0.2)
        uc1_detail = Text("It's better to show a few highly relevant videos, rather than annoy the user with bad ones.", font_size=24, color=DARK_GRAY).next_to(uc1_desc, DOWN, aligned_edge=LEFT, buff=0.2)

        with self.voiceover(text="Take recommendation systems like YouTube or Spotify. Their primary goal is High Precision. They want to ensure that when they recommend a video or song, you will actually like it. It is better to miss some good videos—low recall—than to annoy you with bad recommendations—low precision.") as tracker:
            self.play(Write(uc1_title))
            self.play(Write(uc1_desc))
            self.play(FadeIn(uc1_detail))
            self.wait(1.5)

        # Use Case 2: Tesla/Waymo (High Recall)
        uc2_title = Text("Use Case 2: Tesla / Waymo Autonomous Driving", font_size=28, weight=BOLD).next_to(uc1_detail, DOWN, aligned_edge=LEFT, buff=0.8)
        uc2_desc = Text("Goal: High Recall.", font_size=24).next_to(uc2_title, DOWN, aligned_edge=LEFT, buff=0.2)
        uc2_detail = Text("It's better to false-alarm and brake for a shadow, than to miss a real pedestrian.", font_size=24, color=DARK_GRAY).next_to(uc2_desc, DOWN, aligned_edge=LEFT, buff=0.2)

        with self.voiceover(text="Conversely, look at autonomous driving systems from Tesla or Waymo detecting pedestrians. Their goal is High Recall. It is far better for the car to occasionally brake for a harmless shadow—a false positive—than to ever miss a real pedestrian—a false negative. Safety demands high recall.") as tracker:
            self.play(Write(uc2_title))
            self.play(Write(uc2_desc))
            self.play(FadeIn(uc2_detail))
            self.wait(1.5)

        with self.voiceover(text="Let us move on to the most critical part for your interviews: the tradeoff.") as tracker:
            self.play(
                FadeOut(VGroup(section3_title, uc1_title, uc1_desc, uc1_detail, uc2_title, uc2_desc, uc2_detail))
            )
            self.wait(1)

        # ---------------------------------------------------------
        # SECTION 4: Key Interview Insight
        # ---------------------------------------------------------
        section4_title = Text("4. Key Interview Insight", font_size=36, color=ACCENT_PRIMARY).to_corner(UL).shift(DOWN*1.5)

        with self.voiceover(text="Finally, let us discuss the key interview insight that hiring managers will test you on.") as tracker:
            self.play(FadeIn(section4_title))
            self.wait(1.5)

        # Callout Box
        insight_box = Rectangle(width=12, height=4, color=ACCENT_INFO, fill_color=ACCENT_INFO, fill_opacity=0.1).next_to(section4_title, DOWN, buff=0.5).align_to(section4_title, LEFT)
        insight_title = Text("The Precision-Recall Tradeoff", font_size=32, weight=BOLD).move_to(insight_box.get_top() + DOWN*0.5)
        insight_text_1 = Text("You rarely get both. Increasing Precision almost always decreases Recall.", font_size=24).next_to(insight_title, DOWN, buff=0.5)
        insight_text_2 = Text("Interviewers will present a business problem.", font_size=24).next_to(insight_text_1, DOWN, buff=0.3)
        insight_text_3 = Text("You must identify which metric to maximize based on the cost of False Positives vs False Negatives.", font_size=24, color=ACCENT_SECONDARY).next_to(insight_text_2, DOWN, buff=0.3)

        with self.voiceover(text="The core concept is the Precision-Recall Tradeoff. You rarely get both. As you adjust your model's threshold to increase precision, recall will almost always decrease, and vice versa.") as tracker:
            self.play(Create(insight_box))
            self.play(Write(insight_title))
            self.play(FadeIn(insight_text_1))
            self.wait(1.5)

        with self.voiceover(text="In an interview, you will be presented with a business problem. Your job is not just to build a model, but to identify which metric to maximize. You do this by evaluating the real world cost of a False Positive compared to a False Negative.") as tracker:
            self.play(FadeIn(insight_text_2))
            self.play(Write(insight_text_3))
            self.wait(1.5)

        with self.voiceover(text="If False Positives are expensive, maximize precision. If False Negatives are dangerous, maximize recall. If both matter, optimize the F1 score.") as tracker:
            self.wait(2)

        # Conclusion
        with self.voiceover(text="That wraps up our deep dive into Precision, Recall, and the F1 Score. See you in the next session!") as tracker:
            self.play(
                FadeOut(VGroup(section4_title, insight_box, insight_title, insight_text_1, insight_text_2, insight_text_3, title, underline))
            )
            self.wait(2)
