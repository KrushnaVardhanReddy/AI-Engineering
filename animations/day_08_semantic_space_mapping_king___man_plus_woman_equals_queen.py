from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class SemanticSpaceMapping(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # ==========================================
        # Introduction: Title
        # ==========================================
        title = Text("Semantic Space Mapping", color=BLACK).scale(1.2)
        subtitle = Text("King - Man + Woman = Queen", color=BLUE).scale(0.8).next_to(title, DOWN)

        with self.voiceover(text="Welcome to Day 8 of our AI Engineering mastery series. Today, we are diving deep into the fascinating world of Semantic Space Mapping.") as tracker:
            self.play(Write(title))
            self.wait(1.5)
            self.play(FadeIn(subtitle))
            self.wait(1.5)

        self.play(FadeOut(title), FadeOut(subtitle))

        # ==========================================
        # 1. What is it?
        # ==========================================
        section_1_title = Text("1. What is Semantic Space Mapping?", color=BLACK, weight=BOLD).to_edge(UP)
        with self.voiceover(text="So, what is Semantic Space Mapping? At its core, it is a mathematical representation where words or concepts are mapped to dense vectors in a high-dimensional space.") as tracker:
            self.play(Write(section_1_title))
            self.wait(1.5)

        def_text = Text(
            "Mapping concepts to continuous vectors\nso that spatial distance reflects semantic meaning.",
            color=BLACK, font_size=32
        ).next_to(section_1_title, DOWN, buff=1.0)

        with self.voiceover(text="Simply put, it maps concepts to continuous vectors so that the spatial distance between these vectors directly reflects their semantic meaning. Words that mean similar things are placed closer together in this mathematical space.") as tracker:
            self.play(FadeIn(def_text))
            self.wait(1.5)

        # Draw a 2D axes to represent the space
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"color": BLACK, "include_numbers": False},
        ).scale(0.6).next_to(def_text, DOWN, buff=0.5)

        with self.voiceover(text="Imagine a two-dimensional space. On the x-axis, we might have a feature like gender, going from male to female. On the y-axis, we might have a feature representing royalty or status.") as tracker:
            self.play(Create(axes))
            self.wait(1.5)

        # Plot points: Man, Woman, King, Queen
        p_man = axes.c2p(-1.5, -1.0)
        p_woman = axes.c2p(1.5, -1.0)
        p_king = axes.c2p(-1.5, 1.5)
        p_queen = axes.c2p(1.5, 1.5)

        d_man = Dot(p_man, color=BLUE)
        d_woman = Dot(p_woman, color=RED)
        d_king = Dot(p_king, color=BLUE)
        d_queen = Dot(p_queen, color=RED)

        l_man = Text("Man", color=BLACK, font_size=24).next_to(d_man, DOWN)
        l_woman = Text("Woman", color=BLACK, font_size=24).next_to(d_woman, DOWN)
        l_king = Text("King", color=BLACK, font_size=24).next_to(d_king, UP)
        l_queen = Text("Queen", color=BLACK, font_size=24).next_to(d_queen, UP)

        with self.voiceover(text="Let's place some words in this space. Here is the vector for 'Man', and here is the vector for 'Woman'.") as tracker:
            self.play(FadeIn(d_man), Write(l_man))
            self.play(FadeIn(d_woman), Write(l_woman))
            self.wait(1.5)

        with self.voiceover(text="Now let's add 'King' and 'Queen'. Because they are related concepts, they occupy specific relative positions.") as tracker:
            self.play(FadeIn(d_king), Write(l_king))
            self.play(FadeIn(d_queen), Write(l_queen))
            self.wait(1.5)

        # Show the mathematical operation
        v_man_to_woman = Arrow(p_man, p_woman, buff=0.1, color=PURPLE)
        v_king_to_queen = Arrow(p_king, p_queen, buff=0.1, color=PURPLE)

        with self.voiceover(text="The magic happens when we perform vector arithmetic. The relationship between Man and Woman is mathematically captured by a vector representing a shift in gender.") as tracker:
            self.play(GrowArrow(v_man_to_woman))
            self.wait(1.5)

        with self.voiceover(text="If we take the vector for 'King', subtract the vector for 'Man', and add the vector for 'Woman', we traverse that same relationship vector.") as tracker:
            self.play(TransformFromCopy(v_man_to_woman, v_king_to_queen))
            self.wait(1.5)

        with self.voiceover(text="And remarkably, we end up exactly at the vector for 'Queen'. This arithmetic property proves that the embedding space encodes semantic meaning in its geometry.") as tracker:
            self.play(Indicate(d_queen, color=GREEN, scale_factor=1.5))
            self.wait(2.0)

        self.play(
            FadeOut(section_1_title), FadeOut(def_text), FadeOut(axes),
            FadeOut(d_man), FadeOut(d_woman), FadeOut(d_king), FadeOut(d_queen),
            FadeOut(l_man), FadeOut(l_woman), FadeOut(l_king), FadeOut(l_queen),
            FadeOut(v_man_to_woman), FadeOut(v_king_to_queen)
        )

        # ==========================================
        # 2. Why do we need it?
        # ==========================================
        section_2_title = Text("2. Why Do We Need It?", color=BLACK, weight=BOLD).to_edge(UP)
        with self.voiceover(text="Why do we need this mapping? To understand that, let's look at how computers used to process text before semantic spaces existed.") as tracker:
            self.play(Write(section_2_title))
            self.wait(1.5)

        before_title = Text("Before: One-Hot Encoding", color=RED, font_size=36).shift(UP*2)

        with self.voiceover(text="Historically, words were represented using One-Hot Encoding. Every word had its own unique dimension, and there was no relationship between them.") as tracker:
            self.play(FadeIn(before_title))
            self.wait(1.5)

        # Using TransformMatchingTex for Math derivation constraint
        dog_step1 = MathTex(r"\text{Dog}", color=BLACK).shift(UP*0.5)
        dog_step2 = MathTex(r"\text{Dog} = [1, 0, 0, \dots, 0]", color=BLACK).shift(UP*0.5)

        cat_step1 = MathTex(r"\text{Cat}", color=BLACK).shift(DOWN*0.5)
        cat_step2 = MathTex(r"\text{Cat} = [0, 1, 0, \dots, 0]", color=BLACK).shift(DOWN*0.5)

        car_step1 = MathTex(r"\text{Car}", color=BLACK).shift(DOWN*1.5)
        car_step2 = MathTex(r"\text{Car} = [0, 0, 1, \dots, 0]", color=BLACK).shift(DOWN*1.5)

        with self.voiceover(text="For instance, 'Dog' starts as a concept, but is represented by a one in the first position.") as tracker:
            self.play(Write(dog_step1))
            self.play(TransformMatchingTex(dog_step1, dog_step2))
            self.wait(1.5)

        with self.voiceover(text="Similarly, 'Cat' is derived as a one in the second position, and 'Car' in the third.") as tracker:
            self.play(Write(cat_step1))
            self.play(TransformMatchingTex(cat_step1, cat_step2))
            self.play(Write(car_step1))
            self.play(TransformMatchingTex(car_step1, car_step2))
            self.wait(1.5)

        with self.voiceover(text="In this system, the distance between 'Dog' and 'Cat' is exactly the same as the distance between 'Dog' and 'Car'. The model has no idea that dogs and cats are both animals. The semantics are completely lost.") as tracker:
            cross1 = Cross(Line(dog_step2.get_right()+RIGHT*0.5, cat_step2.get_right()+RIGHT*0.5), stroke_color=RED)
            cross2 = Cross(Line(dog_step2.get_right()+RIGHT*0.5, car_step2.get_right()+RIGHT*0.5), stroke_color=RED)
            self.play(Create(cross1), Create(cross2))
            self.wait(2.0)

        self.play(
            FadeOut(before_title), FadeOut(dog_step2), FadeOut(cat_step2),
            FadeOut(car_step2), FadeOut(cross1), FadeOut(cross2)
        )

        after_title = Text("After: Semantic Embeddings", color=GREEN, font_size=36).shift(UP*2)

        with self.voiceover(text="With semantic embeddings, words are compressed into dense, continuous vectors of numbers that actually represent their meaning.") as tracker:
            self.play(FadeIn(after_title))
            self.wait(1.5)

        # Using TransformMatchingTex for Math derivation constraint
        dog_emb_step1 = MathTex(r"\text{Dog}", color=BLACK).shift(UP*0.5)
        dog_emb_step2 = MathTex(r"\text{Dog} \approx [0.8, -0.2, 0.5]", color=BLACK).shift(UP*0.5)

        cat_emb_step1 = MathTex(r"\text{Cat}", color=BLACK).shift(DOWN*0.5)
        cat_emb_step2 = MathTex(r"\text{Cat} \approx [0.7, -0.1, 0.6]", color=BLACK).shift(DOWN*0.5)

        car_emb_step1 = MathTex(r"\text{Car}", color=BLACK).shift(DOWN*1.5)
        car_emb_step2 = MathTex(r"\text{Car} \approx [-0.9, 0.8, -0.3]", color=BLACK).shift(DOWN*1.5)

        with self.voiceover(text="Now, 'Dog' transforms into a vector representing features like 'is animal' or 'is pet'.") as tracker:
            self.play(Write(dog_emb_step1))
            self.play(TransformMatchingTex(dog_emb_step1, dog_emb_step2))
            self.wait(1.5)

        with self.voiceover(text="'Cat' is mapped to a very similar vector. 'Car', being unrelated, is transformed into completely different values.") as tracker:
            self.play(Write(cat_emb_step1))
            self.play(TransformMatchingTex(cat_emb_step1, cat_emb_step2))
            self.play(Write(car_emb_step1))
            self.play(TransformMatchingTex(car_emb_step1, car_emb_step2))
            self.wait(1.5)

        with self.voiceover(text="This enables neural networks to generalize. If a model learns something about dogs, it automatically applies that knowledge to cats, massively improving efficiency and performance.") as tracker:
            box = SurroundingRectangle(VGroup(dog_emb_step2, cat_emb_step2), color=GREEN, buff=0.2)
            self.play(Create(box))
            self.wait(2.0)

        self.play(
            FadeOut(section_2_title), FadeOut(after_title),
            FadeOut(dog_emb_step2), FadeOut(cat_emb_step2), FadeOut(car_emb_step2), FadeOut(box)
        )

        # ==========================================
        # 3. Use Cases
        # ==========================================
        section_3_title = Text("3. Real-World Use Cases", color=BLACK, weight=BOLD).to_edge(UP)
        with self.voiceover(text="So, where is semantic space mapping actually used in the real world? It forms the foundation of modern artificial intelligence.") as tracker:
            self.play(Write(section_3_title))
            self.wait(1.5)

        case1_title = Text("Case 1: ChatGPT (OpenAI)", color=BLUE, font_size=32).shift(UP*1.5 + LEFT*2)
        case1_desc = Text(
            "Uses dense embeddings to understand context\nand retrieve relevant information in RAG systems.",
            color=BLACK, font_size=24
        ).next_to(case1_title, DOWN, aligned_edge=LEFT)

        with self.voiceover(text="First, Large Language Models like ChatGPT use these dense embeddings fundamentally. When you ask a question, your text is mapped into semantic space to retrieve relevant context, which is especially critical in Retrieval-Augmented Generation or RAG systems.") as tracker:
            self.play(FadeIn(case1_title))
            self.play(Write(case1_desc))
            self.wait(2.0)

        case2_title = Text("Case 2: Spotify Recommendations", color=GREEN, font_size=32).shift(DOWN*1.0 + LEFT*2)
        case2_desc = Text(
            "Maps users and songs into the same space.\nSuggests songs that are 'close' to your preferences.",
            color=BLACK, font_size=24
        ).next_to(case2_title, DOWN, aligned_edge=LEFT)

        with self.voiceover(text="Second, Recommendation Engines like Spotify rely heavily on this. By mapping both users and songs into the same semantic space, Spotify can mathematically calculate which tracks are closest to a user's listening vector, generating highly personalized playlists.") as tracker:
            self.play(FadeIn(case2_title))
            self.play(Write(case2_desc))
            self.wait(2.0)

        self.play(
            FadeOut(section_3_title), FadeOut(case1_title), FadeOut(case1_desc),
            FadeOut(case2_title), FadeOut(case2_desc)
        )

        # ==========================================
        # 4. Key Interview Insight
        # ==========================================
        section_4_title = Text("4. Key Interview Insight", color=BLACK, weight=BOLD).to_edge(UP)
        with self.voiceover(text="Finally, let's cover the key insight that interviewers will test you on. This is where many candidates stumble.") as tracker:
            self.play(Write(section_4_title))
            self.wait(1.5)

        callout_box = Rectangle(width=10, height=4, color=RED, fill_opacity=0.1).center()
        insight_title = Text("The Dimensionality Tradeoff", color=RED, weight=BOLD, font_size=36).move_to(callout_box.get_top() + DOWN*0.5)

        # Split text into two separate objects to prevent slicing issues
        insight_text_1 = Text(
            "Too few dimensions: Underfitting (Loses nuance)",
            color=BLACK, font_size=28, t2c={"Underfitting": RED}
        ).next_to(insight_title, DOWN, buff=0.5)

        insight_text_2 = Text(
            "Too many dimensions: Overfitting (Curse of Dimensionality,\nhigh computational cost, sparse distances)",
            color=BLACK, font_size=28, t2c={"Overfitting": RED, "Curse of Dimensionality": PURPLE}
        ).next_to(insight_text_1, DOWN, buff=0.5)

        with self.voiceover(text="The concept they will ask about is the Dimensionality Tradeoff. When designing an embedding space, how many dimensions should you use?") as tracker:
            self.play(Create(callout_box))
            self.play(Write(insight_title))
            self.wait(1.5)

        with self.voiceover(text="If you use too few dimensions, you get underfitting. The model cannot capture all the complex semantic nuances, and unrelated concepts get squashed together.") as tracker:
            self.play(Write(insight_text_1))
            self.wait(2.0)

        with self.voiceover(text="But if you use too many dimensions, you suffer from overfitting and the Curse of Dimensionality. Computational costs skyrocket, and distances between vectors start to become uniform and meaningless. Finding the sweet spot, typically between 256 and 1536 dimensions depending on the model, is crucial.") as tracker:
            self.play(Write(insight_text_2))
            self.wait(2.0)

        self.play(
            FadeOut(section_4_title), FadeOut(callout_box),
            FadeOut(insight_title), FadeOut(insight_text_1), FadeOut(insight_text_2)
        )

        # ==========================================
        # Outro
        # ==========================================
        outro = Text("Master the math. Ace the interview.", color=BLACK).scale(1.2)
        with self.voiceover(text="Understanding semantic space mapping is essential for any AI engineer. Master the math, and you will ace the interview. Thanks for watching.") as tracker:
            self.play(Write(outro))
            self.wait(2.0)
            self.play(FadeOut(outro))
            self.wait(1.0)
