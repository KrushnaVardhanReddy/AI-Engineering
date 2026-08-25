from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class WordEmbeddingsIntro(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # ==========================================
        # Section 1: What is it?
        # ==========================================
        title = Tex("Introduction to Word Embeddings", color=BLACK).scale(1.2).to_edge(UP)
        with self.voiceover(text="Welcome to our series on AI engineering. Today, we are going to talk about Word Embeddings.") as tracker:
            self.play(Write(title))

        self.wait(1.5)

        definition = Tex(
            "Word Embeddings are dense vector representations\\\\",
            "that capture the semantic meaning of words."
        ).scale(0.8).set_color(BLACK).next_to(title, DOWN, buff=1.0)

        with self.voiceover(text="What is a word embedding? In short, word embeddings are dense vector representations that capture the semantic meaning of words.") as tracker:
            self.play(FadeIn(definition, shift=UP))

        self.wait(1.5)

        # Visualizing "Cat" and "Dog" as vectors
        cat_text = Text("Cat", color=BLACK).move_to(LEFT * 4 + UP * 0.5)
        dog_text = Text("Dog", color=BLACK).move_to(LEFT * 4 + DOWN * 0.5)
        arrow1 = Arrow(start=LEFT, end=RIGHT, color=BLACK).next_to(cat_text, RIGHT)
        arrow2 = Arrow(start=LEFT, end=RIGHT, color=BLACK).next_to(dog_text, RIGHT)
        cat_vec_empty = MathTex(r"\begin{bmatrix} \phantom{0.8} \\ \phantom{0.2} \\ \phantom{-0.5} \end{bmatrix}", color=BLUE).next_to(arrow1, RIGHT)
        dog_vec_empty = MathTex(r"\begin{bmatrix} \phantom{0.7} \\ \phantom{0.3} \\ \phantom{-0.4} \end{bmatrix}", color=BLUE).next_to(arrow2, RIGHT)

        cat_vec = MathTex(r"\begin{bmatrix} 0.8 \\ 0.2 \\ -0.5 \end{bmatrix}", color=BLUE).next_to(arrow1, RIGHT)
        dog_vec = MathTex(r"\begin{bmatrix} 0.7 \\ 0.3 \\ -0.4 \end{bmatrix}", color=BLUE).next_to(arrow2, RIGHT)

        group_vectors = VGroup(cat_text, dog_text, arrow1, arrow2, cat_vec, dog_vec, cat_vec_empty, dog_vec_empty)
        group_vectors.move_to(ORIGIN)

        with self.voiceover(text="Instead of treating words as just arbitrary text strings, we translate them into arrays of numbers, called vectors. Because 'cat' and 'dog' are similar animals, their corresponding vectors will also be mathematically similar.") as tracker:
            self.play(Write(cat_text), Write(dog_text))
            self.play(GrowArrow(arrow1), GrowArrow(arrow2))
            self.play(FadeIn(cat_vec_empty), FadeIn(dog_vec_empty))
            self.play(TransformMatchingTex(cat_vec_empty, cat_vec), TransformMatchingTex(dog_vec_empty, dog_vec))

        self.wait(1.5)
        self.play(FadeOut(definition), FadeOut(group_vectors))

        # ==========================================
        # Section 2: Why do we need it?
        # ==========================================
        why_title = Tex("Why do we need it?", color=BLACK).scale(1.2).to_edge(UP)
        with self.voiceover(text="So, why do we need word embeddings? Why not just use the traditional approach?") as tracker:
            self.play(Transform(title, why_title))

        self.wait(1.5)

        # Before: One-Hot Encoding
        one_hot_title = Tex("Before: One-Hot Encoding", color=RED).scale(0.9).move_to(UP * 2)
        vocab = Tex("Vocab: [apple, orange, car]", color=BLACK).scale(0.8).next_to(one_hot_title, DOWN, buff=0.5)
        apple_oh = MathTex(r"\text{apple} = [1, 0, 0]", color=BLACK).scale(0.8).next_to(vocab, DOWN, buff=0.5)
        orange_oh = MathTex(r"\text{orange} = [0, 1, 0]", color=BLACK).scale(0.8).next_to(apple_oh, DOWN, buff=0.2)
        car_oh = MathTex(r"\text{car} = [0, 0, 1]", color=BLACK).scale(0.8).next_to(orange_oh, DOWN, buff=0.2)

        with self.voiceover(text="Before embeddings, we used One-Hot Encoding. If we had a vocabulary of three words: apple, orange, and car, we would represent them like this.") as tracker:
            self.play(FadeIn(one_hot_title), Write(vocab))
            self.play(Write(apple_oh), Write(orange_oh), Write(car_oh))

        with self.voiceover(text="Notice that the distance between 'apple' and 'orange' is the exact same as the distance between 'apple' and 'car'. One-hot encoding completely ignores meaning and relationship.") as tracker:
            self.play(apple_oh.animate.set_color(RED), orange_oh.animate.set_color(RED))
            self.wait(0.5)
            self.play(apple_oh.animate.set_color(BLACK), orange_oh.animate.set_color(BLACK))

        self.wait(1.5)

        # After: Word Embeddings
        we_title = Tex("After: Word Embeddings", color=GREEN).scale(0.9).move_to(UP * 2)
        apple_we = MathTex(r"\text{apple} = [0.9, 0.8, -0.2]", color=BLACK).scale(0.8).next_to(we_title, DOWN, buff=1.0)
        orange_we = MathTex(r"\text{orange} = [0.8, 0.7, -0.3]", color=BLACK).scale(0.8).next_to(apple_we, DOWN, buff=0.5)
        car_we = MathTex(r"\text{car} = [-0.5, -0.6, 0.9]", color=BLACK).scale(0.8).next_to(orange_we, DOWN, buff=0.5)

        with self.voiceover(text="Now, let's look at the approach with Word Embeddings. The vectors are dense, meaning they contain continuous values.") as tracker:
            self.play(FadeOut(one_hot_title), FadeOut(vocab), FadeIn(we_title))
            self.play(TransformMatchingTex(apple_oh, apple_we))
            self.play(TransformMatchingTex(orange_oh, orange_we))
            self.play(TransformMatchingTex(car_oh, car_we))

        with self.voiceover(text="Here, the vectors for 'apple' and 'orange' are closely aligned, whereas the vector for 'car' is very different. Embeddings inherently capture relationships and context.") as tracker:
            self.play(apple_we.animate.set_color(GREEN), orange_we.animate.set_color(GREEN))
            self.wait(0.5)
            self.play(car_we.animate.set_color(RED))
            self.wait(0.5)

        self.wait(1.5)
        self.play(FadeOut(we_title), FadeOut(apple_we), FadeOut(orange_we), FadeOut(car_we))

        # ==========================================
        # Section 3: Use Cases
        # ==========================================
        use_case_title = Tex("Use Cases", color=BLACK).scale(1.2).to_edge(UP)
        with self.voiceover(text="Let's discuss some real-world use cases for word embeddings.") as tracker:
            self.play(Transform(title, use_case_title))

        self.wait(1.5)

        # Case 1: ChatGPT (LLMs)
        chatgpt = Text("1. ChatGPT / Large Language Models", color=BLUE).scale(0.7).move_to(UP * 1 + LEFT * 1)
        chatgpt_desc = Text("Uses embeddings to understand context and generate text.", color=BLACK).scale(0.5).next_to(chatgpt, DOWN, aligned_edge=LEFT)

        with self.voiceover(text="First, Large Language Models like ChatGPT use embeddings as their foundational layer to understand human language, maintain context over long conversations, and predict the next word.") as tracker:
            self.play(FadeIn(chatgpt, shift=RIGHT))
            self.play(Write(chatgpt_desc))

        self.wait(1.5)

        # Case 2: Spotify (Search and Recommendations)
        spotify = Text("2. Spotify (Search & Recommendations)", color=GREEN).scale(0.7).next_to(chatgpt_desc, DOWN, buff=1.0, aligned_edge=LEFT)
        spotify_desc = Text("Uses embeddings to match search queries with songs and podcasts.", color=BLACK).scale(0.5).next_to(spotify, DOWN, aligned_edge=LEFT)

        with self.voiceover(text="Second, companies like Spotify use embeddings in their search engines. They embed both your search query and their podcast descriptions to find semantic matches, even if you don't type the exact keywords.") as tracker:
            self.play(FadeIn(spotify, shift=RIGHT))
            self.play(Write(spotify_desc))

        self.wait(1.5)
        self.play(FadeOut(chatgpt), FadeOut(chatgpt_desc), FadeOut(spotify), FadeOut(spotify_desc))

        # ==========================================
        # Section 4: Key Interview Insight
        # ==========================================
        insight_title = Tex("Key Interview Insight", color=BLACK).scale(1.2).to_edge(UP)
        with self.voiceover(text="Finally, let's look at a key interview insight you need to know.") as tracker:
            self.play(Transform(title, insight_title))

        self.wait(1.5)

        box = Rectangle(width=10, height=4, color=PURPLE)
        box_text1 = Tex(r"\textbf{Out-of-Vocabulary (OOV) Problem}", color=RED).scale(0.9).move_to(box.get_center() + UP * 0.8)
        box_text2 = Tex(r"Traditional embeddings (like Word2Vec) assign 1 vector per word.\\", color=BLACK).scale(0.7).next_to(box_text1, DOWN, buff=0.3)
        box_text3 = Tex(r"If a model sees a brand new word in production, it fails.", color=BLACK).scale(0.7).next_to(box_text2, DOWN, buff=0.2)
        box_text4 = Tex(r"\textit{Solution: Subword tokenization (e.g., BPE used by BERT/GPT)}", color=PURPLE).scale(0.7).next_to(box_text3, DOWN, buff=0.4)

        insight_group = VGroup(box, box_text1, box_text2, box_text3, box_text4)

        with self.voiceover(text="The most common gotcha interviewers will test you on is the Out-of-Vocabulary problem.") as tracker:
            self.play(Create(box))
            self.play(FadeIn(box_text1, shift=UP))

        with self.voiceover(text="Traditional embedding models like Word2Vec map one unique vector to each exact word. But what happens if the model encounters a brand-new word, like a new slang term or a typo, during production?") as tracker:
            self.play(Write(box_text2))
            self.play(Write(box_text3))

        with self.voiceover(text="The model will crash or fail because it has no vector for it. To solve this, modern models like BERT and GPT don't embed whole words. Instead, they use subword tokenization, breaking unknown words down into known chunks.") as tracker:
            self.play(Write(box_text4))

        self.wait(2.5)

        # Conclusion
        with self.voiceover(text="That covers the basics of Word Embeddings. Thanks for watching, and keep building!") as tracker:
            self.play(FadeOut(insight_group), FadeOut(title))

        self.wait(1)
