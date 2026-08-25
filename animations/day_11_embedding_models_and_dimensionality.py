from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class EmbeddingModelsAndDimensionality(VoiceoverScene, ThreeDScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Title Screen
        title = Text("Embedding Models and Dimensionality", color=BLACK, font_size=48, weight=BOLD)
        subtitle = Text("AI Engineering - Day 11", color=BLUE, font_size=32)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.5)

        with self.voiceover(text="Welcome back to the AI Engineering series, Day 11. Today, we are going to dive deep into Embedding Models and Dimensionality. This is one of the most fundamental concepts for understanding how modern AI systems process and retrieve text, images, and other forms of unstructured data.") as tracker:
            self.play(Write(title_group), run_time=tracker.duration)

        with self.voiceover(text="By the end of this session, you will understand what embeddings are, why we absolutely need them in machine learning, some major real world use cases, and finally, a key interview insight about dimensionality tradeoffs that you are very likely to be asked about.") as tracker:
            self.wait(1.5)
            self.play(FadeOut(title_group), run_time=tracker.duration - 1.5)

        # ---------------------------------------------------------
        # Part 1: What is it?
        # ---------------------------------------------------------
        section_1_title = Text("What is it?", color=BLACK, font_size=40, weight=BOLD).to_edge(UP)
        definition_text = Text(
            "An embedding is a dense vector representation of data in a high-dimensional continuous space.",
            color=BLACK, font_size=28, t2c={"dense vector representation": BLUE, "high-dimensional continuous space": RED}
        ).next_to(section_1_title, DOWN, buff=0.5)

        with self.voiceover(text="Let's start with a clear definition. What exactly is an embedding model? An embedding is a dense vector representation of data in a high-dimensional continuous space. Let me break that down. Essentially, an embedding model takes complex unstructured data like a word, a sentence, or an image, and converts it into a list of numbers. This list of numbers is what we call a vector.") as tracker:
            self.play(Write(section_1_title))
            self.play(Write(definition_text), run_time=tracker.duration - 2)
        self.wait(1.5)

        # Visualizing mapping words to vectors
        word1 = Text('"Apple"', color=BLACK, font_size=36)
        arrow1 = Arrow(start=LEFT, end=RIGHT, color=BLACK)
        vector1 = Matrix([["0.12"], ["-0.55"], ["0.89"], ["..."]], left_bracket="[", right_bracket="]")
        vector1.set_color(BLACK)
        mapping_group1 = VGroup(word1, arrow1, vector1).arrange(RIGHT, buff=0.5).shift(UP * 0.5)

        word2 = Text('"Orange"', color=BLACK, font_size=36)
        arrow2 = Arrow(start=LEFT, end=RIGHT, color=BLACK)
        vector2 = Matrix([["0.15"], ["-0.50"], ["0.85"], ["..."]], left_bracket="[", right_bracket="]")
        vector2.set_color(BLACK)
        mapping_group2 = VGroup(word2, arrow2, vector2).arrange(RIGHT, buff=0.5).next_to(mapping_group1, DOWN, buff=0.5)

        word3 = Text('"Car"', color=BLACK, font_size=36)
        arrow3 = Arrow(start=LEFT, end=RIGHT, color=BLACK)
        vector3 = Matrix([["-0.88"], ["0.22"], ["-0.10"], ["..."]], left_bracket="[", right_bracket="]")
        vector3.set_color(BLACK)
        mapping_group3 = VGroup(word3, arrow3, vector3).arrange(RIGHT, buff=0.5).next_to(mapping_group2, DOWN, buff=0.5)

        with self.voiceover(text="For example, if we pass the word 'Apple' into an embedding model, it outputs a vector, like zero point one two, negative zero point five five, and so on. If we pass the word 'Orange', we get another vector. Notice how the numbers for 'Orange' are mathematically very close to the numbers for 'Apple', because their semantic meanings are related as fruits.") as tracker:
            self.play(FadeIn(mapping_group1))
            self.wait(0.5)
            self.play(FadeIn(mapping_group2))
            self.wait(tracker.duration - 2.5)

        with self.voiceover(text="However, if we pass the word 'Car', we get a completely different vector, mathematically far away from the fruits. The beautiful thing about embeddings is that geometric distance in the vector space represents semantic similarity in the real world.") as tracker:
            self.play(FadeIn(mapping_group3))
            self.wait(tracker.duration - 1)
        self.wait(1.5)

        self.play(FadeOut(mapping_group1), FadeOut(mapping_group2), FadeOut(mapping_group3), FadeOut(definition_text))

        # Show 3D visualization to explain dimensionality
        axes = ThreeDAxes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], z_range=[-2, 2, 1],
                          x_length=5, y_length=5, z_length=5)
        axes.set_color(BLACK)

        # Manually construct 3D vectors
        vec_apple = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(1, 1, 0.5), color=BLUE)
        dot_apple = Dot3D(point=axes.c2p(1, 1, 0.5), color=BLUE, radius=0.1)
        label_apple = Text("Apple", color=BLUE, font_size=24).next_to(dot_apple, RIGHT)

        vec_orange = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(1.2, 0.9, 0.4), color=GREEN)
        dot_orange = Dot3D(point=axes.c2p(1.2, 0.9, 0.4), color=GREEN, radius=0.1)
        label_orange = Text("Orange", color=GREEN, font_size=24).next_to(dot_orange, RIGHT)

        vec_car = Arrow3D(start=axes.c2p(0, 0, 0), end=axes.c2p(-1.5, -0.5, -1), color=RED)
        dot_car = Dot3D(point=axes.c2p(-1.5, -0.5, -1), color=RED, radius=0.1)
        label_car = Text("Car", color=RED, font_size=24).next_to(dot_car, LEFT)

        with self.voiceover(text="When we talk about dimensionality, we are talking about the length of these vectors. A 3-dimensional vector can be plotted in a 3D space with an X, Y, and Z axis. Here we can clearly see Apple and Orange are clustered together in the space, while Car points in an entirely different direction. Modern embedding models, however, don't just use 3 dimensions. They use hundreds or even thousands of dimensions to capture incredibly subtle nuances of meaning, context, and grammar.") as tracker:
            self.play(Create(axes))
            self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES)
            self.play(Create(vec_apple), FadeIn(dot_apple), FadeIn(label_apple))
            self.play(Create(vec_orange), FadeIn(dot_orange), FadeIn(label_orange))
            self.play(Create(vec_car), FadeIn(dot_car), FadeIn(label_car))
            self.wait(tracker.duration - 4)

        self.wait(1.5)
        self.move_camera(phi=0, theta=-90 * DEGREES)
        self.play(FadeOut(axes), FadeOut(vec_apple), FadeOut(dot_apple), FadeOut(label_apple),
                  FadeOut(vec_orange), FadeOut(dot_orange), FadeOut(label_orange),
                  FadeOut(vec_car), FadeOut(dot_car), FadeOut(label_car), FadeOut(section_1_title))

        # ---------------------------------------------------------
        # Part 2: Why do we need it?
        # ---------------------------------------------------------
        section_2_title = Text("Why do we need it?", color=BLACK, font_size=40, weight=BOLD).to_edge(UP)

        with self.voiceover(text="So, why do we actually need embedding models? To understand this, let's look at what the world looked like before embeddings, using a classic technique called One-Hot Encoding.") as tracker:
            self.play(Write(section_2_title))

        before_title = Text("Before: One-Hot Encoding", color=RED, font_size=32).shift(UP * 2)
        vocab = Text('Vocabulary: ["cat", "dog", "car"]', color=BLACK, font_size=24).next_to(before_title, DOWN)

        cat_vec = Tex(r"cat $\rightarrow [1, 0, 0]$", color=BLACK, font_size=32)
        dog_vec = Tex(r"dog $\rightarrow [0, 1, 0]$", color=BLACK, font_size=32)
        car_vec = Tex(r"car $\rightarrow [0, 0, 1]$", color=BLACK, font_size=32)
        one_hot_group = VGroup(cat_vec, dog_vec, car_vec).arrange(DOWN, buff=0.3).next_to(vocab, DOWN, buff=0.5)

        with self.voiceover(text="Before dense embeddings, we used sparse representations like one-hot encoding. In a one-hot vector, the length of the vector is exactly the size of your entire vocabulary. If you have a hundred thousand words, you have a hundred thousand dimensions. Every single word gets exactly one '1' and everything else is '0'.") as tracker:
            self.play(FadeIn(before_title), FadeIn(vocab))
            self.play(Write(one_hot_group))
            self.wait(tracker.duration - 2)

        problem_text = Text("Problem: No semantic meaning, orthogonal vectors.", color=RED, font_size=28).next_to(one_hot_group, DOWN, buff=0.5)

        with self.voiceover(text="The fatal flaw with this approach is that every word is completely orthogonal to every other word. Mathematically, the dot product between 'cat' and 'dog' is exactly zero. The model has absolutely no idea that a cat and a dog are both animals, and are much more similar to each other than to a car. It also wastes massive amounts of memory storing millions of zeros.") as tracker:
            self.play(Write(problem_text))
            self.wait(tracker.duration - 1)
        self.wait(1.5)

        after_title = Text("After: Dense Embeddings", color=BLUE, font_size=32).shift(UP * 2)
        dense_cat = Tex(r"cat $\rightarrow [0.34, -0.12, 0.89, \dots]$", color=BLACK, font_size=32)
        dense_dog = Tex(r"dog $\rightarrow [0.31, -0.09, 0.92, \dots]$", color=BLACK, font_size=32)
        dense_car = Tex(r"car $\rightarrow [-0.88, 0.45, -0.23, \dots]$", color=BLACK, font_size=32)
        dense_group = VGroup(dense_cat, dense_dog, dense_car).arrange(DOWN, buff=0.3).next_to(after_title, DOWN, buff=0.8)

        with self.voiceover(text="By switching to dense embeddings, we compress the representation into a fixed, much smaller number of dimensions, like 384 or 1536. More importantly, we capture the semantic meaning. Because the numbers are learned during training, 'cat' and 'dog' will share similar values across many dimensions. The dot product is no longer zero; it is a high positive number, representing similarity.") as tracker:
            self.play(FadeOut(before_title), FadeOut(vocab), FadeOut(one_hot_group), FadeOut(problem_text))
            self.play(FadeIn(after_title))
            self.play(Write(dense_group))
            self.wait(tracker.duration - 2)
        self.wait(1.5)

        self.play(FadeOut(after_title), FadeOut(dense_group), FadeOut(section_2_title))

        # ---------------------------------------------------------
        # Part 3: Use Cases
        # ---------------------------------------------------------
        section_3_title = Text("Real-World Use Cases", color=BLACK, font_size=40, weight=BOLD).to_edge(UP)

        with self.voiceover(text="So where are embedding models actually used in production systems today? The reality is, they are the backbone of almost every modern AI application you interact with.") as tracker:
            self.play(Write(section_3_title))

        case1_title = Text("1. Retrieval-Augmented Generation (RAG) (e.g., ChatGPT)", color=BLUE, font_size=28).shift(UP * 1.5)
        case1_desc = Text("Converting company documents into vectors for fast semantic search.", color=BLACK, font_size=24).next_to(case1_title, DOWN)

        case2_title = Text("2. Recommender Systems (e.g., Spotify, Netflix)", color=PURPLE, font_size=28).next_to(case1_desc, DOWN, buff=1.0)
        case2_desc = Text("Embedding users and items in the same space to predict preferences.", color=BLACK, font_size=24).next_to(case2_title, DOWN)

        with self.voiceover(text="The first massive use case is Retrieval-Augmented Generation, or RAG, used by tools like ChatGPT when browsing your documents. When you upload a PDF, the system uses an embedding model, like OpenAI's text-embedding-3, to convert your paragraphs into vectors. When you ask a question, your query is also embedded. The system simply searches for the closest document vectors to your query vector. It's semantic search at massive scale.") as tracker:
            self.play(FadeIn(case1_title))
            self.play(Write(case1_desc))
            self.wait(tracker.duration - 2)

        with self.voiceover(text="A second major use case is Recommender Systems, like those built by Spotify or Netflix. They don't just embed text; they embed users and items. If your user profile is mathematically converted into an embedding, and millions of songs are also embeddings, Spotify can recommend new songs simply by finding the song vectors that sit closest to your user vector in that high-dimensional space.") as tracker:
            self.play(FadeIn(case2_title))
            self.play(Write(case2_desc))
            self.wait(tracker.duration - 2)
        self.wait(1.5)

        self.play(FadeOut(case1_title), FadeOut(case1_desc), FadeOut(case2_title), FadeOut(case2_desc), FadeOut(section_3_title))

        # ---------------------------------------------------------
        # Part 4: Key Interview Insight (Tradeoffs)
        # ---------------------------------------------------------
        section_4_title = Text("Key Interview Insight", color=RED, font_size=40, weight=BOLD).to_edge(UP)

        # Callout box
        box = Rectangle(width=10, height=5, color=RED, fill_opacity=0.05)
        insight_title = Text("The Dimensionality Tradeoff", color=RED, font_size=32, weight=BOLD).next_to(box.get_top(), DOWN, buff=0.3)

        insight_pt1 = Text("Higher Dimensions (e.g., 1536):", color=BLACK, font_size=24, weight=BOLD).next_to(insight_title, DOWN, buff=0.5).align_to(box.get_left() + RIGHT*0.5, LEFT)
        insight_pt1_sub = Text("Better semantic accuracy, but slower search & higher RAM costs.", color=BLACK, font_size=24).next_to(insight_pt1, DOWN, buff=0.1).align_to(insight_pt1, LEFT)

        insight_pt2 = Text("Lower Dimensions (e.g., 384):", color=BLACK, font_size=24, weight=BOLD).next_to(insight_pt1_sub, DOWN, buff=0.5).align_to(insight_pt1, LEFT)
        insight_pt2_sub = Text("Faster search & cheaper storage, but less granular meaning.", color=BLACK, font_size=24).next_to(insight_pt2, DOWN, buff=0.1).align_to(insight_pt1, LEFT)

        with self.voiceover(text="Now for the most important part: The Key Interview Insight. When you are interviewing for an AI Engineering role, you will almost certainly be asked about the tradeoffs of dimensionality. Interviewers want to know if you can balance accuracy with system performance and costs.") as tracker:
            self.play(Write(section_4_title))
            self.play(Create(box))
            self.play(Write(insight_title))
            self.wait(tracker.duration - 3)

        with self.voiceover(text="The core tradeoff is this: using an embedding model with higher dimensions, such as OpenAI's standard 1536 dimensions, provides exceptional semantic accuracy. It captures deep subtleties. However, it requires significantly more RAM in your vector database, and calculating distances takes much more compute, leading to higher latency and costs.") as tracker:
            self.play(FadeIn(insight_pt1), FadeIn(insight_pt1_sub))
            self.wait(tracker.duration - 1)

        with self.voiceover(text="Conversely, using a smaller, lower dimension model, like an open-source model with 384 dimensions, gives you incredibly fast search speeds and massively reduces your cloud storage costs. The tradeoff is that you might lose out on some of the fine-grained semantic meaning, potentially reducing the accuracy of your RAG application.") as tracker:
            self.play(FadeIn(insight_pt2), FadeIn(insight_pt2_sub))
            self.wait(tracker.duration - 1)
        self.wait(1.5)

        # Mathtex derivation of memory footprint
        memory_title = Text("Calculating Memory Footprint", color=BLUE, font_size=28).next_to(insight_pt2_sub, DOWN, buff=0.5).align_to(insight_pt1, LEFT)
        math_eq1 = MathTex(r"\text{Memory} = \text{Vectors} \times \text{Dimensions} \times 4 \text{ bytes (float32)}", color=BLACK, font_size=28).next_to(memory_title, DOWN, buff=0.2).align_to(insight_pt1, LEFT)
        math_eq2 = MathTex(r"\text{Memory} = 1,000,000 \times 1536 \times 4", color=BLACK, font_size=28).next_to(memory_title, DOWN, buff=0.2).align_to(insight_pt1, LEFT)
        math_eq3 = MathTex(r"\text{Memory} \approx 6.14 \text{ GB RAM}", color=BLACK, font_size=28).next_to(memory_title, DOWN, buff=0.2).align_to(insight_pt1, LEFT)

        with self.voiceover(text="A senior engineer should be able to calculate this on the fly. To calculate the uncompressed memory footprint of a vector database, you multiply the number of vectors by the number of dimensions, and then multiply by 4 bytes, assuming standard 32-bit floating point numbers.") as tracker:
            self.play(Write(memory_title))
            self.play(Write(math_eq1))
            self.wait(tracker.duration - 2)

        with self.voiceover(text="For one million documents, using a 1536 dimension model, you would need over 6 Gigabytes of raw RAM just to hold the vectors in memory. If you scale this to a billion documents, you are suddenly dealing with Terabytes of RAM, which costs a fortune. This is why understanding dimensionality is critical for production readiness.") as tracker:
            self.play(TransformMatchingTex(math_eq1, math_eq2))
            self.wait(0.5)
            self.play(TransformMatchingTex(math_eq2, math_eq3))
            self.wait(tracker.duration - 1)
        self.wait(1.5)

        self.play(FadeOut(section_4_title), FadeOut(box), FadeOut(insight_title),
                  FadeOut(insight_pt1), FadeOut(insight_pt1_sub),
                  FadeOut(insight_pt2), FadeOut(insight_pt2_sub),
                  FadeOut(memory_title), FadeOut(math_eq3))

        # Conclusion
        conclusion = Text("Thank you for watching Day 11. Keep building!", color=BLACK, font_size=36, weight=BOLD)

        with self.voiceover(text="That concludes our deep dive into Embedding Models and Dimensionality. Understanding how words are mapped to continuous mathematical spaces is the key to mastering semantic search. Thank you for joining Day 11, and keep building!") as tracker:
            self.play(Write(conclusion))
            self.wait(tracker.duration - 1)
        self.wait(2)
        self.play(FadeOut(conclusion))
