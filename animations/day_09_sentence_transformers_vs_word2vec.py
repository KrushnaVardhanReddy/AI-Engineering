from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class SentenceTransformersVsWord2Vec(VoiceoverScene):
    def construct(self):
        # Setup voiceover service and aesthetics
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE
        Text.set_default(color=BLACK)
        Tex.set_default(color=BLACK)
        MathTex.set_default(color=BLACK)

        # Define brand colors for the presentation
        color_w2v = BLUE
        color_st = RED
        color_accent = GREEN
        color_highlight = PURPLE

        # ---------------------------------------------------------
        # INTRO
        # ---------------------------------------------------------
        title = Text("Word2Vec vs Sentence Transformers", font_size=48, weight=BOLD)
        subtitle = Text("AI/ML Interview Prep: Day 9", font_size=32, color=color_highlight)
        header_group = VGroup(title, subtitle).arrange(DOWN, buff=0.5)

        with self.voiceover(text="""Welcome to Day 9 of our AI and Machine Learning Interview Prep series.
        Today, we are going to dive incredibly deep into two monumental embedding models that shaped the history of Natural Language Processing:
        Word2Vec, representing the era of static word embeddings, and Sentence Transformers, representing the modern era of contextual embeddings.
        Understanding the profound differences between these two architectures, their historical context, and the specific problems they solve is absolutely critical for any AI engineer.
        This isn't just about knowing what they are; it's about understanding why we transitioned from one to the other, what trade-offs are involved, and how they operate under the hood in production environments.
        Let's get started by exploring their fundamental definitions and architectural differences.""") as tracker:
            self.play(Write(title))
            self.wait(1)
            self.play(FadeIn(subtitle, shift=UP))
            self.wait(tracker.duration - 2)

        self.play(FadeOut(header_group))

        # ---------------------------------------------------------
        # PART 1: WHAT IS IT? (Word2Vec)
        # ---------------------------------------------------------
        section1_title = Text("Part 1: What is Word2Vec?", font_size=40, color=color_w2v).to_corner(UL)

        with self.voiceover(text="""Let's start with Word2Vec. What exactly is it?
        Word2Vec is a pioneering predictive model developed by Google in 2013 that maps individual words to dense, static vector representations in a continuous vector space based on their local context window.
        By 'static', we mean that a specific word, regardless of how it is used in a sentence, will always have the exact same vector representation.
        The sheer beauty of Word2Vec is that it intrinsically captures semantic and syntactic relationships through simple spatial proximity.
        Words that appear in similar contexts in the training corpus will be located close to each other in the resulting high-dimensional vector space.""") as tracker:
            self.play(Write(section1_title))
            self.wait(2)

            w2v_def = Text("Static Word Embeddings: 1 Word = 1 Vector", font_size=32).next_to(section1_title, DOWN, aligned_edge=LEFT)
            self.play(FadeIn(w2v_def, shift=RIGHT))
            self.wait(tracker.duration - 4)

        with self.voiceover(text="""To illustrate the mathematical elegance of Word2Vec, let's look at its most famous linguistic regularity property.
        Because the vectors capture semantic relationships so well, we can actually perform algebraic operations on them that yield logically sound results.
        Consider the vector for 'King'.""") as tracker:
            eq_king = MathTex(r"\vec{v}_{\text{King}}", font_size=48)
            self.play(Write(eq_king))
            self.wait(tracker.duration - 1)

        with self.voiceover(text="""If we subtract the conceptual vector for 'Man' from 'King'...""") as tracker:
            eq_minus_man = MathTex(r"\vec{v}_{\text{King}}", "-", r"\vec{v}_{\text{Man}}", font_size=48)
            self.play(TransformMatchingTex(eq_king, eq_minus_man))
            self.wait(tracker.duration - 1)

        with self.voiceover(text="""And then we add the conceptual vector for 'Woman'...""") as tracker:
            eq_plus_woman = MathTex(r"\vec{v}_{\text{King}}", "-", r"\vec{v}_{\text{Man}}", "+", r"\vec{v}_{\text{Woman}}", font_size=48)
            self.play(TransformMatchingTex(eq_minus_man, eq_plus_woman))
            self.wait(tracker.duration - 1)

        with self.voiceover(text="""The resulting vector points almost exactly to the vector space occupied by the word 'Queen'.
        This algebraic capability blew the NLP community away, proving that shallow neural networks could encode deep semantic structures just by analyzing billions of local word co-occurrences.""") as tracker:
            eq_queen = MathTex(r"\vec{v}_{\text{King}}", "-", r"\vec{v}_{\text{Man}}", "+", r"\vec{v}_{\text{Woman}}", r"\approx", r"\vec{v}_{\text{Queen}}", font_size=48)
            self.play(TransformMatchingTex(eq_plus_woman, eq_queen))

            box = SurroundingRectangle(eq_queen, color=color_w2v, buff=0.2)
            self.play(Create(box))
            self.wait(tracker.duration - 3)

        self.play(FadeOut(VGroup(eq_queen, box, w2v_def, section1_title)))


        # ---------------------------------------------------------
        # PART 1: WHAT IS IT? (Sentence Transformers)
        # ---------------------------------------------------------
        section1b_title = Text("Part 1: What are Sentence Transformers?", font_size=40, color=color_st).to_corner(UL)

        with self.voiceover(text="""Now, what about Sentence Transformers?
        Sentence Transformers, introduced much later around 2019, are a modification of the pre-trained BERT architecture designed to derive semantically meaningful embeddings for entire sentences, paragraphs, or even short documents.
        Unlike Word2Vec's static approach, Sentence Transformers leverage a deep, multi-layered self-attention mechanism to read a whole sentence, understand the nuanced interplay between all the words, and output a highly contextualized vector representation.""") as tracker:
            self.play(Write(section1b_title))
            self.wait(2)

            st_def = Text("Contextual Sentence Embeddings via Siamese Networks", font_size=32).next_to(section1b_title, DOWN, aligned_edge=LEFT)
            self.play(FadeIn(st_def, shift=RIGHT))
            self.wait(tracker.duration - 4)

        with self.voiceover(text="""Structurally, the most common implementation uses a Siamese or Triplet network framework.
        Let's visualize this. Suppose we want to compare two sentences to see how similar they are.
        Sentence A and Sentence B are both fed independently into identical, weight-sharing BERT models.""") as tracker:

            rect_a = Rectangle(width=3, height=1, color=BLACK).move_to(LEFT * 4 + UP * 1)
            text_a = Text("Sentence A", font_size=24).move_to(rect_a)

            rect_b = Rectangle(width=3, height=1, color=BLACK).move_to(RIGHT * 4 + UP * 1)
            text_b = Text("Sentence B", font_size=24).move_to(rect_b)

            bert_a = Rectangle(width=3, height=1.5, color=color_st, fill_opacity=0.1).move_to(LEFT * 4 + DOWN * 1)
            bert_text_a = Text("BERT\nModel", font_size=24).move_to(bert_a)

            bert_b = Rectangle(width=3, height=1.5, color=color_st, fill_opacity=0.1).move_to(RIGHT * 4 + DOWN * 1)
            bert_text_b = Text("BERT\nModel", font_size=24).move_to(bert_b)

            arrow_a1 = Arrow(rect_a.get_bottom(), bert_a.get_top(), buff=0.1)
            arrow_b1 = Arrow(rect_b.get_bottom(), bert_b.get_top(), buff=0.1)

            shared_weights = Text("Shared Weights (Siamese Network)", font_size=20, color=GRAY).move_to(DOWN * 1)
            line_shared = DashedLine(bert_a.get_right(), bert_b.get_left(), color=GRAY)

            self.play(FadeIn(VGroup(rect_a, text_a, rect_b, text_b)))
            self.play(GrowArrow(arrow_a1), GrowArrow(arrow_b1))
            self.play(FadeIn(VGroup(bert_a, bert_text_a, bert_b, bert_text_b)))
            self.play(Write(shared_weights), Create(line_shared))

            self.wait(tracker.duration - 6)

        with self.voiceover(text="""The raw output from BERT is a matrix of token embeddings. To get a single, fixed-size vector representing the whole sentence, we apply a Pooling Operation, typically Mean Pooling, across all token vectors.
        This gives us two dense vectors, U and V. Finally, we compute the Cosine Similarity between them.
        This entire pipeline allows Sentence Transformers to natively grasp sentence-level semantics much more deeply than just aggregating individual words.""") as tracker:

            pool_a = Rectangle(width=2, height=1, color=color_accent).move_to(LEFT * 4 + DOWN * 3)
            pool_text_a = Text("Pooling (u)", font_size=24).move_to(pool_a)

            pool_b = Rectangle(width=2, height=1, color=color_accent).move_to(RIGHT * 4 + DOWN * 3)
            pool_text_b = Text("Pooling (v)", font_size=24).move_to(pool_b)

            arrow_a2 = Arrow(bert_a.get_bottom(), pool_a.get_top(), buff=0.1)
            arrow_b2 = Arrow(bert_b.get_bottom(), pool_b.get_top(), buff=0.1)

            cosine = Ellipse(width=3, height=1.5, color=color_highlight).move_to(DOWN * 3)
            cosine_text = Text("Cosine\nSimilarity", font_size=24).move_to(cosine)

            arrow_u = Arrow(pool_a.get_right(), cosine.get_left(), buff=0.1)
            arrow_v = Arrow(pool_b.get_left(), cosine.get_right(), buff=0.1)

            self.play(GrowArrow(arrow_a2), GrowArrow(arrow_b2))
            self.play(FadeIn(VGroup(pool_a, pool_text_a, pool_b, pool_text_b)))
            self.play(GrowArrow(arrow_u), GrowArrow(arrow_v))
            self.play(FadeIn(VGroup(cosine, cosine_text)))

            self.wait(tracker.duration - 5)

        self.play(FadeOut(Group(*self.mobjects)))


        # ---------------------------------------------------------
        # PART 2: WHY DO WE NEED IT? (The Problem of Polysemy)
        # ---------------------------------------------------------
        section2_title = Text("Part 2: Why do we need it? (Before vs After)", font_size=36).to_corner(UL)

        with self.voiceover(text="""So, if Word2Vec was so revolutionary, why did we need Sentence Transformers?
        The primary reason is a phenomenon in linguistics known as Polysemy—where a single word has multiple, drastically different meanings depending entirely on the context.
        Let's look at a concrete 'Before and After' scenario to clearly see the limitation of Word2Vec and how Transformers solve it.""") as tracker:
            self.play(Write(section2_title))
            self.wait(tracker.duration - 1)

        with self.voiceover(text="""Consider these two sentences. Sentence 1: 'I sat by the river bank.'
        Sentence 2: 'I deposited money in the bank.'
        In Word2Vec, the word 'bank' maps to the exact same numerical vector in both cases, completely ignoring the surrounding context.
        If we try to create a sentence embedding using Word2Vec, we typically just average all the word vectors together.""") as tracker:

            s1_text = Text("1. I sat by the river bank.", font_size=32).move_to(UP * 2)
            s2_text = Text("2. I deposited money in the bank.", font_size=32).move_to(UP * 0.5)

            self.play(FadeIn(s1_text, shift=UP), FadeIn(s2_text, shift=UP))

            bank1 = s1_text[-5:-1] # roughly 'bank'
            bank2 = s2_text[-5:-1]

            self.play(bank1.animate.set_color(color_w2v), bank2.animate.set_color(color_w2v))
            self.wait(tracker.duration - 3)

        with self.voiceover(text="""Because the 'bank' vectors are identical, and order doesn't matter when averaging, Word2Vec conflates the concept of a natural river edge with a financial institution.
        The overall sentence representation becomes muddy and imprecise. It suffers heavily from the 'Bag of Words' limitation.""") as tracker:
            w2v_fail = Text("Word2Vec: 'bank' vector is identical. Context is lost.", font_size=28, color=color_w2v).move_to(DOWN * 1.5)
            self.play(Write(w2v_fail))
            self.wait(tracker.duration - 2)

        self.play(FadeOut(w2v_fail))

        with self.voiceover(text="""Now, let's observe the 'After' scenario with Sentence Transformers.
        When these sentences pass through the deep self-attention layers of a Transformer architecture, the model dynamically updates the vector for 'bank' based on the surrounding tokens.
        For the first sentence, the attention mechanism heavily weighs the word 'river'. For the second, it weighs 'money' and 'deposited'.""") as tracker:

            river = s1_text[11:16]
            money = s2_text[12:17]

            self.play(river.animate.set_color(color_accent), money.animate.set_color(color_accent))

            attention_arc1 = CurvedArrow(river.get_bottom(), bank1.get_bottom(), angle=-TAU/4, color=color_st)
            attention_arc2 = CurvedArrow(money.get_bottom(), bank2.get_bottom(), angle=-TAU/4, color=color_st)

            self.play(Create(attention_arc1), Create(attention_arc2))
            self.wait(tracker.duration - 4)

        with self.voiceover(text="""Consequently, the Sentence Transformer produces two completely distinct, highly contextualized vectors for the word 'bank'.
        The final pooled sentence embeddings accurately reflect that these two sentences describe entirely unrelated concepts.
        This contextual awareness is the massive leap forward that Transformers provided for semantic search and understanding.""") as tracker:

            st_success = Text("Sentence Transformers: 'bank' vectors are unique and contextualized.", font_size=28, color=color_st).move_to(DOWN * 1.5)
            self.play(Write(st_success))

            self.play(bank1.animate.set_color(color_st), bank2.animate.set_color(color_st))

            self.wait(tracker.duration - 4)

        self.play(FadeOut(Group(*self.mobjects)))


        # ---------------------------------------------------------
        # PART 3: USE CASES
        # ---------------------------------------------------------
        section3_title = Text("Part 3: Real-World Use Cases", font_size=40).to_corner(UL)

        with self.voiceover(text="""To truly grasp these concepts, let's explore how major tech companies utilize them in real-world, large-scale production applications.
        Where do Word2Vec and Sentence Transformers actually shine in the industry today?""") as tracker:
            self.play(Write(section3_title))
            self.wait(tracker.duration - 1)

        with self.voiceover(text="""Believe it or not, Word2Vec is still incredibly relevant for recommendation systems.
        For example, Spotify uses variations of the Word2Vec algorithm to recommend music.
        But instead of words and sentences, they treat individual songs as 'words' and user playlists as 'sentences'.
        By training Word2Vec on millions of user-curated playlists, the model learns that songs frequently placed together in the same playlists are semantically similar in vibe or genre.
        This enables lightning-fast, highly scalable music recommendations without needing complex deep learning inference at runtime.""") as tracker:

            w2v_use_case = VGroup(
                Text("Word2Vec: Recommendation Systems", font_size=32, weight=BOLD, color=color_w2v),
                Text("Example: Spotify Music Recommendations", font_size=28),
                Text("- Treat Songs as 'Words'", font_size=24),
                Text("- Treat Playlists as 'Sentences'", font_size=24),
                Text("- Captures 'vibe' proximity efficiently", font_size=24)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(LEFT * 3 + UP * 0.5)

            self.play(FadeIn(w2v_use_case, shift=RIGHT))
            self.wait(tracker.duration - 3)

        with self.voiceover(text="""On the other hand, Sentence Transformers are the absolute backbone of modern generative AI workflows, specifically Retrieval-Augmented Generation, or RAG.
        Systems like ChatGPT use powerful Sentence Transformers—like OpenAI's embedding models—to embed vast knowledge bases into vector databases.
        When a user asks a complex, nuanced question, the Sentence Transformer embeds the query, capturing the deep contextual intent, and searches the vector database for the most semantically relevant document chunks to feed to the LLM.
        This requires an intense level of language comprehension that Word2Vec simply cannot provide.""") as tracker:

            st_use_case = VGroup(
                Text("Sentence Transformers: Semantic Search / RAG", font_size=32, weight=BOLD, color=color_st),
                Text("Example: ChatGPT Knowledge Retrieval", font_size=28),
                Text("- Embeds complex user queries", font_size=24),
                Text("- Captures deep semantic intent", font_size=24),
                Text("- Powers Vector Database lookups", font_size=24)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3 + UP * 0.5)

            self.play(FadeIn(st_use_case, shift=LEFT))
            self.wait(tracker.duration - 3)

        self.play(FadeOut(Group(*self.mobjects)))


        # ---------------------------------------------------------
        # PART 4: KEY INTERVIEW INSIGHT
        # ---------------------------------------------------------
        section4_title = Text("Part 4: Key Interview Insight", font_size=40, color=color_highlight).to_corner(UL)

        with self.voiceover(text="""Finally, let's cover the most critical part of this lesson: the Key Interview Insight.
        When an interviewer asks you to compare Word2Vec and Sentence Transformers, they are typically testing your understanding of architectural trade-offs, specifically regarding computational complexity and inference speed.
        You must know when to apply which tool in a constrained production environment.""") as tracker:
            self.play(Write(section4_title))
            self.wait(tracker.duration - 1)

        with self.voiceover(text="""Here is the gotcha: Transformers are incredibly powerful, but they are heavy.
        Let's look at a tradeoff comparison box.""") as tracker:

            box_rect = RoundedRectangle(width=10, height=5, corner_radius=0.5, color=BLACK, fill_opacity=0.05).move_to(DOWN * 0.5)
            self.play(Create(box_rect))
            self.wait(tracker.duration - 1)

        with self.voiceover(text="""For Word2Vec, retrieving an embedding during inference is simply an O(1) dictionary lookup.
        You take the word, look up its pre-calculated vector in a hash table, and you are done.
        It is massively scalable, requires almost zero compute at runtime, and can easily run on small edge devices like mobile phones.""") as tracker:

            w2v_insight = VGroup(
                Text("Word2Vec Inference", font_size=32, weight=BOLD, color=color_w2v),
                MathTex(r"O(1) \text{ Lookup Complexity}", font_size=28),
                Text("Just a dictionary hash map.", font_size=24),
                Text("Ultra-fast, runs on mobile/edge.", font_size=24)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(LEFT * 2.5 + DOWN * 0.5)

            self.play(FadeIn(w2v_insight))
            self.wait(tracker.duration - 2)

        with self.voiceover(text="""Conversely, Sentence Transformers require a full forward pass through a massive neural network containing billions of parameters for every single query.
        The self-attention mechanism scales quadratically—O(N squared)—with respect to the sequence length.
        This means it is computationally expensive, highly memory intensive, and usually requires dedicated GPU hardware for acceptable latency in production.""") as tracker:

            st_insight = VGroup(
                Text("Sentence Transformer Inference", font_size=32, weight=BOLD, color=color_st),
                MathTex(r"O(N^2) \text{ Attention Complexity}", font_size=28),
                Text("Full neural network forward pass.", font_size=24),
                Text("Expensive, typically requires GPUs.", font_size=24)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 2.5 + DOWN * 0.5)

            self.play(FadeIn(st_insight))

            divider = Line(UP * 1.5, DOWN * 2.5, color=BLACK).move_to(DOWN * 0.5)
            self.play(Create(divider))

            self.wait(tracker.duration - 3)

        with self.voiceover(text="""So, the ultimate takeaway to present to your interviewer is this:
        Choose Word2Vec or FastText for applications requiring extreme scale, low latency, and low compute where simple proximity is enough.
        Choose Sentence Transformers for semantic search, RAG, and NLP tasks where deep contextual understanding is strictly required and compute resources are available.
        Thank you for watching Day 9 of our interview prep. Keep building, keep studying, and I will see you in the next lesson.""") as tracker:

            takeaway = Text("Tradeoff: Semantic Depth vs. Computational Cost", font_size=32, color=color_highlight, weight=BOLD).move_to(UP * 2)
            box_takeaway = SurroundingRectangle(takeaway, color=color_highlight, buff=0.2)

            self.play(Write(takeaway), Create(box_takeaway))
            self.wait(tracker.duration - 2)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(2)
