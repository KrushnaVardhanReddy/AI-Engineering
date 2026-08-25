from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class ChunkingStrategiesForRAG(VoiceoverScene):
    def construct(self):
        # Setting background color to White per whiteboard style
        self.camera.background_color = WHITE

        # Initialize Speech Service
        self.set_speech_service(GTTSService())

        # Title
        title = Text("Chunking Strategies for RAG", color=BLACK, font_size=48).to_edge(UP)
        with self.voiceover(text="Welcome to Day 12 of our AI Engineering Mastery series. Today, we dive deep into a foundational topic for building robust Retrieval-Augmented Generation systems, commonly known as RAG. Specifically, we will be exploring Chunking Strategies. This video will cover exactly what chunking is, why we desperately need it, some real-world use cases, and finally, a critical interview insight that will help you stand out. Let's get started.") as tracker:
            self.play(Write(title))
        self.wait(1.5)

        # Section 1: What is it?
        section1_title = Text("1. What is it?", color=BLUE, font_size=40).next_to(title, DOWN, buff=0.5).align_to(title, LEFT)
        with self.voiceover(text="So, first of all, what is chunking? In a single sentence: Chunking is the process of breaking down large documents into smaller, semantically meaningful pieces of text, called chunks, so that an AI model can efficiently retrieve and process them.") as tracker:
            self.play(Write(section1_title))
        self.wait(1.5)

        # Visual Diagram for What is it
        doc_rect = Rectangle(width=4, height=6, color=BLACK).shift(LEFT * 4)
        doc_text = Text("Large Document\n(e.g., 50 pages)", font_size=24, color=BLACK).move_to(doc_rect.get_center())

        arrow = Arrow(start=LEFT * 1.5, end=RIGHT * 1.5, color=BLACK, buff=0.1)

        chunks = VGroup(*[Rectangle(width=3, height=1.5, color=GREEN) for _ in range(3)]).arrange(DOWN, buff=0.2).shift(RIGHT * 4)
        chunk_texts = VGroup(
            Text("Chunk 1", font_size=24, color=BLACK).move_to(chunks[0].get_center()),
            Text("Chunk 2", font_size=24, color=BLACK).move_to(chunks[1].get_center()),
            Text("Chunk 3", font_size=24, color=BLACK).move_to(chunks[2].get_center())
        )

        with self.voiceover(text="Imagine you have a massive fifty-page document. Feeding this entire document into a language model is not only inefficient, but often impossible due to context window limits. Instead, we apply a chunking strategy. The document is divided into smaller blocks. These blocks can be based on fixed character counts, natural paragraph boundaries, or even semantic meaning.") as tracker:
            self.play(FadeIn(doc_rect), Write(doc_text))
            self.wait(1)
            self.play(FadeIn(arrow))
            self.wait(1)
            self.play(FadeIn(chunks), Write(chunk_texts))
        self.wait(1.5)

        # Clear for Section 2
        with self.voiceover(text="Now that we know what chunking is, let's explore why it is absolutely necessary for RAG systems.") as tracker:
            self.play(FadeOut(VGroup(doc_rect, doc_text, arrow, chunks, chunk_texts, section1_title)))
        self.wait(1.5)

        # Section 2: Why do we need it?
        section2_title = Text("2. Why do we need it?", color=RED, font_size=40).next_to(title, DOWN, buff=0.5).align_to(title, LEFT)
        with self.voiceover(text="Why do we need chunking? The core problem lies in search accuracy and context limits. When we embed large documents as single vectors, the resulting embedding becomes a 'fuzzy' representation of many different topics, washing out specific details.") as tracker:
            self.play(Write(section2_title))
        self.wait(1.5)

        # Before Chunking
        before_title = Text("Before Chunking", color=BLACK, font_size=32).shift(UP * 2 + LEFT * 3.5)
        big_doc = Rectangle(width=3, height=4, color=BLACK).next_to(before_title, DOWN, buff=0.5)
        big_doc_text = Text("History + Math\n+ Science", font_size=20, color=BLACK).move_to(big_doc.get_center())

        big_embed_arrow = Arrow(start=big_doc.get_right(), end=big_doc.get_right() + RIGHT, color=BLACK)
        big_vector = Matrix([["0.1"], ["0.4"], ["0.5"]]).scale(0.7).next_to(big_embed_arrow, RIGHT)
        big_vector.set_color(BLACK)

        # After Chunking
        after_title = Text("After Chunking", color=BLACK, font_size=32).shift(UP * 2 + RIGHT * 3.5)
        small_docs = VGroup(
            Rectangle(width=2.5, height=1, color=GREEN),
            Rectangle(width=2.5, height=1, color=BLUE),
            Rectangle(width=2.5, height=1, color=PURPLE)
        ).arrange(DOWN, buff=0.2).next_to(after_title, DOWN, buff=0.5)

        small_doc_texts = VGroup(
            Text("History", font_size=20, color=BLACK).move_to(small_docs[0].get_center()),
            Text("Math", font_size=20, color=BLACK).move_to(small_docs[1].get_center()),
            Text("Science", font_size=20, color=BLACK).move_to(small_docs[2].get_center())
        )

        with self.voiceover(text="Consider a document that discusses History, Math, and Science all at once. If we embed this entire document without chunking, we get a single vector. This vector is an average of all these topics. If a user asks a specific math question, the retriever might miss this document because the vector is diluted by history and science concepts.") as tracker:
            self.play(Write(before_title), FadeIn(big_doc), Write(big_doc_text))
            self.wait(1)
            self.play(FadeIn(big_embed_arrow), Write(big_vector))
        self.wait(1.5)

        with self.voiceover(text="Now, look at the system after chunking. By breaking the document into distinct chunks for History, Math, and Science, each chunk gets its own highly specific vector embedding. When a user asks a math question, the vector similarity search will easily find the exact Math chunk, providing the language model with precise and relevant context.") as tracker:
            self.play(Write(after_title), FadeIn(small_docs), Write(small_doc_texts))
        self.wait(1.5)

        with self.voiceover(text="This significantly improves retrieval precision and prevents the language model from hallucinating due to irrelevant context. It also saves tokens, which reduces latency and cost.") as tracker:
            self.wait(2)

        # Clear for Section 3
        self.play(FadeOut(VGroup(before_title, big_doc, big_doc_text, big_embed_arrow, big_vector,
                          after_title, small_docs, small_doc_texts, section2_title)))
        self.wait(1.5)

        # Section 3: Use Cases
        section3_title = Text("3. Use Cases", color=GREEN, font_size=40).next_to(title, DOWN, buff=0.5).align_to(title, LEFT)
        with self.voiceover(text="Let's look at a couple of real-world use cases where chunking strategies are deployed at scale.") as tracker:
            self.play(Write(section3_title))
        self.wait(1.5)

        use_case_1 = Text("1. ChatGPT / OpenAI: Semantic Text Splitter", color=BLACK, font_size=32).shift(UP * 1 + LEFT * 0)
        uc1_desc = Text("Breaks context by paragraphs and sentences\nto maintain logical conversational flow.", color=BLACK, font_size=24).next_to(use_case_1, DOWN, buff=0.3)

        use_case_2 = Text("2. Spotify: Fixed-Size Overlapping Chunks", color=BLACK, font_size=32).next_to(uc1_desc, DOWN, buff=1.0)
        uc2_desc = Text("Uses sliding windows (e.g., 500 tokens with 50 overlap)\nfor dense podcast transcript retrieval.", color=BLACK, font_size=24).next_to(use_case_2, DOWN, buff=0.3)

        with self.voiceover(text="First, consider conversational agents like ChatGPT. When parsing uploaded documents or browsing the web, they typically use semantic chunking. They split text along natural boundaries like paragraphs and sentences. This ensures that the logical flow of a thought is preserved, which is crucial for answering complex questions.") as tracker:
            self.play(Write(use_case_1))
            self.play(Write(uc1_desc))
        self.wait(1.5)

        with self.voiceover(text="Second, consider a platform like Spotify processing massive podcast transcripts for search. They often use fixed-size chunking with an overlap. For example, they might extract chunks of 500 tokens with a 50-token overlap. The overlap ensures that a concept split across two chunks isn't lost, maintaining context continuity in long, unstructured audio transcripts.") as tracker:
            self.play(Write(use_case_2))
            self.play(Write(uc2_desc))
        self.wait(1.5)

        # Clear for Section 4
        self.play(FadeOut(VGroup(use_case_1, uc1_desc, use_case_2, uc2_desc, section3_title)))
        self.wait(1.5)

        # Section 4: Key Interview Insight
        section4_title = Text("4. Key Interview Insight", color=PURPLE, font_size=40).next_to(title, DOWN, buff=0.5).align_to(title, LEFT)
        with self.voiceover(text="Finally, let's discuss the most important part: The key interview insight. Interviewers love to test your understanding of the tradeoffs in chunking.") as tracker:
            self.play(Write(section4_title))
        self.wait(1.5)

        callout_box = Rectangle(width=10, height=4, color=PURPLE, fill_color=PURPLE_A, fill_opacity=0.1).shift(DOWN * 1)
        insight_title = Text("The Chunk Size Tradeoff", color=PURPLE, font_size=36).move_to(callout_box.get_top() + DOWN * 0.5)

        tradeoff_text_1 = Text("Small Chunks: High precision, but poor context.", color=BLACK, font_size=28).next_to(insight_title, DOWN, buff=0.5).align_to(insight_title, LEFT)
        tradeoff_text_2 = Text("Large Chunks: Rich context, but high noise (dilution).", color=BLACK, font_size=28).next_to(tradeoff_text_1, DOWN, buff=0.3).align_to(tradeoff_text_1, LEFT)
        tradeoff_text_3 = Text("Solution: Parent-Document Retrieval", color=RED, font_size=32, weight=BOLD).next_to(tradeoff_text_2, DOWN, buff=0.5).align_to(tradeoff_text_1, LEFT)

        with self.voiceover(text="The classic gotcha question is: 'How do you choose the right chunk size?' You need to explain the tradeoff. If your chunks are too small, you get high retrieval precision, but the language model lacks the broader context needed to formulate a good answer. The answer might be technically accurate but practically useless.") as tracker:
            self.play(FadeIn(callout_box), Write(insight_title))
            self.wait(0.5)
            self.play(Write(tradeoff_text_1))
        self.wait(1.5)

        with self.voiceover(text="On the other hand, if your chunks are too large, you capture rich context, but you introduce a lot of noise. The embedding gets diluted, and you might retrieve irrelevant chunks, eating up the context window and increasing costs.") as tracker:
            self.play(Write(tradeoff_text_2))
        self.wait(1.5)

        with self.voiceover(text="The winning interview answer is to mention 'Parent-Document Retrieval' or 'Small-to-Big' retrieval. You embed and search using very small chunks for high precision, but when a match is found, you pass the larger parent document—or the surrounding chunks—to the LLM. This gives you the best of both worlds: pinpoint search accuracy and rich contextual generation.") as tracker:
            self.play(Write(tradeoff_text_3))
        self.wait(2)

        # Conclusion
        with self.voiceover(text="Mastering chunking strategies and their tradeoffs is essential for designing production-ready RAG systems. Thank you for watching Day 12 of the AI Engineering Mastery series. Happy coding!") as tracker:
            self.play(FadeOut(VGroup(callout_box, insight_title, tradeoff_text_1, tradeoff_text_2, tradeoff_text_3, section4_title, title)))

        self.wait(2)
