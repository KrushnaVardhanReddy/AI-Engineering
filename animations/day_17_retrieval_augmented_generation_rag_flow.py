from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class RAGFlow(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # Section 1: What is it?
        title = Text("Retrieval-Augmented Generation (RAG)", font_size=40, color=BLACK).to_edge(UP)
        with self.voiceover(text="Welcome to Day 17. Today, we are exploring Retrieval-Augmented Generation, commonly known as RAG. RAG is a technique that enhances large language models by fetching relevant information from an external database before generating an answer. Think of it as giving the AI an open-book exam instead of forcing it to rely purely on its memorized knowledge.") as tracker:
            self.play(Write(title))
            self.wait(1.5)

            # Diagram: User -> LLM
            user_text = Text("User Question", font_size=24, color=BLACK).shift(LEFT * 4 + UP * 1)
            llm_text = Text("LLM", font_size=24, color=WHITE).next_to(user_text, RIGHT, buff=2)
            llm_box = SurroundingRectangle(llm_text, color=BLUE, fill_color=BLUE, fill_opacity=1, buff=0.2)
            llm_group = VGroup(llm_box, llm_text)

            db_text = Text("Vector DB\n(External Knowledge)", font_size=24, color=WHITE)
            db_box = SurroundingRectangle(db_text, color=GREEN, fill_color=GREEN, fill_opacity=1, buff=0.2)
            db_group = VGroup(db_box, db_text).next_to(llm_group, DOWN, buff=1.5)

            arrow_user_to_db = Arrow(user_text.get_bottom(), db_group.get_left(), buff=0.1, color=BLACK)
            arrow_db_to_llm = Arrow(db_group.get_top(), llm_group.get_bottom(), buff=0.1, color=BLACK)

            self.play(FadeIn(user_text))
            self.play(FadeIn(db_group), GrowArrow(arrow_user_to_db))
            self.play(FadeIn(llm_group), GrowArrow(arrow_db_to_llm))
            self.wait(1.5)

            # Step by step explanation
            step1_text = Text("1. Retrieve: Search Vector DB for context.", font_size=20, color=BLACK).to_edge(DOWN).shift(UP*1)
            step2_text = Text("2. Augment: Combine context with prompt.", font_size=20, color=BLACK).next_to(step1_text, DOWN)
            step3_text = Text("3. Generate: LLM creates informed response.", font_size=20, color=BLACK).next_to(step2_text, DOWN)

            self.play(Write(step1_text))
            self.wait(1)
            self.play(Write(step2_text))
            self.wait(1)
            self.play(Write(step3_text))
            self.wait(1.5)

        self.play(FadeOut(VGroup(user_text, llm_group, db_group, arrow_user_to_db, arrow_db_to_llm, step1_text, step2_text, step3_text)))

        # Section 2: Why do we need it?
        with self.voiceover(text="Why do we actually need RAG? Large language models have three major limitations when used on their own. First, their training data has a strict cutoff date, meaning they don't know about recent events. Second, they lack access to private or proprietary company data. Third, they are prone to hallucinations, confidently making up facts when they don't know the answer. Let's see this in action.") as tracker:
            why_title = Text("Why do we need RAG?", font_size=36, color=BLACK).next_to(title, DOWN, buff=0.5)
            self.play(FadeIn(why_title))
            self.wait(1.5)

        with self.voiceover(text="Before RAG, if you asked a model a specific question about your company's internal policy, the model would only rely on its pre-trained weights. Because it hasn't seen your private data, it might hallucinate an incorrect answer or simply say 'I don't know'.") as tracker:
            before_text = Text("Before RAG (No Context)", font_size=28, color=RED).shift(UP * 1)
            self.play(Write(before_text))

            q1 = Text("Q: What is our 2024 remote work policy?", font_size=24, color=BLACK).next_to(before_text, DOWN)
            a1 = Text("LLM: I don't have access to internal documents.", font_size=24, color=RED).next_to(q1, DOWN)
            self.play(FadeIn(q1))
            self.wait(1)
            self.play(FadeIn(a1))
            self.wait(2)
            self.play(FadeOut(VGroup(before_text, q1, a1)))

        with self.voiceover(text="After introducing RAG, the system first retrieves the exact internal document from a secure database. It then injects this document into the prompt alongside your original question. Now, the LLM reads the retrieved policy and generates a highly accurate, factual response based exclusively on that exact source material. This grounds the model in truth and prevents hallucination.") as tracker:
            after_text = Text("After RAG (With Context)", font_size=28, color=GREEN).shift(UP * 1)
            self.play(Write(after_text))

            context_box = Rectangle(width=6, height=1.5, color=GREEN).next_to(after_text, DOWN)
            context_label = Text("Context: [Document 42: Employees may work remote 3 days/week]", font_size=20, color=BLACK).move_to(context_box)
            self.play(FadeIn(context_box), Write(context_label))

            a2 = Text("LLM: Employees can work remotely 3 days a week.", font_size=24, color=GREEN).next_to(context_box, DOWN)
            self.play(FadeIn(a2))
            self.wait(2)
            self.play(FadeOut(VGroup(why_title, after_text, context_box, context_label, a2)))

        # Section 3: Use Cases
        with self.voiceover(text="Now let's look at two massive real-world applications of RAG. How are top companies using this pattern in production today?") as tracker:
            use_cases_title = Text("Real-World Use Cases", font_size=36, color=BLACK).next_to(title, DOWN, buff=0.5)
            self.play(FadeIn(use_cases_title))
            self.wait(1.5)

        with self.voiceover(text="First, consider customer support chatbots. Companies like Shopify use RAG to power their merchant assistants. When a store owner asks how to set up a specific shipping rate, the bot queries Shopify's massive documentation database, retrieves the exact tutorial, and formulates a step-by-step response. This ensures merchants get accurate, up-to-date help instantly without waiting for a human agent.") as tracker:
            case1_title = Text("1. Customer Support (e.g., Shopify)", font_size=28, color=BLUE).shift(UP * 0.5)
            case1_desc = Text("Retrieves up-to-date documentation to answer merchant queries.", font_size=22, color=BLACK).next_to(case1_title, DOWN)
            self.play(Write(case1_title))
            self.play(FadeIn(case1_desc))
            self.wait(1.5)

        with self.voiceover(text="Second, consider enterprise search and legal analysis. Law firms use tools powered by RAG to query thousands of historical case files. Instead of a lawyer reading through hundreds of PDFs to find a precedent, they ask the AI. The RAG system retrieves the top 5 most similar case documents from a Vector Database, and the LLM summarizes the exact legal precedents relevant to their current case.") as tracker:
            case2_title = Text("2. Enterprise Search (Legal/Medical)", font_size=28, color=BLUE).next_to(case1_desc, DOWN, buff=0.8)
            case2_desc = Text("Searches thousands of PDFs/case files to synthesize facts instantly.", font_size=22, color=BLACK).next_to(case2_title, DOWN)
            self.play(Write(case2_title))
            self.play(FadeIn(case2_desc))
            self.wait(2)

            self.play(FadeOut(VGroup(use_cases_title, case1_title, case1_desc, case2_title, case2_desc)))

        # Section 4: Key Interview Insight
        with self.voiceover(text="Finally, let's discuss the most important part: the key tradeoff that interviewers will expect you to understand. While RAG drastically improves accuracy, it comes at a significant cost.") as tracker:
            insight_title = Text("Key Interview Insight", font_size=36, color=PURPLE).next_to(title, DOWN, buff=0.5)
            self.play(FadeIn(insight_title))
            self.wait(1)

        with self.voiceover(text="The biggest tradeoff in RAG systems is Latency versus Accuracy. When you introduce a Vector Database lookup step before calling the LLM, you add network delay. The retrieval process takes time. Furthermore, because you are injecting large chunks of retrieved text into the LLM's prompt, the context window grows significantly. Larger context windows increase inference time and computational cost.") as tracker:
            insight_box = Rectangle(width=10, height=4, color=PURPLE, stroke_width=4).shift(DOWN * 0.5)
            insight_heading = Text("The Tradeoff: Latency vs. Context Size", font_size=30, color=BLACK).move_to(insight_box).shift(UP * 1.2)

            point1 = Text("- Network delay from Vector DB lookup.", font_size=24, color=BLACK).next_to(insight_heading, DOWN, aligned_edge=LEFT).shift(DOWN * 0.2)
            point2 = Text("- Larger prompts increase LLM inference time and cost.", font_size=24, color=BLACK).next_to(point1, DOWN, aligned_edge=LEFT)
            point3 = Text("- Need to optimize chunk sizes and retrieval speed.", font_size=24, color=BLACK).next_to(point2, DOWN, aligned_edge=LEFT)

            self.play(FadeIn(insight_box), Write(insight_heading))
            self.wait(1)
            self.play(FadeIn(point1))
            self.wait(1)
            self.play(FadeIn(point2))
            self.wait(1)
            self.play(FadeIn(point3))
            self.wait(2)

        with self.voiceover(text="If an interviewer asks how you would speed up a slow RAG pipeline, you must discuss strategies like semantic caching, asynchronous retrieval, optimizing embedding dimensions, and refining document chunking strategies. Understanding these architectural bottlenecks separates a junior developer from a senior AI Engineer.") as tracker:
            self.wait(1.5)

        with self.voiceover(text="That concludes our deep dive into the RAG flow. You now understand what it is, why it solves the hallucination problem, how real companies use it, and the critical architectural tradeoffs involved. See you in the next session.") as tracker:
            self.play(FadeOut(VGroup(insight_title, insight_box, insight_heading, point1, point2, point3)))
            final_text = Text("Mastering RAG is essential for production AI.", font_size=32, color=BLUE)
            self.play(Write(final_text))
            self.wait(2)
            self.play(FadeOut(final_text), FadeOut(title))
            self.wait(1)
