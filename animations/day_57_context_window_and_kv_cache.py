from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class ContextWindowAndKVCache(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # Introduction
        title = Tex("Context Window \\& KV Cache", color=BLACK).scale(1.5)
        with self.voiceover(text="Welcome to day 57 of our AI engineering mastery series. Today, we are taking a deep dive into two of the most fundamental concepts that govern how modern large language models, like GPT-4 or Claude, process and generate text: the Context Window and the KV Cache.") as tracker:
            self.play(Write(title))
            self.wait(1.5)

        with self.voiceover(text="These two concepts dictate everything from how much information an AI can remember at any given moment, to how fast and efficiently it can stream responses back to you in real time.") as tracker:
            self.wait(1.5)

        self.play(FadeOut(title))

        # Part 1: What is it?
        section_1_title = Tex("1. What is it?", color=BLUE).to_edge(UP, buff=0.5)
        with self.voiceover(text="Let's start by defining what they are, starting with the Context Window.") as tracker:
            self.play(Write(section_1_title))

        cw_def = Tex(
            "\\textbf{Context Window:} The maximum amount of text \\\\ (tokens) the model can process at one time.",
            color=BLACK
        ).scale(0.8).next_to(section_1_title, DOWN, buff=1.0)

        with self.voiceover(text="The Context Window is simply the maximum amount of text, measured in tokens, that the model can hold in its working memory at one time. If you think of a model's weights as its long term memory—everything it learned during training—then the context window is its short term memory, containing your prompt and its ongoing response.") as tracker:
            self.play(Write(cw_def))
            self.wait(1.5)

        kv_def = Tex(
            "\\textbf{KV Cache:} A memory optimization technique that stores \\\\ previously computed Key (K) and Value (V) vectors.",
            color=BLACK
        ).scale(0.8).next_to(cw_def, DOWN, buff=0.8)

        with self.voiceover(text="The KV Cache, on the other hand, is a critical memory optimization technique used entirely during the text generation phase. It stores previously computed Key and Value vectors in the GPU's memory to avoid recalculating them for every new word.") as tracker:
            self.play(Write(kv_def))
            self.wait(1.5)

        self.play(FadeOut(cw_def, kv_def))

        # Show the Attention Math Step by Step
        math_title = Tex("The Attention Mechanism", color=BLACK).next_to(section_1_title, DOWN, buff=0.5)
        with self.voiceover(text="To really understand why the KV Cache exists, we have to look closely at the math behind the self-attention mechanism inside the Transformer architecture.") as tracker:
            self.play(FadeIn(math_title))
            self.wait(1.0)

        # Step-by-step math derivation
        eq_base = MathTex(
            r"\text{Attention}(", r"Q", r", ", r"K", r", ", r"V", r") =", color=BLACK
        ).scale(1.2).next_to(math_title, DOWN, buff=0.8)

        eq_base[1].set_color(RED)
        eq_base[3].set_color(BLUE)
        eq_base[5].set_color(GREEN)

        with self.voiceover(text="For every token processed, the model computes three distinct vectors: a Query vector, a Key vector, and a Value vector.") as tracker:
            self.play(Write(eq_base))
            self.wait(1.5)

        eq_step1 = MathTex(
            r"\text{Attention}(", r"Q", r", ", r"K", r", ", r"V", r") = ", r"Q", r"K", r"^T", color=BLACK
        ).scale(1.2).move_to(eq_base, aligned_edge=LEFT)
        eq_step1[1].set_color(RED)
        eq_step1[3].set_color(BLUE)
        eq_step1[5].set_color(GREEN)
        eq_step1[7].set_color(RED)
        eq_step1[8].set_color(BLUE)

        with self.voiceover(text="First, the Query vector searches for relevant information by computing the dot product with the Key vectors of all previous tokens.") as tracker:
            self.play(TransformMatchingTex(eq_base, eq_step1))
            self.wait(1.5)

        eq_step2 = MathTex(
            r"\text{Attention}(", r"Q", r", ", r"K", r", ", r"V", r") = ", r"\text{softmax}\left(\frac{", r"Q", r"K", r"^T}{\sqrt{d_k}}\right)", color=BLACK
        ).scale(1.2).move_to(eq_step1, aligned_edge=LEFT)
        eq_step2[1].set_color(RED)
        eq_step2[3].set_color(BLUE)
        eq_step2[5].set_color(GREEN)
        eq_step2[8].set_color(RED)
        eq_step2[9].set_color(BLUE)

        with self.voiceover(text="This result is scaled and passed through a softmax function to create attention scores, representing how much focus to give each token.") as tracker:
            self.play(TransformMatchingTex(eq_step1, eq_step2))
            self.wait(1.5)

        eq_final = MathTex(
            r"\text{Attention}(", r"Q", r", ", r"K", r", ", r"V", r") = ", r"\text{softmax}\left(\frac{", r"Q", r"K", r"^T}{\sqrt{d_k}}\right)", r"V", color=BLACK
        ).scale(1.2).move_to(eq_step2, aligned_edge=LEFT)
        eq_final[1].set_color(RED)
        eq_final[3].set_color(BLUE)
        eq_final[5].set_color(GREEN)
        eq_final[8].set_color(RED)
        eq_final[9].set_color(BLUE)
        eq_final[11].set_color(GREEN)

        with self.voiceover(text="Finally, these scores are multiplied by the Value vectors to produce the final context-aware representation. The crucial thing to notice here is that to compute this output, we need the Keys and Values for every single token that came before.") as tracker:
            self.play(TransformMatchingTex(eq_step2, eq_final))
            self.wait(2.0)

        self.play(FadeOut(math_title, eq_final, section_1_title))

        # Part 2: Why do we need it?
        section_2_title = Tex("2. Why do we need it?", color=RED).to_edge(UP, buff=0.5)
        with self.voiceover(text="So, why do we need a specialized cache for this? Let's look at how text generation works under the hood without it.") as tracker:
            self.play(Write(section_2_title))

        # Before KV Cache (O(N^2))
        before_title = Tex("\\textbf{Without KV Cache}", color=BLACK).scale(0.9).next_to(section_2_title, DOWN, buff=0.5).to_edge(LEFT, buff=1)

        tokens_text = ["The", "cat", "sat", "on"]
        boxes = VGroup()
        for text in tokens_text:
            box = Rectangle(width=1.2, height=0.8, color=BLACK)
            label = Tex(text, color=BLACK).move_to(box)
            boxes.add(VGroup(box, label))

        boxes.arrange(RIGHT, buff=0.2).next_to(before_title, DOWN, buff=0.5, aligned_edge=LEFT)

        with self.voiceover(text="Imagine we have the input prompt: 'The cat sat on', and the model is trying to predict the next word in the sequence.") as tracker:
            self.play(FadeIn(before_title))
            self.play(Write(boxes))
            self.wait(1.0)

        # Show recomputation
        qkv_calc = VGroup()
        for i in range(len(boxes)):
            calc = Tex(f"$Q_{i}, K_{i}, V_{i}$", color=PURPLE).scale(0.6).next_to(boxes[i], DOWN, buff=0.2)
            qkv_calc.add(calc)

        with self.voiceover(text="During generation, models predict one token at a time. Without a cache, when it wants to generate the fifth word, the model runs the entire attention mechanism over again. It recomputes the Query, Key, and Value vectors for 'The', for 'cat', for 'sat', and for 'on' entirely from scratch.") as tracker:
            self.play(Write(qkv_calc))
            self.wait(1.5)

        new_token = VGroup(
            Rectangle(width=1.2, height=0.8, color=RED),
            Tex("the", color=RED)
        ).arrange(RIGHT, buff=0).move_to(boxes[-1].get_center() + RIGHT * 1.4)
        new_token[1].move_to(new_token[0])

        with self.voiceover(text="This results in a massive amount of redundant, wasted computation. Every single time it generates a new word, it has to reprocess the entire history.") as tracker:
            self.play(FadeIn(new_token))
            self.wait(1.5)

        with self.voiceover(text="For a sequence of length N, this means the time complexity of generation scales quadratically, becoming O of N squared. This is impossibly slow for long documents.") as tracker:
            complexity_1 = MathTex(r"\mathcal{O}(N^2)", color=RED).next_to(qkv_calc, DOWN, buff=0.5).align_to(boxes, LEFT)
            self.play(Write(complexity_1))
            self.wait(1.5)

        self.play(FadeOut(before_title, boxes, qkv_calc, new_token, complexity_1))

        # After KV Cache (O(N))
        after_title = Tex("\\textbf{With KV Cache}", color=BLACK).scale(0.9).next_to(section_2_title, DOWN, buff=0.5).to_edge(LEFT, buff=1)

        boxes_after = VGroup()
        for text in tokens_text:
            box = Rectangle(width=1.2, height=0.8, color=BLACK)
            label = Tex(text, color=BLACK).move_to(box)
            boxes_after.add(VGroup(box, label))

        boxes_after.arrange(RIGHT, buff=0.2).next_to(after_title, DOWN, buff=0.5, aligned_edge=LEFT)

        with self.voiceover(text="Now, let's see generation utilizing the KV Cache.") as tracker:
            self.play(FadeIn(after_title), Write(boxes_after))
            self.wait(1.0)

        cache_box = Rectangle(width=6, height=1.5, color=GREEN, fill_opacity=0.1).next_to(boxes_after, DOWN, buff=0.5).align_to(boxes_after, LEFT)
        cache_label = Tex("KV Cache", color=GREEN).next_to(cache_box, UP, buff=0.1).align_to(cache_box, LEFT)

        cached_kvs = VGroup()
        for i in range(len(boxes_after)):
            kv = Tex(f"$K_{i}, V_{i}$", color=GREEN).scale(0.6)
            cached_kvs.add(kv)
        cached_kvs.arrange(RIGHT, buff=0.7).move_to(cache_box)

        with self.voiceover(text="During the initial prompt processing phase, known as the prefill phase, the model computes the Keys and Values for all the input tokens. Instead of throwing them away, we store them in a dedicated space in GPU memory called the KV Cache.") as tracker:
            self.play(Create(cache_box), Write(cache_label))
            self.play(Write(cached_kvs))
            self.wait(1.5)

        new_token_after = VGroup(
            Rectangle(width=1.2, height=0.8, color=BLUE),
            Tex("the", color=BLUE)
        ).arrange(RIGHT, buff=0).move_to(boxes_after[-1].get_center() + RIGHT * 1.4)
        new_token_after[1].move_to(new_token_after[0])

        q_new = Tex(f"$Q_{4}$", color=BLUE).scale(0.6).next_to(new_token_after, DOWN, buff=0.2)
        kv_new = Tex(f"$K_{4}, V_{4}$", color=GREEN).scale(0.6).next_to(q_new, DOWN, buff=0.4)

        with self.voiceover(text="When generating the next word, we only need to pass the single new token through the model to compute its Query, Key, and Value vectors.") as tracker:
            self.play(FadeIn(new_token_after))
            self.play(Write(q_new))
            self.wait(1.0)

        with self.voiceover(text="This single new Query simply attends to the saved Keys and Values waiting in the cache. Once the new word is generated, we append its newly computed Key and Value to the cache for the next time step.") as tracker:
            arrow = Arrow(q_new.get_left(), cache_box.get_right(), color=BLUE, buff=0.1)
            self.play(GrowArrow(arrow))
            self.play(Write(kv_new))
            self.wait(1.0)

        with self.voiceover(text="This fundamentally alters the performance characteristics of text generation. It brings the compute complexity down to O of N per step, making the fast, streaming responses we are used to possible.") as tracker:
            complexity_2 = MathTex(r"\mathcal{O}(N)", color=GREEN).next_to(cache_box, DOWN, buff=0.5).align_to(cache_box, LEFT)
            self.play(Write(complexity_2))
            self.wait(1.5)

        self.play(FadeOut(after_title, boxes_after, cache_box, cache_label, cached_kvs, new_token_after, q_new, kv_new, arrow, complexity_2, section_2_title))

        # Part 3: Use Cases
        section_3_title = Tex("3. Use Cases", color=GREEN).to_edge(UP, buff=0.5)
        with self.voiceover(text="Where does the size of the context window and the efficiency of the cache matter in the real world?") as tracker:
            self.play(Write(section_3_title))

        uc1_title = Tex("\\textbf{1. ChatGPT (Long Conversations)}", color=BLACK).scale(0.8).next_to(section_3_title, DOWN, buff=1.0).to_edge(LEFT, buff=1)
        uc1_desc = Tex(
            "Maintains context over thousands of tokens. \\\\ The KV cache ensures the system responds quickly \\\\ without recalculating the entire history.",
            color=DARK_GRAY
        ).scale(0.7).next_to(uc1_title, DOWN, buff=0.3, aligned_edge=LEFT)

        with self.voiceover(text="First, consider an application like ChatGPT maintaining a long, ongoing conversation. Over time, it holds thousands of tokens in its context window. The KV cache is what allows the model to reply to your latest message instantly, without completely recalculating the entire chat history for every single word it types back.") as tracker:
            self.play(Write(uc1_title))
            self.play(Write(uc1_desc))
            self.wait(1.5)

        uc2_title = Tex("\\textbf{2. GitHub Copilot (Code Generation)}", color=BLACK).scale(0.8).next_to(uc1_desc, DOWN, buff=0.8).to_edge(LEFT, buff=1)
        uc2_desc = Tex(
            "Reads large chunks of a repository (e.g., 100k tokens). \\\\ The cache allows instant streaming of code completion \\\\ based on a massive context window.",
            color=DARK_GRAY
        ).scale(0.7).next_to(uc2_title, DOWN, buff=0.3, aligned_edge=LEFT)

        with self.voiceover(text="Second, consider AI coding assistants like GitHub Copilot or Cursor. These tools often pass massive chunks of your entire repository into the context window, sometimes hundreds of thousands of tokens at once. The KV cache makes it feasible to perform instant, streaming code completion based on that massive context.") as tracker:
            self.play(Write(uc2_title))
            self.play(Write(uc2_desc))
            self.wait(1.5)

        self.play(FadeOut(uc1_title, uc1_desc, uc2_title, uc2_desc, section_3_title))

        # Part 4: Key Interview Insight
        section_4_title = Tex("4. Key Interview Insight", color=PURPLE).to_edge(UP, buff=0.5)
        with self.voiceover(text="Finally, let's look at the most critical architectural insight and the most common interview question on this topic.") as tracker:
            self.play(Write(section_4_title))

        insight_text = Tex("\\textbf{The Tradeoff: Compute vs. Memory}", color=BLACK).scale(0.9)
        insight_box = SurroundingRectangle(
            insight_text,
            color=PURPLE, buff=0.3, corner_radius=0.2
        ).next_to(section_4_title, DOWN, buff=0.5)
        insight_text.move_to(insight_box)

        with self.voiceover(text="The KV cache creates a fundamental systems design tradeoff. We are actively trading compute cycles for memory consumption.") as tracker:
            self.play(Create(insight_box))
            self.play(Write(insight_text))
            self.wait(1.5)

        with self.voiceover(text="While we dramatically speed up calculation time, storing all those millions of vectors takes up a massive amount of VRAM on the GPU. Because of this, LLM inference in production is almost always memory-bound, not compute-bound.") as tracker:
            self.wait(1.5)

        formula_title = Tex("KV Cache Memory Formula:", color=BLACK).scale(0.8).next_to(insight_box, DOWN, buff=0.8)

        # Memory = 2 * batch * seq_len * layers * heads * d_k
        form_base = MathTex(
            r"\text{Memory} = ", color=BLACK
        ).scale(0.7).next_to(formula_title, DOWN, buff=0.3).align_to(formula_title, LEFT)

        with self.voiceover(text="Let's build the formula to calculate the exact memory required for the KV Cache.") as tracker:
            self.play(Write(formula_title))
            self.play(Write(form_base))
            self.wait(1.0)

        form_step1 = MathTex(
            r"\text{Memory} = ", r"2", r"\times", r"\text{batch\_size}", color=BLACK
        ).scale(0.7).move_to(form_base, aligned_edge=LEFT)
        form_step1[1].set_color(RED)

        with self.voiceover(text="It starts with a factor of 2, representing both the Key and the Value vectors, multiplied by the batch size.") as tracker:
            self.play(TransformMatchingTex(form_base, form_step1))
            self.wait(1.0)

        form_step2 = MathTex(
            r"\text{Memory} = ", r"2", r"\times", r"\text{batch\_size}", r"\times", r"\text{seq\_len}", color=BLACK
        ).scale(0.7).move_to(form_step1, aligned_edge=LEFT)
        form_step2[1].set_color(RED)
        form_step2[5].set_color(BLUE)

        with self.voiceover(text="Next, we multiply by the sequence length—the total number of tokens in the context window.") as tracker:
            self.play(TransformMatchingTex(form_step1, form_step2))
            self.wait(1.0)

        form_final = MathTex(
            r"\text{Memory} = ", r"2", r"\times", r"\text{batch\_size}", r"\times", r"\text{seq\_len}", r"\times", r"\text{layers}", r"\times", r"\text{heads}", r"\times", r"d_k",
            color=BLACK
        ).scale(0.7).move_to(form_step2, aligned_edge=LEFT)
        form_final[1].set_color(RED)
        form_final[5].set_color(BLUE)

        with self.voiceover(text="Finally, we multiply by the number of transformer layers, the number of attention heads, and the dimension of each head.") as tracker:
            self.play(TransformMatchingTex(form_step2, form_final))
            self.wait(1.5)

        with self.voiceover(text="Notice that the memory scales linearly with sequence length. This means a massive 1 million token context window requires a proportionally massive amount of GPU memory just to store the cache.") as tracker:
            self.play(Indicate(form_final[5], color=BLUE, scale_factor=1.5))
            self.wait(1.5)

        opt_text = Tex("Optimizations: PagedAttention, GQA, MQA", color=DARK_GRAY).scale(0.7).next_to(form_final, DOWN, buff=0.8).align_to(form_final, LEFT)

        with self.voiceover(text="Because of this bottleneck, interviewers will expect you to know how modern AI infrastructure solves this issue. Techniques like Grouped Query Attention reduce the number of KV heads we need to store, and architectures like PagedAttention manage that memory dynamically to eliminate fragmentation.") as tracker:
            self.play(FadeIn(opt_text))
            self.wait(2.0)

        self.play(FadeOut(section_4_title, insight_box, insight_text, formula_title, form_final, opt_text))

        # Outro
        outro = Tex("Context Window \\& KV Cache Mastery", color=BLACK).scale(1.2)
        with self.voiceover(text="Understanding the delicate balance between the size of the context window and the memory requirements of the KV cache is essential for deploying large language models efficiently in production. I hope this deep dive helped, and I will see you in the next lesson!") as tracker:
            self.play(FadeIn(outro))
            self.wait(2.0)

        self.play(FadeOut(outro))
