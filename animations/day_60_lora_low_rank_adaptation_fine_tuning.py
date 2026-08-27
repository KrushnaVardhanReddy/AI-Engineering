from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class LoRAScene(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        self.section_1_what_is_it()
        self.section_2_why_need_it()
        self.section_3_use_cases()
        self.section_4_interview_insight()

    def section_1_what_is_it(self):
        # What is it? — A clear 1-sentence definition (spoken + shown on screen).
        title = Text("What is LoRA?", font_size=48, color=BLACK, weight=BOLD)
        title.to_edge(UP)

        definition = Text(
            "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique\n"
            "that freezes pre-trained model weights and injects trainable rank decomposition\n"
            "matrices into each layer of the Transformer architecture.",
            font_size=28, color=BLACK, t2c={"LoRA (Low-Rank Adaptation)": BLUE, "parameter-efficient": GREEN, "rank decomposition": RED}
        )
        definition.next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="What is LoRA? LoRA, or Low-Rank Adaptation, is a parameter-efficient fine-tuning technique. It freezes the original pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture.") as tracker:
            self.play(Write(title))
            self.play(FadeIn(definition, shift=UP))

        self.wait(1.5)

        # Visual Breakdown of equations
        # Standard Fine-tuning
        standard_text = Text("Standard Fine-Tuning:", font_size=32, color=BLACK)
        standard_text.shift(UP * 0.5 + LEFT * 3.5)

        standard_eq = MathTex(r"W", r"=", r"W_0", r"+", r"\Delta W", color=BLACK, font_size=44)
        standard_eq.next_to(standard_text, DOWN, buff=0.5)

        standard_subtext = Text("Train huge matrix!", font_size=24, color=RED).next_to(standard_eq, DOWN)

        with self.voiceover(text="In standard fine-tuning, to adapt a model, we update the entire weight matrix. So the new weights W equal the original weights W zero, plus the learned updates, Delta W. The problem? Delta W is just as massive as the original matrix, making training extremely expensive.") as tracker:
            self.play(Write(standard_text))
            self.play(Write(standard_eq))
            self.play(FadeIn(standard_subtext))

        self.wait(1.5)

        # LoRA Fine-tuning
        lora_text = Text("LoRA Fine-Tuning:", font_size=32, color=BLACK)
        lora_text.shift(UP * 0.5 + RIGHT * 3.5)

        lora_eq = MathTex(r"W", r"=", r"W_0", r"+", r"B", r"A", color=BLACK, font_size=44)
        lora_eq.next_to(lora_text, DOWN, buff=0.5)
        lora_eq[4].set_color(BLUE)
        lora_eq[5].set_color(GREEN)

        lora_subtext = Text("Train tiny matrices!", font_size=24, color=GREEN).next_to(lora_eq, DOWN)

        with self.voiceover(text="LoRA takes a smarter approach. We freeze W zero, and instead of learning the massive Delta W, we represent the update as the product of two much smaller matrices, B and A. This drastically reduces the number of parameters we actually need to train.") as tracker:
            self.play(Write(lora_text))

            # Transform Delta W to B A
            self.play(
                TransformMatchingTex(standard_eq.copy(), lora_eq, path_arc=90 * DEGREES)
            )
            self.play(FadeIn(lora_subtext))

        self.wait(1.5)
        self.play(FadeOut(VGroup(title, definition, standard_text, standard_eq, standard_subtext, lora_text, lora_eq, lora_subtext)))

    def section_2_why_need_it(self):
        # Why do we need it? — The specific problem it solves, with a "before vs. after" visual if applicable
        title = Text("Why do we need LoRA?", font_size=48, color=BLACK, weight=BOLD)
        title.to_edge(UP)

        with self.voiceover(text="So, why exactly do we need LoRA? The core issue is the massive memory footprint required to fine-tune modern Large Language Models.") as tracker:
            self.play(Write(title))

        # Before (Without LoRA)
        before_text = Text("Before (Full Fine-Tuning)", font_size=32, color=BLACK, weight=BOLD)
        before_text.shift(UP * 1.5 + LEFT * 3.5)

        gpu_box = Rectangle(width=3, height=4, color=RED, fill_opacity=0.1)
        gpu_box.next_to(before_text, DOWN, buff=0.5)

        mem_fill = Rectangle(width=3, height=3.5, color=RED, fill_opacity=0.6)
        mem_fill.move_to(gpu_box.get_bottom(), aligned_edge=DOWN)

        mem_text = Text("VRAM: Full\nParameters\n+ Optimizer States\n+ Gradients", font_size=20, color=BLACK)
        mem_text.move_to(gpu_box.get_center())

        with self.voiceover(text="Without LoRA, during full fine-tuning, you must load the entire model into GPU memory. But that's not all. You also need space for optimizer states, gradients, and activations for every single parameter. For a 70 billion parameter model, this requires clusters of highly expensive GPUs just to fit it in memory.") as tracker:
            self.play(FadeIn(before_text))
            self.play(Create(gpu_box))
            self.play(GrowFromEdge(mem_fill, DOWN))
            self.play(Write(mem_text))

        self.wait(1.5)

        # After (With LoRA)
        after_text = Text("After (With LoRA)", font_size=32, color=BLACK, weight=BOLD)
        after_text.shift(UP * 1.5 + RIGHT * 3.5)

        gpu_box_after = Rectangle(width=3, height=4, color=GREEN, fill_opacity=0.1)
        gpu_box_after.next_to(after_text, DOWN, buff=0.5)

        mem_fill_base = Rectangle(width=3, height=1.5, color=GRAY, fill_opacity=0.3)
        mem_fill_base.move_to(gpu_box_after.get_bottom(), aligned_edge=DOWN)

        mem_fill_lora = Rectangle(width=3, height=0.5, color=GREEN, fill_opacity=0.6)
        mem_fill_lora.next_to(mem_fill_base, UP, buff=0)

        mem_text_base = Text("Base Model\n(Frozen)", font_size=20, color=BLACK)
        mem_text_base.move_to(mem_fill_base.get_center())

        mem_text_lora = Text("LoRA Adapter\n(Trainable)", font_size=20, color=BLACK)
        mem_text_lora.move_to(mem_fill_lora.get_center())

        with self.voiceover(text="With LoRA, the massive base model is frozen and only requires memory for its static weights. We only train the tiny A and B matrices, meaning we only need optimizer states and gradients for a fraction of a percent of the total parameters. This allows you to fine-tune massive models on a single consumer-grade GPU.") as tracker:
            self.play(FadeIn(after_text))
            self.play(Create(gpu_box_after))
            self.play(GrowFromEdge(mem_fill_base, DOWN))
            self.play(Write(mem_text_base))
            self.play(GrowFromEdge(mem_fill_lora, DOWN))
            self.play(Write(mem_text_lora))

        self.wait(1.5)
        self.play(FadeOut(VGroup(title, before_text, gpu_box, mem_fill, mem_text, after_text, gpu_box_after, mem_fill_base, mem_fill_lora, mem_text_base, mem_text_lora)))

    def section_3_use_cases(self):
        # Use Cases — 2 real-world examples
        title = Text("Real-World Use Cases", font_size=48, color=BLACK, weight=BOLD)
        title.to_edge(UP)

        with self.voiceover(text="LoRA has become the industry standard for efficiently adapting foundation models. Let's look at two prominent real-world use cases.") as tracker:
            self.play(Write(title))

        # Use Case 1
        case1_title = Text("1. OpenAI / ChatGPT (Multi-Tenant Serving)", font_size=32, color=BLACK, weight=BOLD)
        case1_title.shift(UP * 1.5)

        server_box = Rectangle(width=8, height=1.5, color=BLUE, fill_opacity=0.1)
        server_box.next_to(case1_title, DOWN, buff=0.5)

        base_model = Text("Shared Base LLM", font_size=28, color=BLACK)
        base_model.move_to(server_box.get_left() + RIGHT * 2)

        user1 = Text("+ User A LoRA", font_size=24, color=GREEN).move_to(server_box.get_center() + UP * 0.3 + RIGHT * 1)
        user2 = Text("+ User B LoRA", font_size=24, color=PURPLE).move_to(server_box.get_center() + DOWN * 0.3 + RIGHT * 1)

        with self.voiceover(text="First, consider platforms like OpenAI's ChatGPT when offering custom GPTs. Instead of hosting thousands of massive 100-gigabyte models for every user, they host one single frozen base model in memory. When a user queries their custom GPT, the system simply loads their tiny, megabyte-sized LoRA adapter on the fly. This enables massive scale multi-tenant serving.") as tracker:
            self.play(Write(case1_title))
            self.play(Create(server_box))
            self.play(Write(base_model))
            self.play(FadeIn(user1, shift=LEFT))
            self.play(FadeIn(user2, shift=LEFT))

        self.wait(1.5)

        # Use Case 2
        case2_title = Text("2. Midjourney / Stable Diffusion (Style Adaptation)", font_size=32, color=BLACK, weight=BOLD)
        case2_title.next_to(server_box, DOWN, buff=1.0)

        art_box = Rectangle(width=8, height=1.5, color=PURPLE, fill_opacity=0.1)
        art_box.next_to(case2_title, DOWN, buff=0.5)

        sd_model = Text("Base Image Model", font_size=28, color=BLACK)
        sd_model.move_to(art_box.get_left() + RIGHT * 2)

        anime_lora = Text("+ Anime Style LoRA", font_size=24, color=RED).move_to(art_box.get_center() + UP * 0.3 + RIGHT * 1)
        pixel_lora = Text("+ Pixel Art LoRA", font_size=24, color=BLUE).move_to(art_box.get_center() + DOWN * 0.3 + RIGHT * 1)

        with self.voiceover(text="Second, in the generative AI art space, tools like Stable Diffusion and Midjourney rely heavily on LoRA. The community creates thousands of LoRA files that teach the base model specific character designs or artistic styles, like anime or pixel art. Users can simply download these tiny files and plug them into their local UI to instantly adapt the model's output.") as tracker:
            self.play(Write(case2_title))
            self.play(Create(art_box))
            self.play(Write(sd_model))
            self.play(FadeIn(anime_lora, shift=LEFT))
            self.play(FadeIn(pixel_lora, shift=LEFT))

        self.wait(1.5)
        self.play(FadeOut(VGroup(title, case1_title, server_box, base_model, user1, user2, case2_title, art_box, sd_model, anime_lora, pixel_lora)))

    def section_4_interview_insight(self):
        # Key Interview Insight — The most common "gotcha" or tradeoff interviewers test
        title = Text("Key Interview Insight", font_size=48, color=RED, weight=BOLD)
        title.to_edge(UP)

        with self.voiceover(text="Now for the most important part. If you are asked about LoRA in an AI engineering interview, there are two major concepts they are testing you on.") as tracker:
            self.play(Write(title))

        # Insight Box
        box = Rectangle(width=12, height=5.5, color=RED, fill_opacity=0.05)
        box.next_to(title, DOWN, buff=0.5)

        # Point 1
        point1_title = Text("1. The Rank 'r' Tradeoff", font_size=32, color=BLACK, weight=BOLD)
        point1_title.move_to(box.get_top() + DOWN * 0.8 + LEFT * 2.5)

        matrix_a = Rectangle(width=0.5, height=2, color=BLUE).next_to(point1_title, DOWN, buff=0.5)
        matrix_a_label = Text("A", font_size=24, color=BLACK).move_to(matrix_a.get_center())

        matrix_b = Rectangle(width=2, height=0.5, color=GREEN).next_to(matrix_a, RIGHT, buff=0.2)
        matrix_b_label = Text("B", font_size=24, color=BLACK).move_to(matrix_b.get_center())

        r_label = Text("Rank (r)", font_size=24, color=RED).next_to(matrix_a, DOWN)

        with self.voiceover(text="First is the tradeoff regarding the rank parameter, 'r'. The rank dictates the inner dimension of the A and B matrices. A higher rank means the model can learn more complex patterns and capture more information, but it increases the parameter count, memory usage, and risk of overfitting. A lower rank is more efficient but might underfit the new data. You must balance capacity versus efficiency.") as tracker:
            self.play(Create(box))
            self.play(Write(point1_title))
            self.play(Create(matrix_a), Write(matrix_a_label))
            self.play(Create(matrix_b), Write(matrix_b_label))
            self.play(FadeIn(r_label, shift=UP))

        self.wait(1.5)

        # Point 2
        point2_title = Text("2. Zero Inference Latency", font_size=32, color=BLACK, weight=BOLD)
        point2_title.move_to(box.get_top() + DOWN * 0.8 + RIGHT * 2.5)

        eq1 = MathTex(r"W", r"=", r"W_0", r"+", r"B", r"A", color=BLACK, font_size=36)
        eq1.next_to(point2_title, DOWN, buff=0.5)

        eq2 = MathTex(r"W_{merged}", r"=", r"W_0 + BA", color=BLACK, font_size=36)
        eq2.next_to(eq1, DOWN, buff=0.5)
        eq2[0].set_color(PURPLE)

        with self.voiceover(text="The second key insight, and the biggest 'gotcha', is inference latency. An interviewer might ask: 'Doesn't calculating the extra B times A matrices slow down the model during generation?' The answer is no. Because matrix addition is linear, once training is complete, you can mathematically multiply B and A, and permanently add that result directly into the original W zero weights. This creates a merged weight matrix, meaning there is absolutely zero extra computational overhead during inference.") as tracker:
            self.play(Write(point2_title))
            self.play(Write(eq1))
            self.play(TransformMatchingTex(eq1.copy(), eq2, path_arc=-90 * DEGREES))

        self.wait(2.0)

        conclusion = Text("You've mastered LoRA fine-tuning!", font_size=36, color=BLUE, weight=BOLD)
        conclusion.next_to(eq2, DOWN, buff=1.0)

        with self.voiceover(text="Understand these tradeoffs and the mathematical merging property, and you'll easily pass any system design interview on parameter-efficient fine-tuning.") as tracker:
            self.play(Write(conclusion))

        self.wait(2.0)
        self.play(FadeOut(VGroup(title, box, point1_title, matrix_a, matrix_a_label, matrix_b, matrix_b_label, r_label, point2_title, eq1, eq2, conclusion)))
