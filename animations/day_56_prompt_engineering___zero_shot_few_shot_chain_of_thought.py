from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class PromptEngineeringScene(VoiceoverScene):
    def construct(self):
        # Setting aesthetic
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Main configuration
        text_color = BLACK
        accent_blue = BLUE_E
        accent_red = RED_E
        accent_green = GREEN_E

        # Helper for common text style
        def get_text(content, font_size=36, color=text_color):
            return Text(content, font_size=font_size, color=color)

        # ---------------------------------------------------------
        # Section 0: Title
        # ---------------------------------------------------------
        title = get_text("Prompt Engineering", font_size=56).to_edge(UP)
        subtitle = get_text("Zero-Shot, Few-Shot, and Chain-of-Thought", font_size=36, color=accent_blue).next_to(title, DOWN)

        with self.voiceover(text="Welcome to our deep dive into Prompt Engineering. Today, we will explore three powerful techniques for guiding Large Language Models: Zero-Shot, Few-Shot, and Chain-of-Thought prompting.") as tracker:
            self.play(Write(title))
            self.play(FadeIn(subtitle))
            self.wait(1.5)

        self.play(FadeOut(subtitle))
        self.play(title.animate.scale(0.7).to_edge(UP))

        # ---------------------------------------------------------
        # Section 1: Zero-Shot Prompting
        # ---------------------------------------------------------
        section1_title = get_text("1. Zero-Shot Prompting", font_size=42, color=accent_blue).next_to(title, DOWN, buff=0.5)

        # What is it?
        zs_def = get_text("Definition: Asking the model to perform a task without providing any examples.", font_size=30).next_to(section1_title, DOWN, buff=0.5)

        with self.voiceover(text="Let's start with Zero-Shot Prompting. What is it? It is the practice of asking a model to perform a task without providing any prior examples or demonstrations in the prompt.") as tracker:
            self.play(FadeIn(section1_title))
            self.play(Write(zs_def))
            self.wait(1.5)

        # Why do we need it?
        zs_box_prompt = Rectangle(width=6, height=2.5, color=text_color, fill_opacity=0.1).move_to(LEFT * 3 + DOWN * 1)
        zs_box_text = get_text("Prompt:\nClassify this review:\n'The food was terrible.'\nSentiment: ", font_size=24).move_to(zs_box_prompt.get_center())

        zs_box_response = Rectangle(width=4, height=1.5, color=accent_green, fill_opacity=0.1).move_to(RIGHT * 3 + DOWN * 1)
        zs_box_resp_text = get_text("Response:\nNegative", font_size=24, color=accent_green).move_to(zs_box_response.get_center())

        arrow1 = Arrow(zs_box_prompt.get_right(), zs_box_response.get_left(), color=text_color)

        with self.voiceover(text="Why do we need it? It allows for immediate, out-of-the-box usage. You simply state your instruction and the input. For instance, classifying a movie review's sentiment.") as tracker:
            self.play(Create(zs_box_prompt), Write(zs_box_text))
            self.wait(1.0)
            self.play(GrowArrow(arrow1))
            self.play(Create(zs_box_response), Write(zs_box_resp_text))
            self.wait(1.5)

        # Use cases
        zs_usecases = get_text("Use Cases:\n- ChatGPT for general queries\n- Zero-shot classification (e.g. HuggingFace pipelines)", font_size=28).next_to(zs_box_prompt, DOWN, buff=0.5).align_to(zs_box_prompt, LEFT)

        with self.voiceover(text="Common use cases include interacting with ChatGPT for general Q&A, or using zero-shot classification pipelines on HuggingFace for dynamically defined categories.") as tracker:
            self.play(Write(zs_usecases))
            self.wait(1.5)

        # Key Insight
        zs_insight_box = SurroundingRectangle(zs_usecases, color=accent_red, buff=0.3)
        zs_insight = get_text("Interview Insight:\nStruggles with complex formatting\nor domain-specific jargon.", font_size=28, color=accent_red).next_to(zs_insight_box, RIGHT, buff=1)

        with self.voiceover(text="The key interview insight here: while zero-shot is highly flexible, it often struggles if you require strict output formatting, or if the task involves highly domain-specific jargon it hasn't seen during pre-training.") as tracker:
            self.play(Create(zs_insight_box))
            self.play(Write(zs_insight))
            self.wait(1.5)

        # Cleanup for Section 2
        self.play(FadeOut(VGroup(zs_def, zs_box_prompt, zs_box_text, zs_box_response, zs_box_resp_text, arrow1, zs_usecases, zs_insight_box, zs_insight, section1_title)))

        # ---------------------------------------------------------
        # Section 2: Few-Shot Prompting
        # ---------------------------------------------------------
        section2_title = get_text("2. Few-Shot Prompting", font_size=42, color=accent_blue).next_to(title, DOWN, buff=0.5)

        # What is it?
        fs_def = get_text("Definition: Providing a few examples (shots) in the prompt to guide the model.", font_size=30).next_to(section2_title, DOWN, buff=0.5)

        with self.voiceover(text="Next is Few-Shot Prompting. What is it? It involves providing a small number of examples, or 'shots', directly within the prompt to guide the model's behavior and formatting.") as tracker:
            self.play(FadeIn(section2_title))
            self.play(Write(fs_def))
            self.wait(1.5)

        # Why do we need it? Before vs After
        fs_before_text = get_text("Zero-Shot:\nTranslate to French: 'Hello'\nOutput: 'Bonjour! Comment ca va?'\n(Too chatty)", font_size=24, color=accent_red).move_to(LEFT * 3.5 + DOWN * 0.5)

        fs_after_text = get_text("Few-Shot:\nEng: 'Apple' -> Fr: 'Pomme'\nEng: 'Dog' -> Fr: 'Chien'\nEng: 'Hello' -> Fr:\nOutput: 'Bonjour'", font_size=24, color=accent_green).move_to(RIGHT * 2.5 + DOWN * 0.5)

        vs_text = get_text("VS", font_size=36, color=text_color).move_to(DOWN * 0.5)

        with self.voiceover(text="Why do we need it? Look at the difference. Without examples, the model might be too chatty or use the wrong format. With a few examples, we teach the model the exact input-output pattern we expect, forcing it to be concise.") as tracker:
            self.play(Write(fs_before_text))
            self.wait(1.0)
            self.play(Write(vs_text))
            self.play(Write(fs_after_text))
            self.wait(1.5)

        # Use Cases
        fs_usecases = get_text("Use Cases:\n- GitHub Copilot (using surrounding code as context)\n- Formatting strict JSON outputs", font_size=28).next_to(fs_before_text, DOWN, buff=1).align_to(fs_before_text, LEFT)

        with self.voiceover(text="Real-world use cases include GitHub Copilot, which uses surrounding code as few-shot context to match your style, or enterprise systems that require strictly formatted JSON outputs.") as tracker:
            self.play(Write(fs_usecases))
            self.wait(1.5)

        # Key Insight
        fs_insight_box = SurroundingRectangle(fs_usecases, color=accent_red, buff=0.3)
        fs_insight = get_text("Interview Insight:\nContext Window Limits.\nToo many shots = higher cost\nand potential truncation.", font_size=28, color=accent_red).next_to(fs_insight_box, RIGHT, buff=1)

        with self.voiceover(text="The critical interview insight: Few-shot prompting consumes the context window. Providing too many examples increases latency, raises API costs, and risks truncating the actual user input.") as tracker:
            self.play(Create(fs_insight_box))
            self.play(Write(fs_insight))
            self.wait(1.5)

        # Cleanup for Section 3
        self.play(FadeOut(VGroup(fs_def, fs_before_text, fs_after_text, vs_text, fs_usecases, fs_insight_box, fs_insight, section2_title)))

        # ---------------------------------------------------------
        # Section 3: Chain-of-Thought (CoT) Prompting
        # ---------------------------------------------------------
        section3_title = get_text("3. Chain-of-Thought (CoT) Prompting", font_size=42, color=accent_blue).next_to(title, DOWN, buff=0.5)

        # What is it?
        cot_def = get_text("Definition: Instructing the model to generate intermediate reasoning steps.", font_size=30).next_to(section3_title, DOWN, buff=0.5)

        with self.voiceover(text="Finally, we have Chain-of-Thought, or CoT, prompting. What is it? It is the technique of instructing the model to generate intermediate reasoning steps before arriving at a final answer.") as tracker:
            self.play(FadeIn(section3_title))
            self.play(Write(cot_def))
            self.wait(1.5)

        # Why do we need it? Math Derivation
        math_prompt = get_text("Q: I have 3 apples. I buy 2 more packs of 4. How many total?", font_size=26).next_to(cot_def, DOWN, buff=0.5)

        cot_step1 = MathTex(r"\text{Total} = 3 + (2 \times 4)", color=text_color).next_to(math_prompt, DOWN, buff=0.5)
        cot_step2 = MathTex(r"\text{Total} = 3 + 8", color=text_color).next_to(math_prompt, DOWN, buff=0.5)
        cot_step3 = MathTex(r"\text{Total} = 11", color=accent_green).next_to(math_prompt, DOWN, buff=0.5)

        with self.voiceover(text="Why do we need it? Standard prompting often fails on complex reasoning or math. By forcing the model to 'think out loud', we give it compute space to solve the problem step-by-step.") as tracker:
            self.play(Write(math_prompt))
            self.wait(1.0)
            self.play(Write(cot_step1))
            self.wait(1.0)
            self.play(TransformMatchingTex(cot_step1, cot_step2))
            self.wait(1.0)
            self.play(TransformMatchingTex(cot_step2, cot_step3))
            self.wait(1.5)

        # Use cases
        cot_usecases = get_text("Use Cases:\n- Google's PaLM solving math word problems\n- AI Agents planning tool usage (e.g. LangChain)", font_size=28).next_to(cot_step3, DOWN, buff=0.5)

        with self.voiceover(text="Prominent use cases include Google using CoT in their PaLM models to achieve state-of-the-art results on math benchmarks, and AI agents in frameworks like LangChain, which use reasoning steps to decide which tools to call.") as tracker:
            self.play(Write(cot_usecases))
            self.wait(1.5)

        # Key Insight
        cot_insight_bg = Rectangle(width=9, height=2, color=accent_red, fill_opacity=0.1).to_edge(DOWN, buff=0.5)
        cot_insight = get_text("Interview Insight:\n'Let's think step by step' (Zero-shot CoT)\nsignificantly improves performance, but increases\nTime-to-First-Byte (TTFB) and token costs.", font_size=28, color=accent_red).move_to(cot_insight_bg.get_center())

        with self.voiceover(text="The essential interview insight: Adding a simple phrase like 'Let's think step by step' activates zero-shot Chain-of-Thought, drastically improving accuracy. However, the trade-off is higher token costs and increased latency, specifically Time-to-First-Byte for the final answer.") as tracker:
            self.play(Create(cot_insight_bg))
            self.play(Write(cot_insight))
            self.wait(1.5)

        # Outro
        with self.voiceover(text="Understanding when to use Zero-Shot, Few-Shot, and Chain of Thought is crucial for building robust AI systems. Thank you for watching.") as tracker:
            self.wait(2)
            self.play(
                FadeOut(cot_def),
                FadeOut(math_prompt),
                FadeOut(cot_step3),
                FadeOut(cot_usecases),
                FadeOut(cot_insight_bg),
                FadeOut(cot_insight),
                FadeOut(section3_title),
                FadeOut(title)
            )
            self.wait(1)
