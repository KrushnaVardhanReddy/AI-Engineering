from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class ModelQuantizationScene(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # Introduction & What is it?
        title = Text("Model Quantization: INT8 and INT4", font_size=48, color=BLACK)
        title.to_edge(UP)

        with self.voiceover(text="Welcome to Day 59 of our AI Engineering Mastery series. Today, we are diving deep into Model Quantization, specifically focusing on INT8 and INT4 precision formats for production environments. As Large Language Models continue to grow in size, deploying them efficiently has become one of the most critical challenges in the industry. Let's explore how quantization solves this.") as tracker:
            self.play(Write(title))
            self.wait(1.5)

        def_text = Text(
            "What is it? Model Quantization is the process of mapping continuous \n"
            "high-precision floating-point numbers into lower-precision discrete integer formats.",
            font_size=28, color=BLACK
        )
        def_text.next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="So, what is Model Quantization? In a single sentence, it is the process of mapping continuous, high-precision floating-point numbers—usually 32-bit or 16-bit—into lower-precision, discrete integer formats, such as 8-bit or 4-bit integers. By doing this, we drastically reduce the memory footprint and computational requirements of the model.") as tracker:
            self.play(FadeIn(def_text, shift=DOWN))
            self.wait(1.5)

        # Visualizing What is it
        fp32_box = Rectangle(width=2, height=1, color=BLUE, fill_opacity=0.2)
        fp32_label = Text("FP32", font_size=32, color=BLUE).move_to(fp32_box)
        fp32_group = VGroup(fp32_box, fp32_label).shift(LEFT * 4 + DOWN * 1)

        arrow1 = Arrow(start=LEFT, end=RIGHT, color=BLACK).next_to(fp32_group, RIGHT)

        int8_box = Rectangle(width=1, height=1, color=GREEN, fill_opacity=0.2)
        int8_label = Text("INT8", font_size=32, color=GREEN).move_to(int8_box)
        int8_group = VGroup(int8_box, int8_label).next_to(arrow1, RIGHT)

        arrow2 = Arrow(start=LEFT, end=RIGHT, color=BLACK).next_to(int8_group, RIGHT)

        int4_box = Rectangle(width=0.5, height=1, color=PURPLE, fill_opacity=0.2)
        int4_label = Text("INT4", font_size=24, color=PURPLE).move_to(int4_box)
        int4_group = VGroup(int4_box, int4_label).next_to(arrow2, RIGHT)

        quant_viz = VGroup(fp32_group, arrow1, int8_group, arrow2, int4_group)

        with self.voiceover(text="Visually, imagine the weights of our neural network. A standard 32-bit floating point number takes up 4 bytes of memory. If we quantize this to an 8-bit integer, INT8, it takes only 1 byte. And if we push it further to a 4-bit integer, INT4, it takes a mere half a byte. This is a massive compression ratio.") as tracker:
            self.play(Create(fp32_box), Write(fp32_label))
            self.play(GrowArrow(arrow1))
            self.play(Create(int8_box), Write(int8_label))
            self.play(GrowArrow(arrow2))
            self.play(Create(int4_box), Write(int4_label))
            self.wait(1.5)

        self.play(FadeOut(def_text), FadeOut(quant_viz))

        # Deep Dive: The Math of Quantization (Scaling & Zero Point)
        math_title = Text("The Math of Quantization", font_size=36, color=BLUE).next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="To truly understand quantization, we must look at the underlying mathematics. The most common approach is affine quantization, which relies on two key parameters: a scaling factor, denoted as S, and a zero point, denoted as Z.") as tracker:
            self.play(Write(math_title))
            self.wait(1.5)

        eq1_part1 = MathTex(r"r", r"=", r"S", r"(", r"q", r"-", r"Z", r")", color=BLACK, font_size=40)
        eq1_part1.set_color_by_tex("r", RED)
        eq1_part1.set_color_by_tex("q", GREEN)
        eq1_part1.next_to(math_title, DOWN, buff=0.5)

        with self.voiceover(text="The core equation maps our real floating-point value, r, to our quantized integer value, q. The equation is: r equals S times the quantity q minus Z. Here, r is the original high-precision weight or activation, and q is the new low-precision integer representation.") as tracker:
            self.play(Write(eq1_part1))
            self.wait(1.5)

        # Rearranging the equation
        eq2_part1 = MathTex(r"q", r"=", r"\text{round}\left(", r"\frac{r}{S}", r"+", r"Z", r"\right)", color=BLACK, font_size=40)
        eq2_part1.set_color_by_tex("r", RED)
        eq2_part1.set_color_by_tex("q", GREEN)
        eq2_part1.next_to(eq1_part1, DOWN, buff=0.5)

        with self.voiceover(text="If we rearrange this equation to solve for our target quantized value q, we divide the real value r by the scaling factor S, add the zero point Z, and then round the result to the nearest integer. The rounding step is crucial, as it introduces the quantization error or information loss that we must manage.") as tracker:
            self.play(TransformMatchingTex(eq1_part1.copy(), eq2_part1))
            self.wait(1.5)

        scale_eq = MathTex(r"S", r"=", r"\frac{r_{\max} - r_{\min}}{q_{\max} - q_{\min}}", color=BLACK, font_size=40)
        scale_eq.next_to(eq2_part1, DOWN, buff=0.5)

        with self.voiceover(text="How do we determine the scaling factor S? We look at the distribution of our real values. S is calculated as the range of the real values—the maximum r minus the minimum r—divided by the range of the quantized values. For an 8-bit unsigned integer, the quantized range is from 0 to 255.") as tracker:
            self.play(Write(scale_eq))
            self.wait(1.5)

        zp_eq = MathTex(r"Z", r"=", r"\text{round}\left(", r"q_{\min} - \frac{r_{\min}}{S}", r"\right)", color=BLACK, font_size=40)
        zp_eq.next_to(scale_eq, DOWN, buff=0.5)

        with self.voiceover(text="Similarly, the zero point Z aligns the real zero with an integer value in our quantized space. It is calculated as the minimum quantized value minus the minimum real value divided by the scaling factor. The zero point ensures that a real value of exactly zero maps precisely to an integer without error, which is vital for operations like padding in neural networks.") as tracker:
            self.play(Write(zp_eq))
            self.wait(1.5)

        self.play(
            FadeOut(math_title), FadeOut(eq1_part1), FadeOut(eq2_part1),
            FadeOut(scale_eq), FadeOut(zp_eq)
        )

        # Why do we need it?
        why_title = Text("Why do we need it?", font_size=36, color=RED).next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="Now that we know what it is and how the math works, why do we need it? The answer is simple: Large Language Models are incredibly memory hungry. Let's look at a concrete example.") as tracker:
            self.play(Write(why_title))
            self.wait(1.5)

        model_size = Text("Llama-3 8B Parameters", font_size=32, color=BLACK).next_to(why_title, DOWN, buff=0.5)

        fp16_mem = Text("FP16 (16-bit): ~16 GB VRAM", font_size=28, color=BLUE).next_to(model_size, DOWN, buff=0.5)
        int8_mem = Text("INT8 (8-bit): ~8 GB VRAM", font_size=28, color=GREEN).next_to(fp16_mem, DOWN, buff=0.3)
        int4_mem = Text("INT4 (4-bit): ~4-5 GB VRAM", font_size=28, color=PURPLE).next_to(int8_mem, DOWN, buff=0.3)

        with self.voiceover(text="Consider a popular open-weights model like Llama 3 with 8 Billion parameters. If we load this model using 16-bit floating point, FP16, we need approximately 16 Gigabytes of VRAM just to fit the weights into memory, not even counting the context window.") as tracker:
            self.play(Write(model_size))
            self.play(FadeIn(fp16_mem, shift=RIGHT))
            self.wait(1.5)

        with self.voiceover(text="This means it won't fit on most consumer GPUs. However, if we quantize the model to INT8, we cut the memory requirement in half, down to 8 Gigabytes. Now it fits on a standard gaming GPU.") as tracker:
            self.play(FadeIn(int8_mem, shift=RIGHT))
            self.wait(1.5)

        with self.voiceover(text="If we aggressively quantize to INT4, the footprint drops to just 4 to 5 Gigabytes. This opens up entirely new possibilities, allowing us to run powerful models locally on edge devices, such as MacBooks or even high-end smartphones, completely offline and with high inference speed.") as tracker:
            self.play(FadeIn(int4_mem, shift=RIGHT))
            self.wait(1.5)

        self.play(FadeOut(why_title), FadeOut(model_size), FadeOut(fp16_mem), FadeOut(int8_mem), FadeOut(int4_mem))

        # Use Cases
        usecase_title = Text("Real-World Use Cases", font_size=36, color=GREEN).next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="Quantization is not just a theoretical concept; it is heavily utilized in production by major tech companies. Let's look at two specific real-world use cases.") as tracker:
            self.play(Write(usecase_title))
            self.wait(1.5)

        uc1 = VGroup(
            Text("1. Server-side Inference", font_size=28, color=BLACK, weight=BOLD),
            Text("Companies like OpenAI use INT8 quantization techniques", font_size=24, color=BLACK),
            Text("to accelerate inference and increase throughput for ChatGPT,", font_size=24, color=BLACK),
            Text("serving millions of concurrent users efficiently.", font_size=24, color=BLACK)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(usecase_title, DOWN, buff=0.5).align_to(usecase_title, LEFT)

        with self.voiceover(text="The first use case is massive server-side inference. Companies like OpenAI employ sophisticated INT8 quantization techniques to accelerate inference and dramatically increase throughput for services like ChatGPT. By moving less data from memory to the compute cores, they overcome memory bandwidth bottlenecks, allowing them to serve millions of concurrent users cost-effectively.") as tracker:
            self.play(FadeIn(uc1, shift=UP))
            self.wait(1.5)

        uc2 = VGroup(
            Text("2. Edge Computing & Local Deployment", font_size=28, color=BLACK, weight=BOLD),
            Text("Tools like Llama.cpp and Ollama rely heavily on INT4", font_size=24, color=BLACK),
            Text("quantization (e.g., GGUF format) to run sophisticated", font_size=24, color=BLACK),
            Text("LLMs locally on consumer hardware like MacBooks.", font_size=24, color=BLACK)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(uc1, DOWN, buff=0.5).align_to(uc1, LEFT)

        with self.voiceover(text="The second use case is edge computing and local deployment. Tools like Llama dot C P P and Ollama rely heavily on INT4 quantization formats, such as GGUF. This enables developers and researchers to run sophisticated Large Language Models locally on consumer hardware, ensuring data privacy and reducing cloud computing costs.") as tracker:
            self.play(FadeIn(uc2, shift=UP))
            self.wait(1.5)

        self.play(FadeOut(usecase_title), FadeOut(uc1), FadeOut(uc2))

        # Key Interview Insight
        insight_title = Text("Key Interview Insight", font_size=36, color=PURPLE).next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="Finally, let's discuss the key interview insight. When interviewing for an AI Engineering role, you will almost certainly be asked about the tradeoffs of quantization.") as tracker:
            self.play(Write(insight_title))
            self.wait(1.5)

        callout_box = RoundedRectangle(corner_radius=0.2, width=10, height=4, color=PURPLE, fill_opacity=0.1)
        callout_box.next_to(insight_title, DOWN, buff=0.5)

        insight_text = VGroup(
            Text("The Tradeoff: Precision vs. Performance", font_size=32, color=BLACK, weight=BOLD),
            Text("Quantization is not free. The rounding process introduces", font_size=26, color=BLACK),
            Text("Quantization Error, leading to accuracy degradation.", font_size=26, color=BLACK),
            Text("Gotcha: Watch out for 'Outlier Features'. In LLMs,", font_size=26, color=RED),
            Text("certain hidden states have massive outlier values.", font_size=26, color=RED),
            Text("Aggressive quantization clips these, causing severe performance drops.", font_size=26, color=RED)
        ).arrange(DOWN, buff=0.2).move_to(callout_box)

        with self.voiceover(text="Here is the core tradeoff you must articulate: Precision versus Performance. Quantization is not a free lunch. The rounding process we saw earlier introduces quantization error, which inevitably leads to some degree of accuracy or perplexity degradation in the model. The lower the precision, the higher the error.") as tracker:
            self.play(Create(callout_box))
            self.play(Write(insight_text[:3]))
            self.wait(1.5)

        with self.voiceover(text="And here is the specific 'gotcha' interviewers look for: Outlier features. In large language models, especially those over 6 billion parameters, certain hidden states develop massive outlier values. If you apply naive quantization across the entire model, these outliers get clipped or squashed, causing severe and sudden performance drops.") as tracker:
            self.play(Write(insight_text[3:]))
            self.wait(1.5)

        with self.voiceover(text="To mitigate this, state-of-the-art techniques like LLM dot int8() use mixed-precision. They keep the problematic outlier dimensions in 16-bit floating point while quantizing the rest of the weights to 8-bit. Demonstrating this nuanced understanding of outliers will set you apart in a system design interview.") as tracker:
            self.wait(2)

        with self.voiceover(text="That concludes our deep dive into Model Quantization for production. Understanding how to shrink models without destroying their capabilities is a superpower in modern AI engineering. Keep practicing, and I'll see you in the next session.") as tracker:
            self.play(FadeOut(insight_title), FadeOut(callout_box), FadeOut(insight_text), FadeOut(title))
            self.wait(2)

        self.wait(1)
