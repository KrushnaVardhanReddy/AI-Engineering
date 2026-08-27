from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class AutoencoderScene(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # Section 1: What is it?
        self.what_is_it()

        # Section 2: Why do we need it?
        self.why_do_we_need_it()

        # Section 3: Use Cases
        self.use_cases()

        # Section 4: Key Interview Insight
        self.key_interview_insight()

    def what_is_it(self):
        title = Text("What is an Autoencoder?", font_size=48, color=BLACK).to_edge(UP)

        definition_text = (
            "An autoencoder is a type of artificial neural network used to learn\n"
            "efficient codings of unlabeled data. It compresses the input into a\n"
            "lower-dimensional latent space and then reconstructs the original data."
        )
        definition = Text(definition_text, font_size=24, color=BLACK).next_to(title, DOWN, buff=0.5)

        with self.voiceover(
            text="Welcome to Day 62 of our AI Engineering series. Today, we are exploring a fundamental concept in deep learning: Autoencoders and the Latent Space. "
                 "First, what exactly is an autoencoder? An autoencoder is a special type of artificial neural network designed to learn efficient codings of unlabeled data, completely unsupervised. "
                 "Its primary job is to compress the input data into a lower-dimensional representation, which we call the latent space, and then reconstruct the original data from this compressed form as accurately as possible. "
                 "Think of it as trying to summarize a long book into a single page of bullet points, and then writing a new book based only on those bullet points that matches the original as closely as possible."
        ) as tracker:
            self.play(Write(title))
            self.play(Write(definition))
            self.wait(1.5)

        # Diagram
        encoder_box = Rectangle(width=2, height=3, color=BLUE, fill_opacity=0.2)
        decoder_box = Rectangle(width=2, height=3, color=RED, fill_opacity=0.2)
        latent_box = Rectangle(width=1, height=1.5, color=PURPLE, fill_opacity=0.2)

        encoder_label = Text("Encoder", font_size=24, color=BLACK).move_to(encoder_box)
        decoder_label = Text("Decoder", font_size=24, color=BLACK).move_to(decoder_box)
        latent_label = Text("Latent Space", font_size=20, color=BLACK).move_to(latent_box)

        network_group = VGroup(
            VGroup(encoder_box, encoder_label),
            VGroup(latent_box, latent_label),
            VGroup(decoder_box, decoder_label)
        ).arrange(RIGHT, buff=1.0).next_to(definition, DOWN, buff=1.0)

        input_arrow = Arrow(start=network_group[0].get_left() + LEFT, end=network_group[0].get_left(), color=BLACK)
        input_label = Text("Input (x)", font_size=24, color=BLACK).next_to(input_arrow, LEFT)

        enc_lat_arrow = Arrow(start=network_group[0].get_right(), end=network_group[1].get_left(), color=BLACK)
        lat_dec_arrow = Arrow(start=network_group[1].get_right(), end=network_group[2].get_left(), color=BLACK)

        output_arrow = Arrow(start=network_group[2].get_right(), end=network_group[2].get_right() + RIGHT, color=BLACK)
        output_label = Tex(r"Output ($\hat{x}$)", font_size=24, color=BLACK).next_to(output_arrow, RIGHT)

        with self.voiceover(
            text="Let's look at the architecture. An autoencoder consists of two main parts joined together. "
                 "First is the Encoder. The encoder takes the high-dimensional input data, let's call it x, and compresses it. "
                 "It maps this input into a hidden, lower-dimensional space. We call this the Latent Space. This is the bottleneck of the network. "
                 "Next is the Decoder. The decoder takes this compressed latent representation and attempts to reconstruct the original input data. "
                 "The output of the decoder, x-hat, should ideally be identical to the original input x."
        ) as tracker:
            self.play(FadeIn(input_label), GrowArrow(input_arrow))
            self.play(FadeIn(network_group[0]))
            self.wait(1.5)
            self.play(GrowArrow(enc_lat_arrow), FadeIn(network_group[1]))
            self.wait(1.5)
            self.play(GrowArrow(lat_dec_arrow), FadeIn(network_group[2]))
            self.wait(1.5)
            self.play(GrowArrow(output_arrow), FadeIn(output_label))
            self.wait(1.5)

        # Math Loss (using TransformMatchingTex as requested)
        loss_eq_1 = MathTex(r'L(x, \hat{x})', color=BLACK).next_to(network_group, DOWN, buff=1.0)
        loss_eq_2 = MathTex(r'L(x, \hat{x})', r'=', r'||', r'x', r'-', r'\hat{x}', r'||^2', color=BLACK).next_to(network_group, DOWN, buff=1.0)

        with self.voiceover(
            text="How does it learn to do this? We train the autoencoder by minimizing the reconstruction loss. "
                 "The loss function measures how different the reconstructed output is from the original input. "
                 "A very common choice is the Mean Squared Error loss. "
                 "We want to minimize the squared difference between the input x and the output x-hat. "
                 "By forcing the data through a bottleneck, the network cannot simply copy the input to the output. "
                 "It is forced to learn the most important, salient features of the data to reconstruct it successfully."
        ) as tracker:
            self.play(Write(loss_eq_1))
            self.wait(1.5)
            self.play(TransformMatchingTex(loss_eq_1, loss_eq_2))
            self.wait(1.5)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

    def why_do_we_need_it(self):
        title = Text("Why do we need it?", font_size=48, color=BLACK).to_edge(UP)

        with self.voiceover(
            text="Why do we need autoencoders? What specific problems do they solve? "
                 "In many real-world scenarios, we deal with extremely high-dimensional data, like high-resolution images or complex user profiles. "
                 "This high dimensionality makes data processing slow, computationally expensive, and often noisy. "
                 "We need a way to extract the meaningful information while discarding the noise."
        ) as tracker:
            self.play(Write(title))
            self.wait(1.5)

        # Before (Without Autoencoder)
        before_title = Text("Without Autoencoder: Curse of Dimensionality", font_size=32, color=RED).next_to(title, DOWN, buff=0.5)

        # Draw a complex messy graph or large set of dots
        np.random.seed(42)
        dots = VGroup(*[Dot(point=[np.random.uniform(-4, 0), np.random.uniform(-2, 2), 0], color=BLACK, radius=0.05) for _ in range(200)])

        with self.voiceover(
            text="Consider a scenario without an autoencoder. You have raw, high-dimensional data. "
                 "It is scattered, noisy, and suffers from the curse of dimensionality. "
                 "Machine learning models struggle to find patterns in this dense, unorganized space."
        ) as tracker:
            self.play(Write(before_title))
            self.play(FadeIn(dots))
            self.wait(1.5)

        # After (With Autoencoder)
        after_title = Text("With Autoencoder: Structured Latent Space", font_size=32, color=GREEN).next_to(title, DOWN, buff=0.5)

        # Transform to organized clusters
        cluster1 = VGroup(*[Dot(point=[np.random.normal(2, 0.5), np.random.normal(1, 0.5), 0], color=BLUE, radius=0.08) for _ in range(100)])
        cluster2 = VGroup(*[Dot(point=[np.random.normal(2, 0.5), np.random.normal(-1, 0.5), 0], color=PURPLE, radius=0.08) for _ in range(100)])
        organized_dots = VGroup(cluster1, cluster2)

        with self.voiceover(
            text="Now, let's apply an autoencoder. By training it to reconstruct the data through a bottleneck, we force it to find a lower-dimensional representation. "
                 "This Latent Space is not just compressed; it's structured. "
                 "Similar data points map to similar regions in the latent space. "
                 "The autoencoder has effectively learned to denoise the data and extract the underlying, meaningful features."
        ) as tracker:
            self.play(FadeOut(before_title), Write(after_title))
            self.play(Transform(dots, organized_dots))
            self.wait(1.5)

        # Show exact dimension mapping example
        dim_text = Text("Example: Image Compression (784 -> 128 -> 32 -> 128 -> 784)", font_size=28, color=BLACK).to_edge(DOWN, buff=1.0)

        with self.voiceover(
            text="For example, if we are processing 28 by 28 pixel images, our input dimension is 784. "
                 "We can design an encoder that compresses this 784-dimensional vector down to 128 dimensions, and then down to a bottleneck of just 32 dimensions. "
                 "The decoder then takes this 32-dimensional vector and expands it back to 128, and finally reconstructs the 784-dimensional image. "
                 "This massive reduction from 784 to 32 forces the network to learn the true essence of the image."
        ) as tracker:
            self.play(Write(dim_text))
            self.wait(1.5)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

    def use_cases(self):
        title = Text("Real-World Use Cases", font_size=48, color=BLACK).to_edge(UP)

        with self.voiceover(
            text="Where are autoencoders actually used in the real world? Let's look at a couple of prominent examples."
        ) as tracker:
            self.play(Write(title))
            self.wait(1.0)

        # Use Case 1
        uc1_title = Text("1. Spotify: Music Recommendation", font_size=36, color=BLUE).next_to(title, DOWN, buff=1.0).align_to(title, LEFT)
        uc1_desc = Text("Compresses complex audio features and user listening\nhistories into dense latent vectors for similarity matching.", font_size=24, color=BLACK).next_to(uc1_title, DOWN, buff=0.3).align_to(uc1_title, LEFT)

        with self.voiceover(
            text="First, consider recommendation systems, like those used by Spotify. "
                 "Spotify needs to match your complex listening history with millions of songs. "
                 "They can use autoencoders to compress user profiles and audio features into dense latent vectors. "
                 "In this latent space, calculating the distance between a user vector and a song vector is extremely fast and accurate, allowing them to recommend songs you will love."
        ) as tracker:
            self.play(Write(uc1_title))
            self.play(Write(uc1_desc))
            self.wait(1.5)

        # Use Case 2
        uc2_title = Text("2. Credit Card Companies: Anomaly Detection", font_size=36, color=RED).next_to(uc1_desc, DOWN, buff=1.0).align_to(title, LEFT)
        uc2_desc = Text("Trains on normal transactions. Fraudulent transactions\ncannot be reconstructed well, resulting in high reconstruction loss.", font_size=24, color=BLACK).next_to(uc2_title, DOWN, buff=0.3).align_to(uc2_title, LEFT)

        with self.voiceover(
            text="Second, anomaly detection, heavily used by credit card companies for fraud detection. "
                 "They train an autoencoder exclusively on normal, non-fraudulent transaction data. "
                 "When a fraudulent transaction occurs, the autoencoder has never seen patterns like it before. "
                 "Therefore, it fails to reconstruct the fraudulent data accurately, resulting in a very high reconstruction loss. "
                 "This high loss acts as an immediate trigger to flag the transaction for fraud review."
        ) as tracker:
            self.play(Write(uc2_title))
            self.play(Write(uc2_desc))
            self.wait(1.5)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

    def key_interview_insight(self):
        title = Text("Key Interview Insight", font_size=48, color=BLACK).to_edge(UP)

        with self.voiceover(
            text="Finally, let's discuss the most important interview insight regarding autoencoders. "
                 "If you are asked about autoencoders in an AI engineering interview, this is the gotcha they are likely testing for."
        ) as tracker:
            self.play(Write(title))
            self.wait(1.0)

        # Callout Box
        box = Rectangle(width=10, height=4, color=PURPLE, fill_opacity=0.1)
        box.next_to(title, DOWN, buff=1.0)

        insight_title = Text("The Identity Function Trap", font_size=36, color=PURPLE).next_to(box.get_top(), DOWN, buff=0.3)
        insight_text = Text(
            "If the latent space capacity is too large (e.g., more dimensions\n"
            "than the input), the autoencoder might simply learn the Identity\n"
            "Function (copying input to output) without learning any useful\n"
            "features. You must constrain the network (bottleneck, sparsity, noise).",
            font_size=28, color=BLACK
        ).next_to(insight_title, DOWN, buff=0.5)

        with self.voiceover(
            text="It is known as the Identity Function Trap. "
                 "Interviewers will often ask: What happens if your latent space is larger than your input space? or How do you ensure your autoencoder actually learns useful features? "
                 "The answer is that if the autoencoder has too much capacity, it will take the lazy route. "
                 "It will simply memorize the data by learning the identity function, directly copying the input to the output without discovering any underlying structure. "
                 "To prevent this, you must apply constraints. The most common constraint is a strict structural bottleneck, forcing a lower dimension. "
                 "Other methods include adding sparsity constraints, or introducing noise to the input, creating what is known as a Denoising Autoencoder. "
                 "Remembering this tradeoff between capacity and feature learning is crucial for demonstrating true understanding of autoencoders."
        ) as tracker:
            self.play(FadeIn(box))
            self.play(Write(insight_title))
            self.play(Write(insight_text))
            self.wait(2.0)

        with self.voiceover(
            text="That wraps up our deep dive into Autoencoders and the Latent Space. Keep practicing, and I will see you in the next lesson."
        ) as tracker:
            self.wait(1.0)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )
