from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class VAESamplingScene(VoiceoverScene):
    def construct(self):
        # Setup the voiceover service
        self.set_speech_service(GTTSService())

        # Setup aesthetic (Whiteboard style)
        self.camera.background_color = WHITE
        Text.set_default(color=BLACK)
        Tex.set_default(color=BLACK)
        MathTex.set_default(color=BLACK)

        # --- Section 1: What is it? ---
        title_what = Text("What is Variational Autoencoder (VAE) Sampling?", font_size=40, weight=BOLD).to_edge(UP)
        self.play(Write(title_what))

        with self.voiceover(text="""
            Welcome to Day 63. Today, we are going to dive deep into Variational Autoencoder, or VAE, Sampling.
            So, what is it? A Variational Autoencoder is a powerful generative model that learns a continuous,
            highly structured latent space. Unlike a standard autoencoder that maps an input to a single, fixed
            point in space, a VAE maps an input to a probability distribution, typically a Gaussian distribution.
            By doing this, it generates new data by sampling from these learned probability distributions rather
            than relying on fixed, rigid points. This gives us the ability to generate entirely new, realistic
            data that shares the characteristics of our training set.
            Let us visualize the fundamental difference between a standard autoencoder and a variational autoencoder.
        """) as tracker:
            # Draw standard autoencoder vs VAE concept
            ae_text = Text("Standard Autoencoder", font_size=24).move_to(LEFT * 3.5 + UP * 1.5)
            vae_text = Text("Variational Autoencoder", font_size=24).move_to(RIGHT * 3.5 + UP * 1.5)
            self.play(Write(ae_text), Write(vae_text))

            # AE drawing
            ae_enc = Rectangle(width=1, height=2, color=BLUE).move_to(LEFT * 5 + UP * 0)
            ae_latent = Dot(color=BLACK).move_to(LEFT * 3.5 + UP * 0)
            ae_dec = Rectangle(width=1, height=2, color=RED).move_to(LEFT * 2 + UP * 0)
            ae_arr1 = Arrow(ae_enc.get_right(), ae_latent.get_left(), buff=0.1, color=BLACK)
            ae_arr2 = Arrow(ae_latent.get_right(), ae_dec.get_left(), buff=0.1, color=BLACK)

            self.play(Create(ae_enc), Create(ae_dec))
            self.play(GrowArrow(ae_arr1), GrowArrow(ae_arr2), FadeIn(ae_latent))

            # VAE drawing
            vae_enc = Rectangle(width=1, height=2, color=BLUE).move_to(RIGHT * 2 + UP * 0)
            vae_mu = Text("μ", font_size=24).move_to(RIGHT * 3.5 + UP * 0.3)
            vae_sigma = Text("σ", font_size=24).move_to(RIGHT * 3.5 + DOWN * 0.3)
            vae_sample = Circle(radius=0.3, color=PURPLE).move_to(RIGHT * 3.5 + UP * 0)
            vae_dec = Rectangle(width=1, height=2, color=RED).move_to(RIGHT * 5 + UP * 0)
            vae_arr1_mu = Arrow(vae_enc.get_right(), vae_mu.get_left(), buff=0.1, color=BLACK)
            vae_arr1_sigma = Arrow(vae_enc.get_right(), vae_sigma.get_left(), buff=0.1, color=BLACK)
            vae_arr2_sample = Arrow(vae_sample.get_right(), vae_dec.get_left(), buff=0.1, color=BLACK)

            self.play(Create(vae_enc), Create(vae_dec))
            self.play(GrowArrow(vae_arr1_mu), GrowArrow(vae_arr1_sigma), Write(vae_mu), Write(vae_sigma))
            self.play(Create(vae_sample))
            self.play(GrowArrow(vae_arr2_sample))

        self.wait(1.5)

        with self.voiceover(text="""
            In a standard autoencoder, the encoder produces a single point, shown as the black dot. But in a VAE,
            the encoder outputs two things: a mean, denoted by mu, and a variance, denoted by sigma. We then define
            a distribution using these parameters, and we randomly sample a point from this distribution to pass
            to the decoder. This introduces stochasticity, which is crucial for generation. However, it also introduces
            a massive mathematical hurdle for training the network via backpropagation.
        """) as tracker:
            self.play(Indicate(ae_latent, color=RED, scale_factor=1.5))
            self.play(Indicate(vae_mu, color=GREEN, scale_factor=1.5), Indicate(vae_sigma, color=GREEN, scale_factor=1.5))
            self.play(Indicate(vae_sample, color=PURPLE, scale_factor=1.5))

        self.wait(1.5)
        self.play(
            FadeOut(ae_text), FadeOut(vae_text), FadeOut(ae_enc), FadeOut(ae_latent),
            FadeOut(ae_dec), FadeOut(ae_arr1), FadeOut(ae_arr2), FadeOut(vae_enc),
            FadeOut(vae_mu), FadeOut(vae_sigma), FadeOut(vae_sample), FadeOut(vae_dec),
            FadeOut(vae_arr1_mu), FadeOut(vae_arr1_sigma), FadeOut(vae_arr2_sample), FadeOut(title_what)
        )

        # --- Section 2: Why do we need it? ---
        title_why = Text("Why do we need VAE Sampling?", font_size=40, weight=BOLD).to_edge(UP)
        self.play(Write(title_why))

        with self.voiceover(text="""
            Now, why do we actually need this probabilistic approach? Why can we not just use a standard
            autoencoder to generate new images or text? The answer lies in the topology of the latent space.
            Let us look at the latent space of a standard autoencoder first. Because a standard autoencoder is
            only trained to reconstruct its exact inputs, it tends to memorize the data. It maps inputs to
            distinct, isolated points in space, creating severe gaps and sharp boundaries between different classes.
        """) as tracker:
            # Draw Standard AE Latent Space
            axes = Axes(x_range=[-3, 3], y_range=[-3, 3], x_length=5, y_length=5, axis_config={"color": BLACK}).move_to(LEFT * 3.5 + DOWN * 0.5)
            ax_labels = axes.get_axis_labels(x_label="z_1", y_label="z_2")
            self.play(Create(axes), Write(ax_labels))

            import numpy as np

            cluster1 = VGroup(*[Dot(axes.c2p(np.random.normal(-1.5, 0.2), np.random.normal(1.5, 0.2)), color=BLUE) for _ in range(20)])
            cluster2 = VGroup(*[Dot(axes.c2p(np.random.normal(1.5, 0.2), np.random.normal(-1.5, 0.2)), color=RED) for _ in range(20)])
            cluster3 = VGroup(*[Dot(axes.c2p(np.random.normal(1.5, 0.2), np.random.normal(1.5, 0.2)), color=GREEN) for _ in range(20)])

            self.play(FadeIn(cluster1), FadeIn(cluster2), FadeIn(cluster3))

            gap_label = Text("Large Empty Gaps", font_size=20, color=RED).move_to(axes.c2p(0, 0))
            gap_arrow = Arrow(gap_label.get_bottom(), axes.c2p(0, -0.5), buff=0.1, color=RED)
            self.play(Write(gap_label), GrowArrow(gap_arrow))

        self.wait(1.5)

        with self.voiceover(text="""
            If you try to generate new data by picking a random point in one of these empty gaps, the decoder
            will have no idea what to do, because it has never seen that region of space during training. The output
            will be complete garbage or meaningless noise. This means standard autoencoders cannot interpolate
            smoothly between concepts.
            Now, contrast this with a Variational Autoencoder.
        """) as tracker:
            bad_sample = Dot(axes.c2p(0, 0), color=PURPLE, radius=0.1)
            self.play(Create(bad_sample))
            cross = Cross(bad_sample, stroke_color=RED, scale_factor=0.5)
            self.play(Create(cross))

        self.wait(1.5)
        self.play(FadeOut(axes), FadeOut(ax_labels), FadeOut(cluster1), FadeOut(cluster2), FadeOut(cluster3), FadeOut(gap_label), FadeOut(gap_arrow), FadeOut(bad_sample), FadeOut(cross))

        with self.voiceover(text="""
            Because a VAE forces the encodings to be probability distributions, usually constrained to a standard
            normal distribution, it naturally fills the space. The distributions overlap and blend together.
            This forces the decoder to learn a smooth, continuous, and highly structured mapping. As a result,
            every point in the latent space decodes into something meaningful, and we can seamlessly interpolate
            between different features or classes.
        """) as tracker:
            # Draw VAE Latent Space
            axes_vae = Axes(x_range=[-3, 3], y_range=[-3, 3], x_length=5, y_length=5, axis_config={"color": BLACK}).move_to(DOWN * 0.5)
            ax_labels_vae = axes_vae.get_axis_labels(x_label="z_1", y_label="z_2")
            self.play(Create(axes_vae), Write(ax_labels_vae))

            # Overlapping distributions (clouds of points)
            cloud1 = VGroup(*[Dot(axes_vae.c2p(np.random.normal(-0.5, 0.8), np.random.normal(0.5, 0.8)), color=BLUE, fill_opacity=0.5) for _ in range(50)])
            cloud2 = VGroup(*[Dot(axes_vae.c2p(np.random.normal(0.5, 0.8), np.random.normal(-0.5, 0.8)), color=RED, fill_opacity=0.5) for _ in range(50)])
            cloud3 = VGroup(*[Dot(axes_vae.c2p(np.random.normal(0, 0.8), np.random.normal(0, 0.8)), color=GREEN, fill_opacity=0.5) for _ in range(50)])

            self.play(FadeIn(cloud1), FadeIn(cloud2), FadeIn(cloud3))

            smooth_label = Text("Smooth, Continuous Space", font_size=24, color=PURPLE).move_to(axes_vae.c2p(0, -3.5))
            self.play(Write(smooth_label))

        self.wait(1.5)
        self.play(FadeOut(axes_vae), FadeOut(ax_labels_vae), FadeOut(cloud1), FadeOut(cloud2), FadeOut(cloud3), FadeOut(smooth_label), FadeOut(title_why))

        # --- Section 3: Use Cases ---
        title_cases = Text("Real-World Use Cases", font_size=40, weight=BOLD).to_edge(UP)
        self.play(Write(title_cases))

        with self.voiceover(text="""
            So where are Variational Autoencoders actually used in the real world?
            One massive application is in Drug Discovery and molecular design.
            Companies and researchers use VAEs to map known chemical compounds into a continuous latent space.
            By sampling new points in that space, or interpolating between two known successful drugs, they can
            generate novel, valid molecular structures that might have the perfect properties to treat a disease,
            rapidly speeding up pharmaceutical research.
        """) as tracker:
            case1_title = Text("1. Drug Discovery (Molecular Design)", font_size=30, color=BLUE).move_to(UP * 1.5 + LEFT * 2)
            self.play(Write(case1_title))

            # Simple molecule graphic
            mol_c = Circle(radius=0.3, color=BLACK, fill_color=WHITE, fill_opacity=1).move_to(UP * 0.5 + LEFT * 4)
            mol_text = Text("C", font_size=20).move_to(mol_c.get_center())
            mol_o = Circle(radius=0.3, color=RED, fill_color=RED, fill_opacity=0.5).move_to(UP * 1.5 + LEFT * 5)
            mol_text_o = Text("O", font_size=20).move_to(mol_o.get_center())
            bond = Line(mol_c.get_center(), mol_o.get_center(), color=BLACK)

            molecule = VGroup(bond, mol_c, mol_text, mol_o, mol_text_o)
            self.play(FadeIn(molecule))

        self.wait(1.5)

        with self.voiceover(text="""
            A second massive use case you might interact with daily is Stable Diffusion.
            The high-resolution images generated by tools like Midjourney or open-source Stable Diffusion models
            are actually generated in a much smaller latent space. They use a highly specialized VAE to compress
            pixel-perfect high-dimensional images into a lower-dimensional latent representation. The diffusion
            process runs efficiently in this compressed space, and the VAE decoder is then used to translate it
            back into stunning, high-definition pixels.
        """) as tracker:
            case2_title = Text("2. Stable Diffusion (Image Compression)", font_size=30, color=RED).move_to(DOWN * 1.5 + LEFT * 1.5)
            self.play(Write(case2_title))

            # Simple image compression graphic
            img_rect = Rectangle(width=1.5, height=1.5, color=GREEN).move_to(DOWN * 0.5 + LEFT * 5)
            img_text = Text("1024x1024", font_size=16).move_to(img_rect.get_center())

            latent_rect = Rectangle(width=0.5, height=0.5, color=PURPLE).move_to(DOWN * 0.5 + LEFT * 2)
            latent_text = Text("64x64", font_size=12).move_to(latent_rect.get_center())

            img_arrow = Arrow(img_rect.get_right(), latent_rect.get_left(), buff=0.1, color=BLACK)
            vae_label_img = Text("VAE Encoder", font_size=16).next_to(img_arrow, UP)

            img_group = VGroup(img_rect, img_text, latent_rect, latent_text, img_arrow, vae_label_img)
            self.play(FadeIn(img_group))

        self.wait(1.5)
        self.play(FadeOut(case1_title), FadeOut(molecule), FadeOut(case2_title), FadeOut(img_group), FadeOut(title_cases))

        # --- Section 4: Key Interview Insight ---
        title_insight = Text("Key Interview Insight", font_size=40, weight=BOLD).to_edge(UP)
        self.play(Write(title_insight))

        with self.voiceover(text="""
            Finally, we arrive at the most critical concept, and the most common interview question you will
            face regarding Variational Autoencoders. The interviewer will ask: "How do you backpropagate
            gradients through a random sampling process?"
            Let us look at the standard approach. We have our mean mu and standard deviation sigma. We want to
            sample a variable z.
        """) as tracker:
            # Draw standard sampling node
            node_mu = Text("μ", font_size=36).move_to(LEFT * 4 + UP * 1)
            node_sigma = Text("σ", font_size=36).move_to(LEFT * 4 + DOWN * 1)

            sampler_box = Rectangle(width=2, height=1, color=RED).move_to(LEFT * 1)
            sampler_text = Text("Random\nSample", font_size=20).move_to(sampler_box.get_center())

            node_z = Text("z", font_size=36).move_to(RIGHT * 2)

            arr_mu_samp = Arrow(node_mu.get_right(), sampler_box.get_left(), buff=0.1, color=BLACK)
            arr_sig_samp = Arrow(node_sigma.get_right(), sampler_box.get_left(), buff=0.1, color=BLACK)
            arr_samp_z = Arrow(sampler_box.get_right(), node_z.get_left(), buff=0.1, color=BLACK)

            self.play(Write(node_mu), Write(node_sigma))
            self.play(Create(sampler_box), Write(sampler_text), GrowArrow(arr_mu_samp), GrowArrow(arr_sig_samp))
            self.play(Write(node_z), GrowArrow(arr_samp_z))

        with self.voiceover(text="""
            The problem is that the sampling operation is inherently non-deterministic. It involves drawing a random
            value. Because it is random, there is no mathematical derivative. Backpropagation relies on the chain
            rule to compute gradients, and it hits a brick wall at this random node. The gradient cannot flow back
            to update mu and sigma, meaning the encoder cannot learn.
        """) as tracker:
            grad_arrow = Arrow(node_z.get_bottom(), sampler_box.get_bottom(), buff=0.1, color=RED)
            grad_text = Text("Gradient Blocked!", font_size=20, color=RED).next_to(grad_arrow, DOWN)
            cross_grad = Cross(sampler_box, stroke_color=RED, scale_factor=0.8)

            self.play(GrowArrow(grad_arrow), Write(grad_text))
            self.play(Create(cross_grad))

        self.wait(1.5)
        self.play(FadeOut(sampler_box), FadeOut(sampler_text), FadeOut(arr_mu_samp), FadeOut(arr_sig_samp), FadeOut(arr_samp_z), FadeOut(grad_arrow), FadeOut(grad_text), FadeOut(cross_grad))

        with self.voiceover(text="""
            The genius solution to this is called the Reparameterization Trick.
            Instead of drawing z directly from the distribution defined by mu and sigma, we introduce an auxiliary
            noise variable, called epsilon, which is sampled from a standard normal distribution with a mean of zero
            and a variance of one.
        """) as tracker:
            callout = SurroundingRectangle(VGroup(node_mu, node_sigma, node_z), color=BLUE, buff=1.5).shift(RIGHT*1)
            callout_title = Text("The Reparameterization Trick", font_size=24, color=BLUE).next_to(callout, UP)
            self.play(Create(callout), Write(callout_title))

            node_epsilon = Text("ε ~ N(0, 1)", font_size=24, color=PURPLE).move_to(LEFT * 1 + DOWN * 2)
            self.play(Write(node_epsilon))

        with self.voiceover(text="""
            We then calculate our sample z deterministically using a simple equation.
            z is equal to our mean mu, plus our standard deviation sigma, element-wise multiplied by our
            random noise epsilon.
            Let us animate this equation step-by-step.
        """) as tracker:
            eq1 = MathTex("z", "=", r"\mu").move_to(RIGHT * 1 + UP * 1)
            self.play(Write(eq1))
            self.wait(0.5)
            eq2 = MathTex("z", "=", r"\mu", "+", r"\sigma").move_to(RIGHT * 1 + UP * 1)
            self.play(TransformMatchingTex(eq1, eq2))
            self.wait(0.5)
            eq3 = MathTex("z", "=", r"\mu", "+", r"\sigma", r"\odot", r"\epsilon").move_to(RIGHT * 1 + UP * 1)
            self.play(TransformMatchingTex(eq2, eq3))

            # Show deterministic graph
            math_node = Circle(radius=0.5, color=GREEN).move_to(LEFT * 1)
            math_text = Text("+ , x", font_size=20).move_to(math_node.get_center())

            new_arr_mu = Arrow(node_mu.get_right(), math_node.get_left(), buff=0.1, color=BLACK)
            new_arr_sig = Arrow(node_sigma.get_right(), math_node.get_left(), buff=0.1, color=BLACK)
            new_arr_eps = Arrow(node_epsilon.get_top(), math_node.get_bottom(), buff=0.1, color=BLACK)
            new_arr_z = Arrow(math_node.get_right(), node_z.get_left(), buff=0.1, color=BLACK)

            self.play(Create(math_node), Write(math_text))
            self.play(GrowArrow(new_arr_mu), GrowArrow(new_arr_sig), GrowArrow(new_arr_eps))
            self.play(GrowArrow(new_arr_z))

        with self.voiceover(text="""
            By doing this, the randomness is completely isolated to the epsilon input. The operations connecting
            mu and sigma to z are now just standard addition and multiplication. These are continuous, deterministic
            mathematical functions. Because of this, the gradients can easily flow backward from z, through the
            addition and multiplication, directly into mu and sigma, allowing the entire model to be trained end-to-end
            with standard backpropagation.
        """) as tracker:
            success_grad1 = Arrow(node_z.get_bottom(), math_node.get_bottom(), buff=0.1, color=GREEN)
            success_grad2 = Arrow(math_node.get_top(), node_mu.get_right(), buff=0.1, color=GREEN)
            success_grad3 = Arrow(math_node.get_top(), node_sigma.get_right(), buff=0.1, color=GREEN)

            grad_success_text = Text("Gradients Flow Freely!", font_size=20, color=GREEN).next_to(success_grad1, DOWN)

            self.play(GrowArrow(success_grad1), Write(grad_success_text))
            self.play(GrowArrow(success_grad2), GrowArrow(success_grad3))

        self.wait(2)

        with self.voiceover(text="""
            This trick is the absolute cornerstone of Variational Autoencoders. If you can confidently explain the
            reparameterization trick on a whiteboard, you will absolutely ace this portion of your machine learning
            interview. That concludes our deep dive into VAE sampling. Good luck with your studies, and keep building.
        """) as tracker:
            self.play(Indicate(eq3, color=BLUE, scale_factor=1.2))

        self.wait(2)

        self.play(
            FadeOut(node_mu), FadeOut(node_sigma), FadeOut(node_z), FadeOut(callout),
            FadeOut(callout_title), FadeOut(node_epsilon), FadeOut(eq3), FadeOut(math_node),
            FadeOut(math_text), FadeOut(new_arr_mu), FadeOut(new_arr_sig), FadeOut(new_arr_eps),
            FadeOut(new_arr_z), FadeOut(success_grad1), FadeOut(success_grad2), FadeOut(success_grad3),
            FadeOut(grad_success_text), FadeOut(title_insight)
        )
        self.wait(1)
