from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class CNNConvLayerForwardPassScene(VoiceoverScene):
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(GTTSService())

        # ==========================================
        # SECTION 1: What is it?
        # ==========================================
        title = Text("Convolutional Neural Networks (CNN):", color=BLACK, font_size=40).to_edge(UP)
        subtitle = Text("The Convolution Layer Forward Pass", color=BLUE, font_size=36).next_to(title, DOWN)

        with self.voiceover(text="""Welcome to Day 39 of our AI and Machine Learning interview preparation series.
            Today, we are going deep into the architecture of Convolutional Neural Networks, specifically focusing
            on the forward pass of the fundamental building block: the Convolution Layer.
            So, what exactly is a convolution layer?""") as tracker:
            self.play(Write(title))
            self.play(FadeIn(subtitle))
            self.wait(1.5)

        def_text = Text("A linear operation that applies a sliding filter to extract spatial features.",
                        color=BLACK, font_size=28).next_to(subtitle, DOWN, buff=0.5)

        with self.voiceover(text="""At its core, a convolution layer performs a specialized linear operation.
            Instead of general matrix multiplication used in standard dense layers, it applies a small sliding
            window, known as a filter or kernel, across the input data. The goal of this sliding filter is to
            extract local spatial features, such as edges, textures, or specific patterns.""") as tracker:
            self.play(Write(def_text))
            self.wait(1.5)

        with self.voiceover(text="Now let's see how this works visually.") as tracker:
            self.play(FadeOut(def_text))

        # Visualizing the sliding filter
        # Create input matrix (5x5)
        input_matrix = VGroup()
        for i in range(5):
            row = VGroup()
            for j in range(5):
                val = (i * 2 + j) % 3
                cell = VGroup(
                    Square(side_length=0.8, color=BLACK, stroke_width=2),
                    Text(str(val), color=BLACK, font_size=24)
                )
                row.add(cell)
            row.arrange(RIGHT, buff=0)
            input_matrix.add(row)
        input_matrix.arrange(DOWN, buff=0)

        input_label = Text("Input Image (5x5)", color=BLACK, font_size=24).next_to(input_matrix, UP)
        input_group = VGroup(input_label, input_matrix).scale(0.8).move_to(LEFT * 3)

        # Create filter matrix (3x3)
        filter_matrix = VGroup()
        filter_vals = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
        for i in range(3):
            row = VGroup()
            for j in range(3):
                cell = VGroup(
                    Square(side_length=0.8, color=RED, stroke_width=2, fill_opacity=0.2, fill_color=RED),
                    Text(str(filter_vals[i][j]), color=RED, font_size=24)
                )
                row.add(cell)
            row.arrange(RIGHT, buff=0)
            filter_matrix.add(row)
        filter_matrix.arrange(DOWN, buff=0)

        filter_label = Text("Filter (3x3)", color=RED, font_size=24).next_to(filter_matrix, UP)
        filter_group = VGroup(filter_label, filter_matrix).scale(0.8).next_to(input_group, RIGHT, buff=1)

        with self.voiceover(text="""Let's visualize this process. Imagine we have a simplified input image,
            represented here as a 5 by 5 matrix of pixel intensities. We also have our filter, a smaller 3 by 3 matrix
            containing learnable weights. In this example, our filter is designed to detect vertical edges.""") as tracker:
            self.play(FadeIn(input_group))
            self.play(FadeIn(filter_group))
            self.wait(1.5)

        # Create output matrix (3x3)
        output_matrix = VGroup()
        for i in range(3):
            row = VGroup()
            for j in range(3):
                cell = VGroup(
                    Square(side_length=0.8, color=BLACK, stroke_width=2),
                    Text("", color=BLUE, font_size=24)
                )
                row.add(cell)
            row.arrange(RIGHT, buff=0)
            output_matrix.add(row)
        output_matrix.arrange(DOWN, buff=0)

        output_label = Text("Feature Map (3x3)", color=BLUE, font_size=24).next_to(output_matrix, UP)
        output_group = VGroup(output_label, output_matrix).scale(0.8).next_to(filter_group, RIGHT, buff=1)

        with self.voiceover(text="""During the forward pass, this filter slides, or convolves, across the
            input image. At each position, we perform an element-wise multiplication between the filter weights
            and the corresponding patch of the input image, and then sum the results. This single scalar value
            becomes one pixel in our new output matrix, which we call a feature map. By sliding the filter over
            the entire input, we construct the complete feature map, which highlights where the filter's specific
            pattern was found in the original image.""") as tracker:
            self.play(FadeIn(output_group))

            # Animation of sliding filter
            moving_filter = filter_matrix.copy()
            moving_filter.move_to(input_matrix[0][0].get_center())
            self.play(FadeIn(moving_filter))

            # First position (sliding to top-left aligned)
            self.play(moving_filter.animate.move_to(input_matrix[1][1].get_center()), run_time=1)

            # Fill first output
            out_val_text = Text("3", color=BLUE, font_size=24).move_to(output_matrix[0][0][0].get_center())
            self.play(Write(out_val_text))

            # Second position (sliding right)
            self.play(moving_filter.animate.move_to(input_matrix[1][2].get_center()), run_time=1)

            out_val_text2 = Text("-1", color=BLUE, font_size=24).move_to(output_matrix[0][1][0].get_center())
            self.play(Write(out_val_text2))

            self.play(FadeOut(moving_filter))
            self.wait(1.5)

        # Math formula
        math_text1 = MathTex(r"\text{Output}[i, j] = ", color=BLACK, font_size=32)
        math_text2 = MathTex(r"\text{Output}[i, j] = ", r"\sum_{m} \sum_{n}", color=BLACK, font_size=32)
        math_text3 = MathTex(r"\text{Output}[i, j] = ", r"\sum_{m} \sum_{n}", r"\text{Input}[i+m, j+n]", color=BLACK, font_size=32)
        math_text4 = MathTex(r"\text{Output}[i, j] = ", r"\sum_{m} \sum_{n}", r"\text{Input}[i+m, j+n]", r"\cdot \text{Filter}[m, n]", color=BLACK, font_size=32)

        math_group = VGroup(math_text1).next_to(subtitle, DOWN, buff=0.5)

        with self.voiceover(text="""Mathematically, the forward pass for a single output pixel at position i, j
            is computed as a double summation over the spatial dimensions of the filter. We multiply the input pixels
            by the corresponding filter weights and sum them all up. We do this for every position where the filter fits
            on the input image.""") as tracker:
            self.play(FadeOut(input_group), FadeOut(filter_group), FadeOut(output_group))
            self.play(Write(math_text1))
            self.play(TransformMatchingTex(math_text1, math_text2))
            self.play(TransformMatchingTex(math_text2, math_text3))
            self.play(TransformMatchingTex(math_text3, math_text4))
            self.wait(1.5)


        # ==========================================
        # SECTION 2: Why do we need it?
        # ==========================================
        with self.voiceover(text="""Now that we understand what a convolution layer is, a crucial question arises:
            Why do we need it? Why can't we just use standard dense, or fully connected, layers for image data?""") as tracker:
            self.play(FadeOut(math_text4))
            self.wait(1)

        why_title = Text("Why Convolution over Dense Layers?", color=BLUE, font_size=36).next_to(title, DOWN)

        with self.voiceover(text="""Let's look at a comparison. First, consider how a standard dense layer processes
            an image.""") as tracker:
            self.play(Transform(subtitle, why_title))
            self.wait(1)

        # Before (Dense Layer)
        dense_label = Text("Without CNN (Dense Layer):", color=RED, font_size=28).move_to(UP * 1.5 + LEFT * 3)
        image_grid = VGroup()
        for i in range(4):
            row = VGroup()
            for j in range(4):
                color = BLUE if (i+j)%2==0 else GREEN
                cell = Square(side_length=0.4, color=BLACK, fill_opacity=0.6, fill_color=color)
                row.add(cell)
            row.arrange(RIGHT, buff=0)
            image_grid.add(row)
        image_grid.arrange(DOWN, buff=0).next_to(dense_label, DOWN)

        flattened_array = VGroup()
        for i in range(16):
            color = BLUE if (i//4 + i%4)%2==0 else GREEN
            cell = Square(side_length=0.3, color=BLACK, fill_opacity=0.6, fill_color=color)
            flattened_array.add(cell)
        flattened_array.arrange(DOWN, buff=0).next_to(image_grid, RIGHT, buff=1.5)

        arrow_flatten = Arrow(start=image_grid.get_right(), end=flattened_array.get_left(), color=BLACK)
        flatten_text = Text("Flatten", color=BLACK, font_size=20).next_to(arrow_flatten, UP)

        dense_nodes = VGroup(*[Circle(radius=0.15, color=BLACK, fill_opacity=1, fill_color=RED) for _ in range(5)])
        dense_nodes.arrange(DOWN, buff=0.2).next_to(flattened_array, RIGHT, buff=1.5)

        connections = VGroup()
        for i in range(16):
            for j in range(5):
                connections.add(Line(flattened_array[i].get_right(), dense_nodes[j].get_left(), color=BLACK, stroke_width=0.5, stroke_opacity=0.3))

        with self.voiceover(text="""To feed a 2D image into a dense layer, we must first flatten it into a single
            1D vector. As you can see, this flattening process completely destroys the spatial structure and topology
            of the image. The network loses the information about which pixels are next to each other. Furthermore, every
            input pixel is connected to every neuron, leading to an explosion in the number of parameters, making the
            model highly prone to overfitting and computationally very expensive.""") as tracker:
            self.play(Write(dense_label))
            self.play(FadeIn(image_grid))
            self.play(GrowArrow(arrow_flatten), FadeIn(flatten_text))
            self.play(TransformFromCopy(image_grid, flattened_array))
            self.play(FadeIn(dense_nodes))
            self.play(FadeIn(connections))
            self.wait(1.5)

        with self.voiceover(text="Moving on to the next concept.") as tracker:
            self.play(FadeOut(VGroup(dense_label, image_grid, arrow_flatten, flatten_text, flattened_array, dense_nodes, connections)))

        # After (CNN)
        cnn_label = Text("With CNN (Convolution Layer):", color=GREEN, font_size=28).move_to(UP * 1.5 + LEFT * 3)

        image_grid_cnn = image_grid.copy().next_to(cnn_label, DOWN)

        filter_box = Square(side_length=0.8, color=RED, stroke_width=4).move_to(image_grid_cnn[0][0].get_center() + DR*0.2)

        feature_map_cnn = VGroup()
        for i in range(3):
            row = VGroup()
            for j in range(3):
                cell = Square(side_length=0.4, color=BLACK, fill_opacity=0.6, fill_color=PURPLE)
                row.add(cell)
            row.arrange(RIGHT, buff=0)
            feature_map_cnn.add(row)
        feature_map_cnn.arrange(DOWN, buff=0).next_to(image_grid_cnn, RIGHT, buff=2)

        arrow_conv = Arrow(start=image_grid_cnn.get_right(), end=feature_map_cnn.get_left(), color=BLACK)
        conv_text = Text("Convolve", color=BLACK, font_size=20).next_to(arrow_conv, UP)

        with self.voiceover(text="""In contrast, a convolution layer preserves the 2D spatial structure of the input.
            By sliding a small filter across the image, it processes patches of local pixels together, recognizing that
            neighboring pixels have strong local correlations. This approach dramatically reduces the number of parameters
            because the same filter weights are shared across the entire image. This property, known as parameter sharing,
            makes the network much more efficient and less prone to overfitting while successfully capturing spatial hierarchies.""") as tracker:
            self.play(Write(cnn_label))
            self.play(FadeIn(image_grid_cnn))
            self.play(FadeIn(filter_box))
            self.play(GrowArrow(arrow_conv), FadeIn(conv_text))
            self.play(FadeIn(feature_map_cnn))

            # Slide filter a bit
            self.play(filter_box.animate.move_to(image_grid_cnn[1][1].get_center() + DR*0.2), run_time=2)
            self.wait(1.5)


        # ==========================================
        # SECTION 3: Use Cases
        # ==========================================
        with self.voiceover(text="Let's look at some real world use cases.") as tracker:
            self.play(FadeOut(VGroup(cnn_label, image_grid_cnn, filter_box, arrow_conv, conv_text, feature_map_cnn)))

        use_cases_title = Text("Real-World Use Cases", color=BLUE, font_size=36).next_to(title, DOWN)

        with self.voiceover(text="""Because convolution layers are so incredibly effective at processing grid-like
            topology, such as image pixels, they are the backbone of modern computer vision systems.""") as tracker:
            self.play(Transform(subtitle, use_cases_title))
            self.wait(1)

        case1 = VGroup(
            Text("1. Autonomous Driving (Tesla Autopilot)", color=BLACK, font_size=28, weight=BOLD),
            Text("Uses CNNs for real-time object detection, identifying pedestrians, vehicles, \nand lane markings from camera feeds.", color=BLACK, font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT).move_to(LEFT * 1 + UP * 0.5)

        case2 = VGroup(
            Text("2. Image Search & Tagging (Google Photos)", color=BLACK, font_size=28, weight=BOLD),
            Text("Uses deep CNNs to automatically categorize photos, recognize faces, \nand allow users to search for objects like 'dog' or 'beach'.", color=BLACK, font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(case1, DOWN, buff=1)

        with self.voiceover(text="""For example, Tesla Autopilot heavily relies on deep convolutional networks to
            process video feeds from its cameras. The convolution layers extract features necessary for real-time object
            detection, allowing the system to identify pedestrians, other vehicles, and lane markings with high accuracy.""") as tracker:
            self.play(FadeIn(case1))
            self.wait(1.5)

        with self.voiceover(text="""Another ubiquitous example is Google Photos. It utilizes massive CNN architectures
            to automatically categorize your uploaded photos, recognize specific faces, and extract semantic features that
            allow you to effortlessly search your entire library using text queries like 'dog' or 'beach'.""") as tracker:
            self.play(FadeIn(case2))
            self.wait(1.5)

        with self.voiceover(text="Now, for the key interview insight.") as tracker:
            self.play(FadeOut(case1), FadeOut(case2))


        # ==========================================
        # SECTION 4: Key Interview Insight
        # ==========================================
        insight_title = Text("Key Interview Insight", color=RED, font_size=36).next_to(title, DOWN)

        with self.voiceover(text="""When discussing CNNs in a machine learning interview, there is a specific concept
            that interviewers love to test you on to gauge your practical understanding of network architecture design.""") as tracker:
            self.play(Transform(subtitle, insight_title))
            self.wait(1)

        box = Rectangle(width=12, height=4.5, color=RED, fill_color=RED_E, fill_opacity=0.05, stroke_width=3).move_to(DOWN*1)
        insight_heading = Text("The Receptive Field vs. Computational Cost Tradeoff", color=RED, font_size=28, weight=BOLD).next_to(box.get_top(), DOWN, buff=0.3)

        insight_point1 = Text("• Larger filters (e.g., 5x5, 7x7) increase the Receptive Field faster...", color=BLACK, font_size=24)
        insight_point2 = Text("• ...BUT they drastically increase computational cost and parameters.", color=BLACK, font_size=24)
        insight_point3 = Text("• The 'Gotcha': Stacking multiple small filters (e.g., two 3x3s) gives the same \n  effective receptive field as one 5x5, but with fewer parameters and more non-linearity.", color=BLUE, font_size=24)

        insight_points = VGroup(insight_point1, insight_point2, insight_point3).arrange(DOWN, aligned_edge=LEFT, buff=0.4).next_to(insight_heading, DOWN, buff=0.4)
        insight_points.align_to(insight_heading, LEFT)

        with self.voiceover(text="""The most common interview 'gotcha' centers around the tradeoff between the
            Receptive Field and Computational Cost when choosing filter sizes.""") as tracker:
            self.play(Create(box))
            self.play(Write(insight_heading))
            self.wait(1)

        with self.voiceover(text="""Interviewers might ask: 'Why not just use large filters, like 5 by 5 or 7 by 7,
            in the early layers to capture larger patterns faster?' While it's true that larger filters increase the
            receptive field more quickly...""") as tracker:
            self.play(FadeIn(insight_point1))
            self.wait(1)

        with self.voiceover(text="""...the tradeoff is that they drastically increase the number of parameters and
            the overall computational cost of the layer.""") as tracker:
            self.play(FadeIn(insight_point2))
            self.wait(1)

        with self.voiceover(text="""The key insight, and the answer they are looking for, is that modern architectures
            often stack multiple smaller filters instead. For example, stacking two successive 3 by 3 convolutional layers
            provides the exact same effective receptive field of 5 by 5 as a single 5 by 5 layer. However, the stacked
            3 by 3 approach uses significantly fewer parameters, and it allows you to inject an additional non-linear
            activation function between the layers, making the network more expressive. Understanding this architectural
            choice shows you have deep, practical knowledge of CNN design.""") as tracker:
            self.play(FadeIn(insight_point3))
            self.wait(2)

        with self.voiceover(text="""That concludes our deep dive into the forward pass of a Convolutional Layer.
            Keep practicing these concepts, and good luck with your AI engineering interviews!""") as tracker:
            self.play(FadeOut(VGroup(title, subtitle, box, insight_heading, insight_points)))
            self.wait(1)
