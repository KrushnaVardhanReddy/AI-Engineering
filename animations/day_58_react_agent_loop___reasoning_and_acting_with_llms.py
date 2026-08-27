from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService

class ReActAgentLoop(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE

        # Section 1: What is it?
        title = Tex("The ReAct Agent Loop", color=BLACK, font_size=60)
        subtitle = Tex("Reasoning and Acting with LLMs", color=BLUE, font_size=40)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)

        definition = Tex(
            r"A paradigm where an AI dynamically interleaves \\",
            r"internal \textbf{reasoning} with external \textbf{actions} \\",
            r"to solve complex problems.",
            color=BLACK, font_size=36
        )
        definition.next_to(title_group, DOWN, buff=1.0)

        with self.voiceover(
            text="Welcome to Day 58 of the AI Engineering series! "
                 "Today, we are diving deep into the ReAct Agent Loop, which stands for Reasoning and Acting. "
                 "So, what exactly is the ReAct framework? "
                 "At its core, ReAct is a powerful paradigm where an artificial intelligence model dynamically interleaves "
                 "its internal reasoning capabilities with the ability to take external actions in the real world or in a software environment "
                 "in order to solve complex, multi-step problems."
        ) as tracker:
            self.play(Write(title_group))
            self.wait(1)
            self.play(FadeIn(definition, shift=UP))
            self.wait(1.5)

        self.play(FadeOut(definition))

        # Draw the loop
        query = Tex("User Query", color=PURPLE, font_size=36).to_edge(UP, buff=1.5)

        thought_box = RoundedRectangle(corner_radius=0.2, width=3, height=1.5, color=BLUE, fill_opacity=0.1)
        thought_text = Tex(r"\textbf{Thought:}\\Analyze the current state.", color=BLACK, font_size=28)
        thought_group = VGroup(thought_box, thought_text)

        action_box = RoundedRectangle(corner_radius=0.2, width=3, height=1.5, color=RED, fill_opacity=0.1)
        action_text = Tex(r"\textbf{Action:}\\Call an external tool/API.", color=BLACK, font_size=28)
        action_group = VGroup(action_box, action_text)

        obs_box = RoundedRectangle(corner_radius=0.2, width=3, height=1.5, color=GREEN, fill_opacity=0.1)
        obs_text = Tex(r"\textbf{Observation:}\\Receive the tool's result.", color=BLACK, font_size=28)
        obs_group = VGroup(obs_box, obs_text)

        # Position the nodes in a triangle
        thought_group.move_to(LEFT * 4 + DOWN * 1)
        action_group.move_to(RIGHT * 4 + DOWN * 1)
        obs_group.move_to(DOWN * 3)

        # Arrows
        arrow_q_t = Arrow(query.get_bottom(), thought_group.get_top(), color=BLACK, buff=0.2)
        arrow_t_a = Arrow(thought_group.get_right(), action_group.get_left(), color=BLACK, buff=0.2)
        arrow_a_o = Arrow(action_group.get_bottom(), obs_group.get_right() + UP * 0.5, color=BLACK, buff=0.2, path_arc=-0.5)
        arrow_o_t = Arrow(obs_group.get_left() + UP * 0.5, thought_group.get_bottom(), color=BLACK, buff=0.2, path_arc=-0.5)

        with self.voiceover(
            text="Let's trace how this works in practice. It all starts with a User Query. "
                 "Unlike a standard LLM that just tries to guess the final answer immediately, a ReAct agent pauses to generate a 'Thought'. "
                 "In this step, the model explicitly analyzes the current state of the problem and decides what to do next. "
                 "Next, based on that thought, it executes an 'Action'. This could involve searching the web, calling a calculator, or querying a database. "
                 "The environment then returns an 'Observation', which is the concrete result of that action. "
                 "The agent takes this new observation and loops back to generate a new thought, continuing this cycle until it reaches a final, grounded answer."
        ) as tracker:
            self.play(FadeIn(query, shift=DOWN))
            self.wait(1)
            self.play(FadeIn(arrow_q_t), FadeIn(thought_group))
            self.wait(1.5)
            self.play(FadeIn(arrow_t_a), FadeIn(action_group))
            self.wait(1.5)
            self.play(FadeIn(arrow_a_o), FadeIn(obs_group))
            self.wait(1.5)
            self.play(FadeIn(arrow_o_t))
            self.wait(1.5)

        self.play(FadeOut(VGroup(title_group, query, thought_group, action_group, obs_group, arrow_q_t, arrow_t_a, arrow_a_o, arrow_o_t)))
        self.wait(0.5)

        # Section 2: Why do we need it?
        why_title = Tex("Why do we need ReAct?", color=BLACK, font_size=50).to_edge(UP)

        # Standard LLM (Before)
        standard_title = Tex("Standard LLM (Before)", color=RED, font_size=40).move_to(LEFT * 3.5 + UP * 2)
        std_query = Tex(r"\textbf{User:}", " What's the weather in Tokyo right now?", color=BLACK, font_size=28).next_to(standard_title, DOWN, buff=0.5)
        std_ans = Tex(r"\textbf{LLM:}", " Based on my training data, Tokyo is... (Hallucination)", color=BLACK, font_size=28).next_to(std_query, DOWN, buff=0.5)
        cross = Cross(std_ans, stroke_color=RED, stroke_width=4)

        # ReAct Agent (After)
        react_title = Tex("ReAct Agent (After)", color=GREEN, font_size=40).move_to(RIGHT * 3.5 + UP * 2)
        re_query = Tex(r"\textbf{User:}", " What's the weather in Tokyo right now?", color=BLACK, font_size=28).next_to(react_title, DOWN, buff=0.5)
        re_thought = Tex(r"\textbf{Thought:}", r" I don't know current weather.\\I should search.", color=BLUE, font_size=24).next_to(re_query, DOWN, buff=0.3)
        re_action = Tex(r"\textbf{Action:}", r" \texttt{get\_weather(location='Tokyo')}", color=RED, font_size=24).next_to(re_thought, DOWN, buff=0.3)
        re_obs = Tex(r"\textbf{Obs:}", " 75°F, Sunny.", color=GREEN, font_size=24).next_to(re_action, DOWN, buff=0.3)
        re_ans = Tex(r"\textbf{LLM:}", r" It is currently 75°F\\and sunny in Tokyo.", color=BLACK, font_size=28).next_to(re_obs, DOWN, buff=0.3)
        check = Tex(r"\checkmark", color=GREEN, font_size=48).next_to(re_ans, RIGHT)

        # Divider
        divider = Line(UP * 2.5, DOWN * 3, color=BLACK)

        with self.voiceover(
            text="So, why do we actually need this complicated loop? Why not just ask the model directly? "
                 "Let's look at a before and after comparison. "
                 "First, consider a standard Large Language Model. If a user asks, 'What's the weather in Tokyo right now?', "
                 "the standard LLM cannot browse the internet by default. It might try to guess or hallucinate an answer based on outdated training data, which is completely unreliable."
        ) as tracker:
            self.play(Write(why_title), FadeIn(divider))
            self.play(FadeIn(standard_title, shift=UP))
            self.play(Write(std_query))
            self.wait(1)
            self.play(FadeIn(std_ans, shift=UP))
            self.play(FadeIn(cross))
            self.wait(1.5)

        with self.voiceover(
            text="Now, let's look at the After scenario with a ReAct Agent. "
                 "When given the exact same query, the ReAct agent doesn't just guess. "
                 "First, it generates a Thought: 'I don't know the current weather. I should use a search tool.' "
                 "Then, it executes an Action: it calls the 'get_weather' API with the location set to Tokyo. "
                 "The API returns a factual Observation: '75 degrees Fahrenheit and Sunny'. "
                 "Finally, armed with this grounded truth, the agent synthesizes the final answer: 'It is currently 75 degrees and sunny in Tokyo.' "
                 "This entirely eliminates the hallucination problem for temporal or factual queries."
        ) as tracker:
            self.play(FadeIn(react_title, shift=UP))
            self.play(Write(re_query))
            self.wait(1)
            self.play(FadeIn(re_thought, shift=RIGHT))
            self.wait(1)
            self.play(FadeIn(re_action, shift=RIGHT))
            self.wait(1)
            self.play(FadeIn(re_obs, shift=RIGHT))
            self.wait(1)
            self.play(FadeIn(re_ans, shift=UP), Write(check))
            self.wait(1.5)

        self.play(FadeOut(VGroup(why_title, divider, standard_title, std_query, std_ans, cross, react_title, re_query, re_thought, re_action, re_obs, re_ans, check)))
        self.wait(0.5)

        # Section 3: Use Cases
        usecase_title = Tex("Real-World Use Cases", color=BLACK, font_size=50).to_edge(UP)

        # Use case 1
        uc1_box = Rectangle(width=10, height=2.5, color=BLUE, fill_opacity=0.05).move_to(UP * 1)
        uc1_text = Tex(r"\textbf{1. Advanced Data Analysis (e.g., ChatGPT)}", color=BLACK, font_size=36).move_to(uc1_box.get_top() + DOWN * 0.4)
        uc1_desc = Tex(r"When asked to solve complex math, the agent \\",
                       r"\textbf{Thoughts:} 'I should write a Python script.' \\",
                       r"\textbf{Actions:} Executes code in a sandbox to get the exact answer.", color=BLACK, font_size=28).next_to(uc1_text, DOWN, buff=0.3)

        # Use case 2
        uc2_box = Rectangle(width=10, height=2.5, color=GREEN, fill_opacity=0.05).move_to(DOWN * 2)
        uc2_text = Tex(r"\textbf{2. API Integration (e.g., Spotify AI DJ)}", color=BLACK, font_size=36).move_to(uc2_box.get_top() + DOWN * 0.4)
        uc2_desc = Tex(r"When asked for 'Indie Rock from 2010', the agent \\",
                       r"\textbf{Thoughts:} 'I need to query the Spotify catalog.' \\",
                       r"\textbf{Actions:} Calls internal APIs to fetch tracks and build a playlist.", color=BLACK, font_size=28).next_to(uc2_text, DOWN, buff=0.3)

        with self.voiceover(
            text="Let's ground this concept with two real-world use cases where ReAct patterns are actively deployed today. "
                 "Our first example is Advanced Data Analysis, similar to what you see in ChatGPT's Code Interpreter. "
                 "When asked to solve a complex math problem or analyze a CSV file, the agent uses the ReAct loop. "
                 "It thinks: 'I can't calculate this in my head reliably, so I should write a Python script.' "
                 "It then acts by generating and running the code in a secure sandbox, observing the output, and returning the mathematically perfect result."
        ) as tracker:
            self.play(Write(usecase_title))
            self.play(FadeIn(uc1_box), Write(uc1_text))
            self.wait(0.5)
            self.play(FadeIn(uc1_desc))
            self.wait(1.5)

        with self.voiceover(
            text="Our second example involves API integrations, like building an AI DJ for a platform such as Spotify. "
                 "If a user asks for 'upbeat Indie Rock from the year 2010', the language model itself doesn't contain a searchable audio database. "
                 "Instead, the agent thinks: 'I need to query the catalog.' It then acts by calling Spotify's internal APIs, passing the correct genre and year filters, "
                 "observes the JSON response containing the track IDs, and dynamically constructs a personalized playlist for the user."
        ) as tracker:
            self.play(FadeIn(uc2_box), Write(uc2_text))
            self.wait(0.5)
            self.play(FadeIn(uc2_desc))
            self.wait(1.5)

        self.play(FadeOut(VGroup(usecase_title, uc1_box, uc1_text, uc1_desc, uc2_box, uc2_text, uc2_desc)))
        self.wait(0.5)

        # Section 4: Key Interview Insight
        insight_title = Tex("Key Interview Insight", color=PURPLE, font_size=50).to_edge(UP)

        callout_box = RoundedRectangle(corner_radius=0.3, width=11, height=5, color=PURPLE, fill_color=PURPLE, fill_opacity=0.05).move_to(DOWN * 0.5)

        tradeoff_title = Tex(r"\textbf{The Tradeoff: Capabilities vs. Latency \& Cost}", color=BLACK, font_size=40).move_to(callout_box.get_top() + DOWN * 0.7)

        token_eq1 = MathTex(r"\text{Context}_{t=1}", r"=", r"Q", r"+", r"T_1", r"+", r"A_1", r"+", r"O_1", color=BLACK, font_size=36)
        token_eq2 = MathTex(r"\text{Context}_{t=2}", r"=", r"\text{Context}_{t=1}", r"+", r"T_2", r"+", r"A_2", r"+", r"O_2", color=BLACK, font_size=36)
        token_eq3 = MathTex(r"\text{Context}_{t=n}", r"\rightarrow", r"\textbf{Context Window Exhaustion!}", color=RED, font_size=36)

        token_group = VGroup(token_eq1, token_eq2, token_eq3).arrange(DOWN, buff=0.5).next_to(tradeoff_title, DOWN, buff=0.8)

        with self.voiceover(
            text="If you are interviewing for an AI Engineering role, this is the most critical part to understand. "
                 "Interviewers will almost always ask you about the drawbacks of the ReAct framework. "
                 "The key insight here is the massive tradeoff between Agentic Capabilities versus Latency and Cost. "
                 "Every time the agent loops, it must append its previous Thoughts, Actions, and Observations to its prompt. "
        ) as tracker:
            self.play(Write(insight_title))
            self.play(FadeIn(callout_box), Write(tradeoff_title))
            self.wait(1)

        with self.voiceover(
            text="Let's look at the math. In step one, the context contains the initial query, plus the first thought, action, and observation. "
        ) as tracker:
            self.play(Write(token_eq1))
            self.wait(1)

        with self.voiceover(
            text="In step two, the entire history from step one is fed back into the model, along with the new thought, action, and observation. "
                 "Because LLMs charge per token and compute attention quadratically, this means every single iteration of the loop gets slower and significantly more expensive."
        ) as tracker:
            self.play(TransformMatchingTex(token_eq1.copy(), token_eq2))
            self.wait(1)

        with self.voiceover(
            text="Eventually, if the agent gets stuck in an infinite loop or requires too many steps, it will hit a critical failure: Context Window Exhaustion. "
                 "To mitigate this in production, you must implement strict mechanisms, such as a 'maximum steps' limit, or dynamic context summarization techniques, "
                 "to prevent the agent from burning through your API budget or crashing due to token limits. "
                 "Understanding this architectural vulnerability is what separates junior developers from senior AI engineers."
        ) as tracker:
            self.play(TransformMatchingTex(token_eq2.copy(), token_eq3))
            self.wait(1.5)

        with self.voiceover(
            text="That concludes our deep dive into the ReAct Agent loop. You now understand not just how it works, but how to deploy it safely in production. "
                 "Happy coding, and see you in the next lesson!"
        ) as tracker:
            self.play(FadeOut(VGroup(insight_title, callout_box, tradeoff_title, token_group)))
            self.wait(2)
