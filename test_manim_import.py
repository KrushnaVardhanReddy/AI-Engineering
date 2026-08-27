import manim
print(manim.__version__)
try:
    from manim_voiceover import VoiceoverScene
    from manim_voiceover.services.gtts import GTTSService
    print("manim_voiceover successfully imported.")
except ImportError:
    print("manim_voiceover not installed.")
