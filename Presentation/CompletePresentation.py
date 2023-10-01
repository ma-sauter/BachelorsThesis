# example.py

from manim import *

# or: from manimlib import *
from manim_slides import Slide

from Titlepage import TitleIntro
from NTKExplanation import NTKExplanation


class Presentation(Slide):
    def construct(self):
        TitleIntro.construct(self)
        NTKExplanation.construct(self)
