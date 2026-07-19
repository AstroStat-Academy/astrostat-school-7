"""Animated Agent anatomy diagram, rendered with manim.

The full diagram is present from the first frame (nothing "appears"):
only the interactions are animated, as message dots traveling along
the arrows, with a soft pulse on the receiving component.

Component shapes: the LLM is a processor chip, the tools are a hexagon
nut, the history is a database cylinder, the user and the human
reviewer are person glyphs. The AGENT container is centered on screen.

Render (from this folder). Manim's own --format=gif produces color
artifacts, so render to mp4 first and convert with a 2-pass palette:
    manim render -qm -r 960,540 --fps 12 \
        fig_agent_anatomy_anim.py AgentAnatomy -o fig_agent_anatomy_anim
    ffmpeg -i media/videos/fig_agent_anatomy_anim/540p12/fig_agent_anatomy_anim.mp4 \
        -vf "fps=10,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=none" \
        -loop 0 fig_agent_anatomy_anim.gif
"""

import numpy as np
from manim import (
    Scene, RoundedRectangle, Rectangle, Ellipse, Circle, Arc, Polygon,
    DashedVMobject, Arrow, DoubleArrow, Text, Dot, VGroup, Line, Union,
    MoveAlongPath, Indicate, ShowPassingFlash, FadeIn, FadeOut,
    config, ITALIC, WHITE, BLACK, PI,
)

# Same palette as the static figure (fig_agent_anatomy.py)
STEELBLUE = "#4682B4"
FIREBRICK = "#B22222"
SEAGREEN  = "#3CB371"
PURPLE_C4 = "#9467bd"
DIMGRAY   = "#696969"
GRAY      = "#808080"

FONT = "Inter"

config.background_color = WHITE


def pt(x, y):
    """Return a manim 3D point from 2D scene coordinates."""
    return np.array([x, y, 0.0])


def label(text, x, y, size=19, color=BLACK, bold=False, italic=False):
    """Place a text label centered at (x, y).

    The text is laid out at 3x the requested size and scaled back down:
    at small font sizes Pango's glyph placement is uneven (stray gaps
    inside words), while a scaled-down large layout stays clean.
    """
    kwargs = {"font": FONT, "font_size": size * 3, "color": color}
    if bold:
        kwargs["weight"] = "BOLD"
    if italic:
        kwargs["slant"] = ITALIC
    text_mobject = Text(text, **kwargs)
    text_mobject.scale(1 / 3)
    text_mobject.move_to(pt(x, y))
    return text_mobject


def make_chip(cx, cy, color):
    """Processor chip: a rounded square with pins on all 4 sides (the LLM)."""
    body = RoundedRectangle(corner_radius=0.12, width=2.2, height=1.6)
    body.set_stroke(color, width=3)
    body.set_fill(color, opacity=0.13)
    body.move_to(pt(cx, cy))

    pins = VGroup()
    for dy in (-0.45, 0.0, 0.45):
        pins.add(Line(pt(cx - 1.1, cy + dy), pt(cx - 1.28, cy + dy)))
        pins.add(Line(pt(cx + 1.1, cy + dy), pt(cx + 1.28, cy + dy)))
    for dx in (-0.6, 0.0, 0.6):
        pins.add(Line(pt(cx + dx, cy + 0.8), pt(cx + dx, cy + 0.98)))
        pins.add(Line(pt(cx + dx, cy - 0.8), pt(cx + dx, cy - 0.98)))
    pins.set_stroke(color, width=3)

    title = label("LLM", cx, cy, size=30, color=color, bold=True)
    return VGroup(body, pins, title)


def make_hexagon(cx, cy, color):
    """Hexagon nut with a flat top (the tools)."""
    radius = 1.15
    vertices = []
    for k in range(6):
        angle = k * PI / 3
        vertices.append(pt(cx + radius * np.cos(angle),
                           cy + radius * np.sin(angle)))
    shape = Polygon(*vertices)
    shape.round_corners(radius=0.08)
    shape.set_stroke(color, width=3)
    shape.set_fill(color, opacity=0.13)

    title = label("TOOLS", cx, cy, size=24, color=color, bold=True)
    return VGroup(shape, title)


def make_cylinder(cx, cy, color):
    """Database cylinder (the history)."""
    width = 2.5
    body_height = 1.2
    ellipse_height = 0.52

    top_y = cy + body_height / 2
    bottom_y = cy - body_height / 2

    # Silhouette: union of the body and the 2 end ellipses, so the fill
    # has a uniform opacity (stacked fills would double up where they
    # overlap).
    body = Rectangle(width=width, height=body_height).move_to(pt(cx, cy))
    top = Ellipse(width=width, height=ellipse_height).move_to(pt(cx, top_y))
    bottom = Ellipse(width=width, height=ellipse_height).move_to(pt(cx, bottom_y))
    silhouette = Union(body, top, bottom)
    silhouette.set_stroke(color, width=3)
    silhouette.set_fill(color, opacity=0.13)

    # The full top ellipse gives the classic database-icon look
    rim = Ellipse(width=width, height=ellipse_height).move_to(pt(cx, top_y))
    rim.set_stroke(color, width=2)
    rim.set_fill(opacity=0)

    title = label("HISTORY", cx, cy - 0.15, size=21, color=color, bold=True)
    return VGroup(silhouette, rim, title)


def make_person(cx, cy, color, scale=1.0):
    """Person glyph: a head circle over half-disk shoulders."""
    shoulders = Arc(radius=0.3 * scale, start_angle=0, angle=PI)
    shoulders.set_stroke(width=0)
    shoulders.set_fill(color, opacity=0.8)
    shoulders.move_arc_center_to(pt(cx, cy))

    head = Circle(radius=0.17 * scale)
    head.set_stroke(width=0)
    head.set_fill(color, opacity=0.8)
    head.move_to(pt(cx, cy + 0.42 * scale))
    return VGroup(shoulders, head)


def make_card(cx, cy, width, height, color, dashed=False):
    """Rounded card used as the frame of the external actors."""
    card = RoundedRectangle(corner_radius=0.12, width=width, height=height)
    card.set_stroke(color, width=3)
    card.set_fill(color, opacity=0.10)
    card.move_to(pt(cx, cy))
    if dashed:
        card = DashedVMobject(card, num_dashes=45)
    return card


def make_arrow(x0, y0, x1, y1, color=GRAY, dashed=False, double=False):
    """Straight arrow between 2 points."""
    if double:
        arrow = DoubleArrow(pt(x0, y0), pt(x1, y1), buff=0,
                            stroke_width=3, tip_length=0.18, color=color)
    else:
        arrow = Arrow(pt(x0, y0), pt(x1, y1), buff=0,
                      stroke_width=3, tip_length=0.18, color=color)
    if dashed:
        arrow = DashedVMobject(arrow, num_dashes=18)
    return arrow


class AgentAnatomy(Scene):
    """One full Agent loop: task -> tool call -> history -> review -> answer."""

    def send(self, x0, y0, x1, y1, color, receiver=None):
        """Animate a message dot traveling from (x0, y0) to (x1, y1)."""
        path = Line(pt(x0, y0), pt(x1, y1))
        dot = Dot(pt(x0, y0), radius=0.09, color=color)

        self.play(FadeIn(dot, run_time=0.15))
        self.play(
            MoveAlongPath(dot, path),
            ShowPassingFlash(path.copy().set_stroke(color, width=6),
                             time_width=0.5),
            run_time=0.9,
        )
        if receiver is None:
            self.play(FadeOut(dot, run_time=0.15))
        else:
            self.play(
                FadeOut(dot, run_time=0.15),
                Indicate(receiver, scale_factor=1.06, color=color,
                         run_time=0.6),
            )

    def construct(self):
        # --- Static diagram: everything is on screen from frame 0 ---------

        # Agent container, horizontally centered on screen
        container = RoundedRectangle(corner_radius=0.18, width=9.6, height=5.3)
        container.set_stroke(DIMGRAY, width=2.5)
        container.set_fill("#f7f7f7", opacity=1.0)
        container.move_to(pt(0, 0.55))
        agent_label = label("AGENT", -4.0, 2.72, size=26,
                            color=DIMGRAY, bold=True)

        # Components inside the Agent
        llm = make_chip(-2.0, 0.55, STEELBLUE)
        tools = make_hexagon(2.7, 1.75, FIREBRICK)
        history = make_cylinder(2.7, -0.75, SEAGREEN)

        # External actors
        user_card = make_card(-6.0, 0.55, 1.8, 1.9, GRAY)
        user_glyph = make_person(-6.0, 0.62, GRAY)
        user_title = label("USER", -6.0, 0.05, size=21, color=DIMGRAY, bold=True)
        user = VGroup(user_card, user_glyph, user_title)

        human_card = make_card(-2.0, -3.05, 4.0, 1.1, PURPLE_C4, dashed=True)
        human_glyph = make_person(-3.5, -3.2, PURPLE_C4, scale=0.85)
        human_title = label("HUMAN-IN-THE-LOOP", -1.7, -2.86, size=18,
                            color=PURPLE_C4, bold=True)
        human_note = label("(optional)", -1.7, -3.26, size=15, color=DIMGRAY)
        human = VGroup(human_card, human_glyph, human_title, human_note)

        # Arrows and their labels
        a_task = make_arrow(-5.1, 1.0, -3.3, 1.0)
        l_task = label("task", -4.2, 1.28, italic=True)
        a_answer = make_arrow(-3.3, 0.1, -5.1, 0.1)
        l_answer = label("answer", -4.2, -0.2, italic=True)
        a_call = make_arrow(-0.7, 1.05, 1.55, 1.75, color=FIREBRICK)
        l_call = label("call", 0.35, 1.72, italic=True)
        a_result = make_arrow(1.75, 1.4, -0.7, 0.65, color=FIREBRICK)
        l_result = label("result", 0.65, 0.62, italic=True)
        a_history = make_arrow(-0.7, -0.2, 1.45, -0.75,
                               color=SEAGREEN, double=True)
        l_history = label("read / write", 0.0, -0.95, italic=True)
        a_review = make_arrow(-2.0, -0.55, -2.0, -2.42,
                              color=PURPLE_C4, dashed=True, double=True)
        l_review = label("review / approve", -0.85, -1.45, italic=True)

        self.add(container, agent_label,
                 llm, tools, history, user, human,
                 a_task, l_task, a_answer, l_answer,
                 a_call, l_call, a_result, l_result,
                 a_history, l_history, a_review, l_review)

        # --- Animated interactions: 1 full Agent loop ---------------------

        self.wait(0.6)

        # The user sends the task; the LLM reasons
        self.send(-5.1, 1.0, -3.3, 1.0, GRAY, receiver=llm)

        # The LLM decides to call a tool; the tool returns a result
        self.send(-0.7, 1.05, 1.55, 1.75, FIREBRICK, receiver=tools)
        self.send(1.75, 1.4, -0.7, 0.65, FIREBRICK, receiver=llm)

        # The LLM stores and re-reads the conversation history
        self.send(-0.7, -0.2, 1.45, -0.75, SEAGREEN, receiver=history)
        self.send(1.45, -0.75, -0.7, -0.2, SEAGREEN, receiver=llm)

        # Optionally, a human reviews the step before it is finalized
        self.send(-2.0, -0.55, -2.0, -2.42, PURPLE_C4, receiver=human)
        self.send(-2.0, -2.42, -2.0, -0.55, PURPLE_C4, receiver=llm)

        # The LLM sends the final answer back to the user
        self.send(-3.3, 0.1, -5.1, 0.1, GRAY, receiver=user)

        self.wait(1.0)
