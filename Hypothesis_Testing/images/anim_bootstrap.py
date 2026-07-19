"""Manim animation of bootstrap resampling.

Left: the original sample, shown as 9 scattered colored circles.
Right: 3 bootstrap samples are built one at a time.
Each bootstrap sample draws 9 circles with replacement from the
original sample: a copy of a randomly picked circle flies to its
slot in the row, keeping the original untouched.

Render with (from the images/ folder):
    manim -qm anim_bootstrap.py BootstrapResampling
Output video ends up in media/videos/anim_bootstrap/.
"""

import numpy as np
from manim import (
    Circle,
    Create,
    FadeIn,
    LaggedStart,
    Line,
    Scene,
    Text,
    TransformFromCopy,
    VGroup,
    config,
)

# One distinct color per circle, so repeats in the bootstrap rows are
# visible at a glance. First 3 are the project accents (steelblue,
# firebrick, mediumseagreen), the rest are picked to stay distinguishable.
PALETTE = [
    "#4682B4",  # steelblue
    "#B22222",  # firebrick
    "#3CB371",  # mediumseagreen
    "#DAA520",  # goldenrod
    "#9370DB",  # mediumpurple
    "#FF8C00",  # darkorange
    "#008080",  # teal
    "#A0522D",  # sienna
    "#708090",  # slategray
]

N_POINTS = 9  # size of the original sample
N_BOOT = 3    # number of bootstrap samples shown

config.background_color = "#FFFFFF"


def make_circle(color, radius=0.22):
    """Return a filled circle with a thin dark outline."""
    circle = Circle(radius=radius)
    circle.set_fill(color, opacity=1.0)
    circle.set_stroke("#333333", width=1.5)
    return circle


class BootstrapResampling(Scene):
    def construct(self):
        rng = np.random.default_rng(seed=7)

        # ── Original sample: 9 scattered circles on the left ─────────
        # One distinct color per circle.
        colors = list(PALETTE)

        # Hand-placed scatter positions (x, y), roughly like the image.
        scatter_xy = [
            (-5.9,  1.9), (-4.9,  2.3), (-3.9,  1.8),
            (-5.5,  0.9), (-4.3,  1.0), (-3.6,  0.2),
            (-5.9, -0.1), (-4.9, -0.4), (-4.1, -1.0),
        ]

        originals = VGroup()
        for (x, y), color in zip(scatter_xy, colors):
            circle = make_circle(color)
            circle.move_to([x, y, 0])
            originals.add(circle)

        label_orig = Text("original sample", color="#333333", font_size=26)
        label_orig.next_to(originals, direction=[0, -1, 0], buff=0.6)

        divider = Line([-2.7, 2.8, 0], [-2.7, -2.2, 0], color="#888888",
                       stroke_width=2)

        self.play(FadeIn(originals, lag_ratio=0.1), run_time=1.5)
        self.play(FadeIn(label_orig), Create(divider), run_time=0.8)
        self.wait(0.5)

        # ── Bootstrap samples: 3 rows of 9 draws with replacement ────
        row_ys = [1.8, 0.4, -1.0]
        slot_xs = np.linspace(-2.0, 2.8, N_POINTS)

        for k in range(N_BOOT):
            row_label = Text(f"bootstrap sample {k + 1}",
                             color="#333333", font_size=24)
            row_label.move_to([4.8, row_ys[k], 0])
            self.play(FadeIn(row_label), run_time=0.4)

            # Draw 9 indices with replacement from the original sample.
            draws = rng.integers(0, N_POINTS, size=N_POINTS)

            animations = []
            for i, idx in enumerate(draws):
                source = originals[idx]
                copy_target = make_circle(source.get_fill_color())
                copy_target.move_to([slot_xs[i], row_ys[k], 0])

                number = Text(str(i + 1), color="#FFFFFF", font_size=18,
                              weight="BOLD")
                number.move_to(copy_target.get_center())

                animations.append(
                    LaggedStart(
                        TransformFromCopy(source, copy_target),
                        FadeIn(number),
                        lag_ratio=0.7,
                    )
                )

            self.play(LaggedStart(*animations, lag_ratio=0.25), run_time=3.0)
            self.wait(0.4)

        self.wait(2.0)
