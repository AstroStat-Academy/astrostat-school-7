"""Helper functions for the NN Explainability notebook.

Currently holds `draw_circuit`, the generic circuit-diagram renderer used in the
mechanistic-interpretability sections (MNIST and galaxy morphology). The function
is self-contained: everything it needs is passed in as arguments.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox


def draw_circuit(stages, logits_vec, p, class_names, img,
                  input_display=None, top_k=None, figsize=(10, 10.5)):
    """Generic circuit diagram: gradient x activation scoring, top-k members
    per stage kept as the circuit, drawn with their own activation map as a
    thumbnail (non-spatial stages get a plain colored dot instead).

    stages:  list of (name, activation_tensor) pairs, in forward order,
             EXCLUDING the final logits -- e.g. [("conv1", a1), ("fc1", a3)].
    logits_vec: the full (num_classes,) logits tensor these stages feed into.
    p:       the predicted/target class index (row highlighted in green).
    class_names: list of class names.
    img:     the raw input tensor, only used for the top-left thumbnail.
    input_display: (array, cmap) for the input thumbnail; defaults to
             img.squeeze() in grayscale.
    top_k:   list of ints, one per stage (top |score| units to keep as
             circuit members); defaults to min(6, stage_size) per stage.
    """
    logit_p = logits_vec[p]                              # the scalar we explain

    # contribution score per channel/unit: gradient x activation -- exact
    # for the LAST stage (logit linear in it), first-order estimate earlier;
    # conv maps are summed over spatial positions -> one score per CHANNEL
    grads = torch.autograd.grad(logit_p, [t for _, t in stages])
    layer_scores = []
    for (name, t), g in zip(stages, grads):
        s = (g * t).squeeze(0)                           # contribution density
        layer_scores.append((s.sum((-2, -1)) if s.dim() == 3 else s).detach().cpu())

    # the circuit: top-k units by |score| in each stage
    if top_k is None:
        top_k = [min(6, len(s)) for s in layer_scores]
    tops = [s.abs().topk(k).indices.tolist() for s, k in zip(layer_scores, top_k)]

    # activation maps for the thumbnails (None for non-spatial stages)
    fmaps = [t[0].detach().cpu().numpy() if t.dim() == 4 else None
             for _, t in stages]

    # output column: all classes, or the top-10 logits if there are many
    if len(class_names) <= 10:
        out_ids = list(range(len(class_names)))
    else:
        out_ids = logits_vec.topk(10).indices.tolist()
    p_row = out_ids.index(p)                             # where to highlight

    # layout: columns auto-spaced over a fixed total width, however many stages
    n_stages = len(stages)
    cols = np.linspace(1.7, 5.7, n_stages + 1).tolist()  # x of the node columns

    # data-unit -> inches scale (this Axes is not aspect-equal), so a "square"
    # box on screen needs different widths and heights in data units
    x_range = (cols[-1] + 1.05) - 0.05                # xlim span
    y_range = 1.10 - (-0.09)                          # ylim span
    names = [f"{name}\n{len(s)} units" for (name, _), s in zip(stages, layer_scores)]
    names.append("logits")
    sizes = [len(s) for s in layer_scores]
    ys_out = np.linspace(0.28, 0.72, len(out_ids))

    # thumbnail height in DATA units: the column has a fixed budget (y1 - y0,
    # see column_positions below) to fit every circuit member's thumbnail in
    # the most crowded spatial stage. Shrink TH if needed so they never
    # overlap, then GROW the figure height so the shrunk thumbnails still
    # come out at roughly the same size on screen -- otherwise a wide layer's
    # thumbnails would either overlap (fixed TH) or shrink to illegibility
    # (fixed figure height).
    MEMBER_BUDGET = 0.92                              # y1 - y0 in column_positions
    MIN_GAP = 0.004                                   # min clearance around EVERY slot, thumbnail or dot
    TH_DEFAULT = 0.085                                # thumbnail height, uncrowded case
    TARGET_THUMB_IN = 0.75                            # desired on-screen thumbnail height
    MAX_FIG_HEIGHT = 12.5                             # inches -- beyond this, a crowded
                                                       # stage shrinks its thumbnails/spacing
                                                       # instead of growing the canvas further

    def max_th_for_stage(n_total, n_members):
        """Largest TH that still leaves >= MIN_GAP between EVERY one of the
        n_total slots in this column (not just the thumbnail slots) -- a
        crowded column has many non-member dots too, and they need real
        breathing room or they render right up against a thumbnail's edge."""
        if n_members == 0 or n_total <= 1:
            return TH_DEFAULT
        budget_for_members = MEMBER_BUDGET - (n_total - 1) * MIN_GAP
        return max(0.02, budget_for_members / n_members)

    # natural TH: the default, or whatever avoids overlap in the most
    # crowded stage -- same as before.
    TH = min([TH_DEFAULT] + [max_th_for_stage(sizes[L], top_k[L])
                             for L in range(n_stages) if fmaps[L] is not None])
    fig_height = TARGET_THUMB_IN * y_range / TH

    # only kick in when that natural height would exceed the cap: a stage
    # with many more units (e.g. a wide conv layer) then gets a SMALLER TH
    # than "no overlap" strictly requires, trading a bit of extra crowding
    # for a bounded figure instead of an ever-taller one.
    if fig_height > MAX_FIG_HEIGHT:
        TH = TARGET_THUMB_IN * y_range / MAX_FIG_HEIGHT
        fig_height = MAX_FIG_HEIGHT
    else:
        fig_height = max(figsize[1], fig_height)
    figsize = (figsize[0], fig_height)

    x_scale, y_scale = figsize[0] / x_range, figsize[1] / y_range
    TW = TH * (y_scale / x_scale)                     # thumbnail width -> square on screen

    def column_positions(n, members, th, y0=0.04, y1=0.96):
        """Equal edge-to-edge gaps: dots are points, thumbnails occupy th."""
        heights = [th if i in set(members) else 0.0 for i in range(n)]
        gap = (y1 - y0 - sum(heights)) / (n - 1)
        ys, y = [], y0
        for hh in heights:
            ys.append(y + hh / 2)
            y += hh + gap
        return ys

    # dense (non-spatial, "flattened") stages get a slightly TALLER column
    # than the conv stages -- a bit more y0/y1 headroom, so units end up a
    # little more spread out rather than sharing the exact same span as the
    # thumbnail-bearing columns.
    pos = [dict(enumerate(column_positions(
               sizes[L], tops[L] if fmaps[L] is not None else [], TH,
               *((0.0, 1.0) if fmaps[L] is None else (0.04, 0.96)))))
           for L in range(n_stages)]

    C_POS = C_NEG = "#c0392b"                            # all attributions in red (sign not colored)
    C_PRED, C_OFF = "#2e8b57", "#9a9a9a"                 # prediction / rest

    if input_display is None:
        input_display = (img.squeeze(), "gray")
    im_disp, cmap_disp = input_display

    with plt.rc_context({"font.family": "serif"}):
        fig, ax = plt.subplots(figsize=figsize)

        # -- input image, square, thin border, vertically centered ---------
        # target a constant ON-SCREEN size (in inches): as fig_height grows
        # for a crowded circuit, in_h (in DATA units) shrinks to compensate,
        # so the input thumbnail does not balloon along with a taller canvas.
        IN_TARGET_IN = 1.76                               # on-screen height, uncrowded case
        in_h = min(0.20, IN_TARGET_IN * y_range / fig_height)
        in_w = in_h * (y_scale / x_scale)                 # -> visually square
        in_cx, in_cy = 0.55, 0.5                           # centered vertically
        in_ext = (in_cx - in_w / 2, in_cx + in_w / 2, in_cy - in_h / 2, in_cy + in_h / 2)
        ax.imshow(im_disp, cmap=cmap_disp, aspect="equal",
                  extent=in_ext, zorder=2)
        ax.add_patch(Rectangle((in_ext[0], in_ext[2]),
                               in_ext[1] - in_ext[0], in_ext[3] - in_ext[2],
                               fill=False, edgecolor="black", lw=0.5, zorder=3))
        ax.text(in_cx, in_ext[3] + 0.03, "input", ha="center", fontsize=10)

        # -- edges: a SUBSET of the dense wiring (faint) + the circuit (red) --
        # drawing every pair would be up to ~10^5 lines; a random sample of a
        # few hundred per layer conveys the same "dense mesh" impression
        rng = np.random.default_rng(0)

        def sample_pairs(n_i, n_j, n_max=400):
            ii = rng.integers(0, n_i, size=min(n_max, n_i * n_j))
            jj = rng.integers(0, n_j, size=len(ii))
            return zip(ii.tolist(), jj.tolist())

        # spatial (conv-like) stages form a contiguous prefix of `stages`.
        # A "wiring" edge only means something once we reach a genuinely
        # fully-connected layer -- a conv layer mixes ALL input channels
        # through a shared, spatially-local kernel, so there is no single
        # weight connecting one channel to another the way this mesh would
        # imply. ANY edge originating from a spatial stage is excluded here
        # (spatial -> spatial AND, for now, spatial -> dense too) and gets a
        # receptive-field-window treatment instead (drawn with the nodes,
        # below) that actually reflects what a conv stack does.
        is_spatial = [f is not None for f in fmaps]

        segs = [] if is_spatial[0] else [[(in_ext[1], in_cy), (cols[0], pos[0][i])] for i in range(sizes[0])]
        for L in range(n_stages - 1):
            if not is_spatial[L]:
                segs += [[(cols[L], pos[L][i]), (cols[L + 1], pos[L + 1][j])]
                         for i, j in sample_pairs(sizes[L], sizes[L + 1])]
        segs += [[(cols[n_stages - 1], pos[-1][i]), (cols[-1], ys_out[d])]
                 for i, d in sample_pairs(sizes[-1], len(out_ids))]
        ax.add_collection(LineCollection(segs, colors="gray", linewidths=0.4,
                                         alpha=0.12, zorder=1))

        red = [] if is_spatial[0] else [[(in_ext[1], in_cy), (cols[0], pos[0][i])] for i in tops[0]]
        for L in range(n_stages - 1):
            if not is_spatial[L]:
                red += [[(cols[L], pos[L][i]), (cols[L + 1], pos[L + 1][j])]
                        for i in tops[L] for j in tops[L + 1]]
        ax.add_collection(LineCollection(red, colors=C_POS, linewidths=1.0,
                                         alpha=0.5, zorder=2))

        # last stage -> predicted logit: EXACT contributions (color = sign),
        # same fixed width as the earlier membership edges for visual consistency
        for i in tops[-1]:
            s = layer_scores[-1][i]
            ax.plot([cols[n_stages - 1], cols[-1]],
                    [pos[-1][i], ys_out[p_row]],
                    c=C_POS if s > 0 else C_NEG,
                    alpha=0.5, lw=1.0, zorder=2)

        # -- nodes ----------------------------------------------------------
        for L in range(n_stages):
            rest = [i for i in range(sizes[L]) if i not in tops[L]]
            ax.scatter([cols[L]] * len(rest), [pos[L][i] for i in rest],
                       s=11, c=C_OFF, zorder=3)
            for i in tops[L]:
                if fmaps[L] is not None:               # spatial: activation map
                    cx, cy = cols[L], pos[L][i]
                    ext = (cx - TW / 2, cx + TW / 2, cy - TH / 2, cy + TH / 2)
                    ax.imshow(fmaps[L][i], cmap="magma", aspect="auto",
                              extent=ext, zorder=5)
                    ax.add_patch(Rectangle((ext[0], ext[2]), TW, TH, fill=False,
                                           edgecolor=C_POS if layer_scores[L][i] > 0 else C_NEG,
                                           lw=1.6, zorder=6))
                else:                                  # non-spatial: colored dot
                    ax.scatter(cols[L], pos[L][i], s=85,
                               c=C_POS if layer_scores[L][i] > 0 else C_NEG,
                               edgecolors="white", linewidths=0.7, zorder=4)

        # -- receptive-field connector: ONE schematic connector between each
        # pair of consecutive spatial stages, not one per member pair. Two
        # lines leave the RIGHT edge of a small growing window on the
        # BOTTOM-MOST member thumbnail in stage L and converge on a single
        # point in the TOP-LEFT portion of the bottom-most member thumbnail
        # in stage L+1 -- a schematic of "a patch here feeds a point there",
        # without the clutter of a separate connector for every circuit
        # member. The LAST spatial stage gets a window on every member but no
        # connector out of it (see the "flatten" beam drawn below instead).
        def window_frac(step):
            return min(0.9, 0.15 * (1.6 ** step))

        def member_ext(L, i):
            cx, cy = cols[L], pos[L][i]
            return (cx - TW / 2, cx + TW / 2, cy - TH / 2, cy + TH / 2)

        def draw_window(ext):
            x0, x1, y0, y1 = ext
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=C_POS, lw=1.3, zorder=7))

        def bottom_member(L):
            """The circuit member drawn lowest on screen in stage L."""
            lowest_i = tops[L][0]
            for i in tops[L]:
                if pos[L][i] < pos[L][lowest_i]:
                    lowest_i = i
            return lowest_i

        def connect_to_point(win_ext, dst_L):
            """Two lines from win_ext's right corners into ONE point in the
            top-left portion of stage dst_L's bottom-most member thumbnail."""
            wx0, wx1, wy0, wy1 = win_ext
            dx0, dx1, dy0, dy1 = member_ext(dst_L, bottom_member(dst_L))
            tx, ty = dx0 + 0.25 * (dx1 - dx0), dy1 - 0.25 * (dy1 - dy0)
            ax.plot([wx1, tx], [wy1, ty], c=C_POS, lw=1.0, alpha=0.6, zorder=6)
            ax.plot([wx1, tx], [wy0, ty], c=C_POS, lw=1.0, alpha=0.6, zorder=6)
            ax.scatter([tx], [ty], s=10, c=C_POS, zorder=7)

        spatial_Ls = [L for L in range(n_stages) if fmaps[L] is not None]
        last_spatial = spatial_Ls[-1] if spatial_Ls else None

        if spatial_Ls:
            in_win_w, in_win_h = window_frac(0) * in_w, window_frac(0) * in_h
            in_win_ext = (in_ext[1] - in_win_w, in_ext[1], in_ext[3] - in_win_h, in_ext[3])
            draw_window(in_win_ext)
            connect_to_point(in_win_ext, spatial_Ls[0])

        for step, L in enumerate(spatial_Ls, start=1):
            if L == last_spatial:
                # last conv stage: a FULL-frame window (the whole thumbnail,
                # not a corner sub-region) -- flatten reads the ENTIRE map,
                # not a local patch, so no connector leaves this stage; see
                # the "flatten" beam drawn below instead.
                for i in tops[L]:
                    draw_window(member_ext(L, i))
                continue
            next_L = spatial_Ls[spatial_Ls.index(L) + 1]
            x0, x1, y0, y1 = member_ext(L, bottom_member(L))
            win_w, win_h = window_frac(step) * TW, window_frac(step) * TH
            win_ext = (x1 - win_w, x1, y1 - win_h, y1)
            draw_window(win_ext)
            connect_to_point(win_ext, next_L)

        # -- "flatten" beam: the last conv stage's output is reshaped into ONE
        # long vector and fed through a genuinely dense layer -- every value
        # really does connect to every unit, so instead of drawing thousands
        # of individual (or a misleadingly sparse sample of) edges, one single
        # translucent beam communicates "fully, densely connected" honestly,
        # spanning the WHOLE column on both sides (not just circuit members --
        # flatten involves every channel, highlighted or not).
        if last_spatial is not None and last_spatial + 1 < n_stages:
            dense_L = last_spatial + 1
            beam_x0 = cols[last_spatial]           # starts at the grey (inactive) units
            beam_x1 = cols[dense_L]                 # x -- passes BEHIND the thumbnails too
            left_y0, left_y1 = 0.04, 0.96           # conv2's own column range
            right_y0, right_y1 = 0.0, 1.0           # the flat layer's taller range -- the
                                                     # beam's top/bottom corners meet it exactly
            beam = Polygon([(beam_x0, left_y1), (beam_x1, right_y1),
                            (beam_x1, right_y0), (beam_x0, left_y0)],
                           closed=True, facecolor=C_POS, edgecolor="none",
                           alpha=0.08, zorder=0)
            ax.add_patch(beam)

        # logit nodes; the predicted class bold & green
        ax.scatter([cols[-1]] * len(out_ids), ys_out, s=55, c=C_OFF, zorder=3)
        ax.scatter(cols[-1], ys_out[p_row], s=160, c=C_PRED,
                   edgecolors="white", linewidths=0.9, zorder=4)
        for row, d in enumerate(out_ids):
            ax.text(cols[-1] + 0.13, ys_out[row], str(class_names[d])[:16],
                    va="center",
                    fontsize=11 if d == p else 9,
                    color=C_PRED if d == p else "black",
                    fontweight="bold" if d == p else "normal")

        # -- titles & cosmetics ----------------------------------------------
        for x_, name in zip(cols, names):
            ax.text(x_, 1.02, name, ha="center", fontsize=10)

        ax.set_xlim(0.05, cols[-1] + 1.05)
        ax.set_ylim(-0.09, 1.10)
        ax.axis("off")
        ax.set_title(f"The circuit behind '{class_names[p]}'   ", fontsize=12)

        # settle the Axes' final position FIRST -- tight_layout() can still
        # move/resize `ax` after this point, and if the colorbar were pinned
        # (via set_axes_locator(None) + an absolute set_position below) to
        # `ax`'s PRE-tight_layout bbox, it would end up misaligned from the
        # legend by however much tight_layout() nudges `ax` -- an amount that
        # varies figure to figure (title length, class list length, ...),
        # which is exactly why this only looked shifted on some circuits
        # (e.g. galaxy morphology) and not others (e.g. MNIST).
        plt.tight_layout()

        # -- legend (attribution) + colorbar (activation map), bottom-right --
        legend_handles = [
            Line2D([0], [0], color=C_POS, lw=3, label="attribution"),
        ]
        legend = ax.legend(handles=legend_handles, loc="lower right",
                           bbox_to_anchor=(0.99, 0.02), fontsize=9, frameon=True,
                           facecolor="white", edgecolor="lightgray", borderpad=1.0)

        # match the colorbar's width/x-position to the legend's own (auto-fit)
        # rendered bbox, and sit it just above it -- both only knowable AFTER
        # the legend has actually been laid out, and now AFTER `ax` itself has
        # settled into its final, post-tight_layout position too
        fig.canvas.draw()
        leg_bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())

        cbar_ax = ax.inset_axes([leg_bbox.x0, leg_bbox.y1 + 0.015, leg_bbox.width, 0.02])
        sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("top")     # ticks + labels pointing upward
        cbar.ax.xaxis.set_label_position("top")
        cbar.set_label("activation", fontsize=8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["min", "max"])
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(0.8)

        # the bar's own axes rectangle already equals leg_bbox exactly, but
        # the "min"/"max" tick labels overhang past it on both sides, so the
        # bar+labels VISUALLY look wider/shifted than the legend. Shrink and
        # shift the bar so the bar+labels footprint matches leg_bbox exactly.
        fig.canvas.draw()
        own_bbox = cbar.ax.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        tight_bbox = cbar.ax.get_tightbbox(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        left_overhang = own_bbox.x0 - tight_bbox.x0
        right_overhang = tight_bbox.x1 - own_bbox.x1
        corrected = Bbox.from_bounds(leg_bbox.x0 + left_overhang, leg_bbox.y1 + 0.015,
                                     leg_bbox.width - left_overhang - right_overhang, 0.02)
        cbar.ax.set_axes_locator(None)   # detach inset_axes' dynamic locator,
                                          # which would otherwise reset the
                                          # position below on the next draw
        cbar.ax.set_position(corrected.transformed(ax.transAxes).transformed(fig.transFigure.inverted()))

        plt.show()

    return tops   # per-stage circuit membership (indices), for reuse (e.g. ablation)
