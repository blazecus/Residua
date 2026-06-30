#!/usr/bin/env python3
# generated this tool to test some animations
# will probably manually improve later, it's not great but at least does produce working json files 

"""
Residua Animation Tool
Run from the project root: python tools/anim_tool.py

Controls:
  Space         play / pause
  Left / Right  step one frame
  K             add keyframe at current frame
  Delete        remove selected keyframe
  Ctrl+S        save animation
  Ctrl+O        open animation
  Ctrl+C        open character definition
  Right-click timeline   add keyframe at clicked frame
"""

import json
import math
import os
import sys
from collections import deque
from pathlib import Path
from tkinter import Tk, filedialog

import pygame

# ── Layout constants ───────────────────────────────────────────────────────────
VIEW_W      = 800
VIEW_H      = 520
PANEL_W     = 300
TIMELINE_H  = 160
WIN_W       = VIEW_W + PANEL_W
WIN_H       = VIEW_H + TIMELINE_H
SCALE       = 4          # world units → screen pixels
ANIM_FPS    = 60

# ── Palette ────────────────────────────────────────────────────────────────────
C_BG        = (28,  28,  34 )
C_PANEL     = (38,  38,  48 )
C_TIMELINE  = (22,  22,  28 )
C_TRACK     = (33,  33,  42 )
C_ACCENT    = (80,  160, 255)
C_TEXT      = (220, 220, 220)
C_DIM       = (110, 110, 125)
C_GRID      = (44,  44,  56 )
C_KF        = (255, 200, 55 )
C_KF_SEL    = (255, 110, 30 )
C_PLAYHEAD  = (255, 70,  70 )
C_BTN       = (52,  52,  66 )
C_BTN_BD    = (75,  75,  92 )
C_FLOOR     = (55,  55,  70 )

# ── Data model ─────────────────────────────────────────────────────────────────

class LimbDef:
    __slots__ = ("id", "sprite", "parent")
    def __init__(self, d):
        self.id     = d["id"]
        self.sprite = d["sprite"]
        self.parent = d.get("parent")

class JointDef:
    __slots__ = ("id", "parent_limb", "child_limb",
                 "attach_parent", "attach_child", "animatable", "default")
    def __init__(self, d):
        self.id             = d["id"]
        self.parent_limb    = d["parent_limb"]
        self.child_limb     = d["child_limb"]
        self.attach_parent  = tuple(d.get("attach_parent", [0., 0.]))
        self.attach_child   = tuple(d.get("attach_child",  [0., 0.]))
        self.animatable     = d.get("animatable", True)
        self.default        = float(d.get("default", 0.))

class CharacterDef:
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.name       = raw.get("name", Path(path).stem)
        self.path       = str(path)
        self.limbs      = {d["id"]: LimbDef(d)  for d in raw["limbs"]}
        self.joints     = {d["id"]: JointDef(d) for d in raw["joints"]}
        self.draw_order = raw.get("draw_order", list(self.limbs.keys()))
        # child_limb → joint
        self.joint_for_child = {jd.child_limb: jd for jd in self.joints.values()}

class Keyframe:
    def __init__(self, frame: int, angles: dict):
        self.frame  = frame
        self.angles = dict(angles)

class Animation:
    def __init__(self, char: CharacterDef):
        self.char      = char
        self.keyframes = []   # sorted by frame
        self.length    = 120  # total frames

    # -- interpolation --------------------------------------------------------

    def get_pose(self, frame: float) -> dict:
        """Interpolated pose (only animatable joints). Falls back to defaults."""
        anim_joints = {jid: jd for jid, jd in self.char.joints.items() if jd.animatable}
        defaults    = {jid: jd.default for jid, jd in anim_joints.items()}

        kfs = self.keyframes
        if not kfs:
            return defaults
        if frame <= kfs[0].frame:
            return {**defaults, **kfs[0].angles}
        if frame >= kfs[-1].frame:
            return {**defaults, **kfs[-1].angles}

        for i in range(len(kfs) - 1):
            a, b = kfs[i], kfs[i + 1]
            if a.frame <= frame <= b.frame:
                span = b.frame - a.frame
                t    = (frame - a.frame) / span if span else 0.
                out  = dict(defaults)
                for jid in anim_joints:
                    v0 = a.angles.get(jid, defaults[jid])
                    v1 = b.angles.get(jid, defaults[jid])
                    out[jid] = v0 + (v1 - v0) * t
                return out
        return defaults

    def full_pose(self, frame: float) -> dict:
        """Pose for ALL joints (animatable + fixed defaults)."""
        pose = {jid: jd.default for jid, jd in self.char.joints.items()}
        pose.update(self.get_pose(frame))
        return pose

    # -- editing --------------------------------------------------------------

    def upsert_keyframe(self, frame: int, angles: dict):
        for kf in self.keyframes:
            if kf.frame == frame:
                kf.angles = dict(angles)
                return
        self.keyframes.append(Keyframe(frame, angles))
        self.keyframes.sort(key=lambda k: k.frame)

    def remove_keyframe(self, frame: int):
        self.keyframes = [kf for kf in self.keyframes if kf.frame != frame]

    def get_keyframe(self, frame: int):
        for kf in self.keyframes:
            if kf.frame == frame:
                return kf
        return None

    # -- serialisation --------------------------------------------------------

    def to_dict(self):
        return {
            "fps":       ANIM_FPS,
            "length":    self.length,
            "keyframes": [{"frame": kf.frame, "joints": kf.angles}
                          for kf in self.keyframes],
        }

    @classmethod
    def from_dict(cls, data: dict, char: CharacterDef):
        anim = cls(char)
        anim.length = data.get("length", 120)
        for kfd in data.get("keyframes", []):
            anim.upsert_keyframe(kfd["frame"], kfd["joints"])
        return anim

# ── Geometry helpers ───────────────────────────────────────────────────────────

def rot2(v, a):
    c, s = math.cos(a), math.sin(a)
    return (v[0]*c - v[1]*s, v[0]*s + v[1]*c)

def compute_transforms(char: CharacterDef, pose: dict, origin=(0., 0.)):
    """
    Returns {limb_id: (world_x, world_y, world_angle)} for every limb.
    pose maps joint_id -> rest_angle  (= parent_angle - child_angle).
    """
    transforms = {}
    root = next(lid for lid, ld in char.limbs.items() if ld.parent is None)
    transforms[root] = (origin[0], origin[1], 0.)

    queue = deque([root])
    while queue:
        pid = queue.popleft()
        px, py, pa = transforms[pid]
        for lid, ld in char.limbs.items():
            if ld.parent != pid:
                continue
            jd   = char.joint_for_child.get(lid)
            rest = pose.get(jd.id, jd.default) if jd else 0.
            ca   = pa - rest

            if jd:
                aw = rot2(jd.attach_parent, pa)
                wp = (px + aw[0], py + aw[1])
                bw = rot2(jd.attach_child, ca)
                cx, cy = wp[0] - bw[0], wp[1] - bw[1]
            else:
                cx, cy = px, py

            transforms[lid] = (cx, cy, ca)
            queue.append(lid)

    return transforms

# ── Renderer ──────────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, char: CharacterDef, root: str, scale: int = SCALE):
        self.char       = char
        self.scale      = scale
        self.images     = {}
        self.com_locals = {}  # lid -> (cx, cy) in world units (same as sprite pixels)
        for lid, ld in char.limbs.items():
            path = os.path.join(root, ld.sprite)
            if os.path.exists(path):
                raw = pygame.image.load(path).convert_alpha()
                self.com_locals[lid] = self._calc_com(raw)
                w   = max(1, raw.get_width()  * scale)
                h   = max(1, raw.get_height() * scale)
                self.images[lid] = pygame.transform.scale(raw, (w, h))
            else:
                surf = pygame.Surface((8 * scale, 20 * scale), pygame.SRCALPHA)
                surf.fill((180, 60, 60, 160))
                self.images[lid] = surf

    @staticmethod
    def _calc_com(surf) -> tuple:
        """centroid of alpha pixels minus geometric center, in pixels (= world units)."""
        import numpy as np
        w, h  = surf.get_width(), surf.get_height()
        alpha = pygame.surfarray.array_alpha(surf).astype(np.float32)  # (w, h)
        total = alpha.sum()
        if total < 1.:
            return (0., 0.)
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        cx = float((alpha * xs[:, None]).sum()) / total
        cy = float((alpha * ys[None, :]).sum()) / total
        return (cx - w / 2., cy - h / 2.)

    def draw(self, surface, transforms, screen_origin):
        ox, oy = screen_origin
        for lid in self.char.draw_order:
            if lid not in transforms:
                continue
            wx, wy, wa = transforms[lid]
            img = self.images.get(lid)
            if img is None:
                continue
            # rb.position is the CoM; offset to place sprite's geometric center correctly
            cl    = self.com_locals.get(lid, (0., 0.))
            cl_r  = rot2(cl, wa)
            sx    = int((wx - cl_r[0]) * self.scale + ox)
            sy    = int((wy - cl_r[1]) * self.scale + oy)
            # Physics: CCW positive in math = CW on screen (Y-down) → negate for pygame
            deg     = -math.degrees(wa)
            rotated = pygame.transform.rotate(img, deg)
            rect    = rotated.get_rect(center=(sx, sy))
            surface.blit(rotated, rect)

# ── Simple UI widgets ─────────────────────────────────────────────────────────

class Slider:
    H = 8

    def __init__(self, x, y, w, lo, hi, value, label):
        self.rect  = pygame.Rect(x, y, w, self.H)
        self.lo    = lo
        self.hi    = hi
        self.value = float(value)
        self.label = label
        self._drag = False

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hit = pygame.Rect(self.rect.x - 4, self.rect.y - 14,
                              self.rect.w + 8, self.rect.h + 20)
            if hit.collidepoint(event.pos):
                self._drag = True
                self._set(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag = False
        elif event.type == pygame.MOUSEMOTION and self._drag:
            self._set(event.pos[0])
            return True
        return False

    def _set(self, mx):
        t = (mx - self.rect.x) / max(1, self.rect.w)
        self.value = self.lo + max(0., min(1., t)) * (self.hi - self.lo)

    def draw(self, surface, font, active: bool):
        col_fill  = C_ACCENT if active else (48, 88, 140)
        col_text  = C_TEXT   if active else C_DIM
        pygame.draw.rect(surface, C_TRACK, self.rect, border_radius=3)
        t  = (self.value - self.lo) / max(1e-6, self.hi - self.lo)
        fw = int(t * self.rect.w)
        pygame.draw.rect(surface, col_fill,
                         pygame.Rect(self.rect.x, self.rect.y, fw, self.rect.h),
                         border_radius=3)
        txt = font.render(f"{self.label}: {self.value:+.3f}", True, col_text)
        surface.blit(txt, (self.rect.x, self.rect.y - 14))

def draw_button(surface, font, rect, label, hover=False):
    col = (65, 65, 82) if hover else C_BTN
    pygame.draw.rect(surface, col,    rect, border_radius=4)
    pygame.draw.rect(surface, C_BTN_BD, rect, 1, border_radius=4)
    t = font.render(label, True, C_TEXT)
    surface.blit(t, t.get_rect(center=rect.center))

# ── Main application ──────────────────────────────────────────────────────────

class AnimTool:
    def __init__(self, project_root: str = "."):
        self.root = project_root
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Residua Animation Tool")
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont("monospace", 12)
        self.fontb  = pygame.font.SysFont("monospace", 12, bold=True)
        self.fonts  = pygame.font.SysFont("monospace", 10)

        self.char     : CharacterDef | None = None
        self.renderer : Renderer     | None = None
        self.anim     : Animation    | None = None
        self.anim_path: str          | None = None

        self.frame   = 0.
        self.playing = False
        self.speed   = 1.

        self.sel_kf_frame : int  | None = None
        self.edit_pose    : dict = {}

        self.sliders   : dict[str, Slider] = {}
        self.tl_drag   = False
        self.view_pan  = [VIEW_W // 2, int(VIEW_H * 0.45)]  # screen origin for world (0,0)

        # Load default character
        default = os.path.join(project_root, "assets", "animations", "player.json")
        if os.path.exists(default):
            self._load_char(default)

    # ── Character / animation IO ──────────────────────────────────────────────

    def _load_char(self, path):
        self.char     = CharacterDef(path)
        self.renderer = Renderer(self.char, self.root, SCALE)
        self.anim     = Animation(self.char)
        self._rebuild_sliders()
        self.sel_kf_frame = None
        self.edit_pose    = {}
        self.frame        = 0.

    def _rebuild_sliders(self):
        self.sliders.clear()
        if not self.char:
            return
        x, y, w = VIEW_W + 10, 32, PANEL_W - 22
        for jid, jd in self.char.joints.items():
            if not jd.animatable:
                continue
            self.sliders[jid] = Slider(x, y, w, -2.5, 2.5, jd.default, jid)
            y += 30

    def _open_char_dialog(self):
        root = Tk(); root.withdraw()
        p = filedialog.askopenfilename(title="Open Character",
                                       filetypes=[("JSON", "*.json")])
        root.destroy()
        if p:
            self._load_char(p)

    def _open_anim_dialog(self):
        if not self.char:
            return
        root = Tk(); root.withdraw()
        p = filedialog.askopenfilename(title="Open Animation",
                                       filetypes=[("JSON", "*.json")])
        root.destroy()
        if p:
            with open(p) as f:
                data = json.load(f)
            self.anim      = Animation.from_dict(data, self.char)
            self.anim_path = p
            self._on_frame_change()

    def _save_anim_dialog(self):
        if not self.anim:
            return
        root = Tk(); root.withdraw()
        p = filedialog.asksaveasfilename(title="Save Animation",
                                         defaultextension=".json",
                                         filetypes=[("JSON", "*.json")])
        root.destroy()
        if p:
            with open(p, "w") as f:
                json.dump(self.anim.to_dict(), f, indent=2)
            self.anim_path = p

    # ── Keyframe helpers ──────────────────────────────────────────────────────

    def _current_anim_pose(self):
        """Pose of animatable joints only, from sliders or interpolation."""
        if self.sel_kf_frame is not None:
            return dict(self.edit_pose)
        return self.anim.get_pose(self.frame)

    def _sync_sliders(self, pose: dict):
        for jid, sl in self.sliders.items():
            if jid in pose:
                sl.value = pose[jid]

    def _add_kf_here(self):
        f    = int(self.frame)
        pose = self._current_anim_pose()
        self.anim.upsert_keyframe(f, pose)
        self.sel_kf_frame = f
        self.edit_pose    = dict(pose)
        self._sync_sliders(pose)

    def _on_frame_change(self):
        f  = int(self.frame)
        kf = self.anim.get_keyframe(f)
        if kf:
            self.sel_kf_frame = f
            self.edit_pose    = dict(kf.angles)
        else:
            self.sel_kf_frame = None
            self.edit_pose    = {}
        self._sync_sliders(self._current_anim_pose())

    # ── Timeline geometry ─────────────────────────────────────────────────────

    @property
    def _tl_track(self):
        return pygame.Rect(8, VIEW_H + 50, VIEW_W - 16, TIMELINE_H - 58)

    def _fx(self, frame):
        r = self._tl_track
        return r.x + int(frame / max(1, self.anim.length) * r.w)

    def _xf(self, x):
        r = self._tl_track
        t = (x - r.x) / max(1, r.w)
        return int(max(0., min(1., t)) * self.anim.length)

    # ── Event handling ────────────────────────────────────────────────────────

    def _handle_buttons(self, pos):
        ty = VIEW_H + 8
        if pygame.Rect(10,  ty, 80, 24).collidepoint(pos):
            self.playing = not self.playing
        if pygame.Rect(100, ty, 90, 24).collidepoint(pos):
            self._add_kf_here()

        by = VIEW_H + TIMELINE_H - 28
        if pygame.Rect(VIEW_W + 8,   by, 88,  22).collidepoint(pos):
            self._open_char_dialog()
        if pygame.Rect(VIEW_W + 102, by, 88,  22).collidepoint(pos):
            self._open_anim_dialog()
        if pygame.Rect(VIEW_W + 196, by, 88,  22).collidepoint(pos):
            self._save_anim_dialog()

    def run(self):
        while True:
            dt = self.clock.tick(60) / 1000.

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return

                if event.type == pygame.KEYDOWN:
                    self._on_key(event)

                # Sliders are always interactive — auto-create a keyframe on first drag.
                if self.anim:
                    for jid, sl in self.sliders.items():
                        if sl.handle(event):
                            if self.sel_kf_frame is None:
                                self._add_kf_here()
                            self.edit_pose[jid] = sl.value
                            kf = self.anim.get_keyframe(self.sel_kf_frame)
                            if kf:
                                kf.angles[jid] = sl.value

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self._tl_track.collidepoint(event.pos):
                            self.tl_drag = True
                            self.frame = float(self._xf(event.pos[0]))
                            self._on_frame_change()
                        else:
                            self._handle_buttons(event.pos)
                    elif event.button == 3 and self._tl_track.collidepoint(event.pos):
                        self.frame = float(self._xf(event.pos[0]))
                        self._add_kf_here()

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.tl_drag = False

                elif event.type == pygame.MOUSEMOTION and self.tl_drag:
                    self.frame = float(self._xf(event.pos[0]))
                    self._on_frame_change()

            if self.playing and self.anim:
                self.frame += dt * ANIM_FPS * self.speed
                if self.frame >= self.anim.length:
                    self.frame = 0.

            self._draw()

    def _on_key(self, event):
        mods = pygame.key.get_mods()
        if event.key == pygame.K_SPACE:
            self.playing = not self.playing
        elif event.key == pygame.K_LEFT:
            self.frame = max(0., self.frame - 1.)
            self._on_frame_change()
        elif event.key == pygame.K_RIGHT and self.anim:
            self.frame = min(float(self.anim.length), self.frame + 1.)
            self._on_frame_change()
        elif event.key == pygame.K_k:
            self._add_kf_here()
        elif event.key == pygame.K_DELETE and self.sel_kf_frame is not None:
            self.anim.remove_keyframe(self.sel_kf_frame)
            self.sel_kf_frame = None
        elif event.key == pygame.K_s and (mods & pygame.KMOD_CTRL):
            self._save_anim_dialog()
        elif event.key == pygame.K_o and (mods & pygame.KMOD_CTRL):
            self._open_anim_dialog()
        elif event.key == pygame.K_c and (mods & pygame.KMOD_CTRL):
            self._open_char_dialog()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self):
        self.screen.fill(C_BG)
        self._draw_viewport()
        self._draw_panel()
        self._draw_timeline()
        pygame.display.flip()

    def _draw_viewport(self):
        vp = pygame.Surface((VIEW_W, VIEW_H))
        vp.fill(C_BG)

        # Grid
        ox, oy = self.view_pan
        step = SCALE * 10
        for gx in range(ox % step, VIEW_W, step):
            pygame.draw.line(vp, C_GRID, (gx, 0), (gx, VIEW_H))
        for gy in range(oy % step, VIEW_H, step):
            pygame.draw.line(vp, C_GRID, (0, gy), (VIEW_W, gy))

        # Axes
        pygame.draw.line(vp, (55, 55, 75), (ox, 0), (ox, VIEW_H))
        pygame.draw.line(vp, (55, 55, 75), (0, oy), (VIEW_W, oy))

        # Floor line (approximate ground)
        floor_y = oy + int(40 * SCALE)  # ~40 world units below torso origin
        pygame.draw.line(vp, C_FLOOR, (0, floor_y), (VIEW_W, floor_y), 2)

        # Character
        if self.char and self.renderer and self.anim:
            pose = self.anim.full_pose(self.frame)
            xf   = compute_transforms(self.char, pose)
            self.renderer.draw(vp, xf, self.view_pan)

        self.screen.blit(vp, (0, 0))

        # Overlay: frame / state
        label = f"{'▶' if self.playing else '⏸'}  frame {int(self.frame):03d}"
        if self.sel_kf_frame is not None:
            label += f"  [KF {self.sel_kf_frame}]"
        t = self.fontb.render(label, True, C_KF if self.sel_kf_frame is not None else C_TEXT)
        self.screen.blit(t, (8, 6))

        if self.char:
            cn = self.fontb.render(self.char.name, True, C_DIM)
            self.screen.blit(cn, (VIEW_W - cn.get_width() - 8, 6))

    def _draw_panel(self):
        pygame.draw.rect(self.screen, C_PANEL, pygame.Rect(VIEW_W, 0, PANEL_W, VIEW_H))

        if not self.char:
            msg = self.font.render("No character loaded  (Ctrl+C)", True, C_DIM)
            self.screen.blit(msg, (VIEW_W + 10, 10))
            return

        on_kf   = self.sel_kf_frame is not None
        hdr_col = C_KF_SEL if on_kf else C_DIM
        hdr     = f"{'KF #' + str(self.sel_kf_frame) if on_kf else 'drag a slider to create KF'}"
        self.screen.blit(self.fontb.render(hdr, True, hdr_col), (VIEW_W + 10, 8))

        for sl in self.sliders.values():
            sl.draw(self.screen, self.fonts, active=True)

        # Help
        help_y = VIEW_H - 100
        for line in ["Space play/pause", "K add keyframe",
                     "Del remove keyframe", "← → step frame",
                     "Right-click TL add KF"]:
            self.screen.blit(self.fonts.render(line, True, C_DIM),
                             (VIEW_W + 10, help_y))
            help_y += 13

    def _draw_timeline(self):
        ty = VIEW_H
        pygame.draw.rect(self.screen, C_TIMELINE,
                         pygame.Rect(0, ty, WIN_W, TIMELINE_H))

        # Control buttons
        mp = pygame.mouse.get_pos()
        play_r = pygame.Rect(10, ty + 8, 80, 24)
        kf_r   = pygame.Rect(100, ty + 8, 90, 24)
        draw_button(self.screen, self.font, play_r,
                    "⏸ Pause" if self.playing else "▶ Play",
                    play_r.collidepoint(mp))
        draw_button(self.screen, self.font, kf_r,
                    "+ Keyframe", kf_r.collidepoint(mp))

        if not self.anim:
            return

        # Track background
        tr = self._tl_track
        pygame.draw.rect(self.screen, C_TRACK, tr, border_radius=3)

        # Frame ruler
        total   = self.anim.length
        density = max(1, total // 24)
        for f in range(0, total + 1, density):
            x = self._fx(f)
            pygame.draw.line(self.screen, (65, 65, 82), (x, tr.y), (x, tr.y + 7))
            self.screen.blit(
                self.fonts.render(str(f), True, C_DIM),
                (x - 8, tr.y + 8))

        # Keyframe diamonds
        for kf in self.anim.keyframes:
            x   = self._fx(kf.frame)
            col = C_KF_SEL if kf.frame == self.sel_kf_frame else C_KF
            cy  = tr.y + tr.h // 2
            pts = [(x, cy - 7), (x + 5, cy), (x, cy + 7), (x - 5, cy)]
            pygame.draw.polygon(self.screen, col, pts)
            pygame.draw.aalines(self.screen, (0, 0, 0), True, pts)

        # Playhead
        px = self._fx(self.frame)
        pygame.draw.line(self.screen, C_PLAYHEAD, (px, tr.y - 4), (px, tr.bottom + 2), 2)
        pygame.draw.polygon(self.screen, C_PLAYHEAD,
                            [(px, tr.y - 4), (px - 5, tr.y - 12), (px + 5, tr.y - 12)])

        # Panel buttons
        by = ty + TIMELINE_H - 28
        for rect, label in [
            (pygame.Rect(VIEW_W + 8,   by, 88, 22), "Open Char"),
            (pygame.Rect(VIEW_W + 102, by, 88, 22), "Open Anim"),
            (pygame.Rect(VIEW_W + 196, by, 88, 22), "Save Anim"),
        ]:
            draw_button(self.screen, self.fonts, rect, label, rect.collidepoint(mp))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Resolve project root (one level up from tools/)
    script_dir   = Path(__file__).parent
    project_root = str(script_dir.parent)
    os.chdir(project_root)
    AnimTool(project_root).run()
