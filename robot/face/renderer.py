"""SDL2 / pygame procedural vector renderer for the 800x480 robot face display."""

from __future__ import annotations

import math
from pathlib import Path

import pygame

from robot.face.animator import FaceFrameParams

BG_COLOR = (10, 14, 22)
BEZEL_COLOR = (22, 30, 46)


class PygameFaceRenderer:
    """Renders `FaceFrameParams` onto an 800x480 pygame Surface at 60 FPS."""

    def __init__(self, width: int = 800, height: int = 480, *, headless: bool = False) -> None:
        self.width = width
        self.height = height
        self.headless = headless
        self._surface: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None

    def init_surface(self, title: str = "Sprocket — Robot Face (800x480)") -> pygame.Surface:
        if not pygame.get_init():
            pygame.init()
        if self.headless:
            self._surface = pygame.Surface((self.width, self.height))
        else:
            self._surface = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(title)
        if pygame.font.get_init():
            self._font = pygame.font.SysFont("consolas", 16)
        return self._surface

    @property
    def surface(self) -> pygame.Surface:
        if self._surface is None:
            return self.init_surface()
        return self._surface

    def render(self, params: FaceFrameParams) -> pygame.Surface:
        """Draw a single frame of the procedural face and flip display if windowed."""
        surf = self.surface
        surf.fill(BG_COLOR)

        # Outer rounded visor frame
        visor_rect = pygame.Rect(24, 20, self.width - 48, self.height - 40)
        pygame.draw.rect(surf, BEZEL_COLOR, visor_rect, width=2, border_radius=36)

        # Eye centers and dimensions
        eye_w = 180
        eye_h = 190
        center_y = int(self.height * 0.44) + int(params.gaze_y * 28)
        gaze_shift_x = int(params.gaze_x * 42)
        left_cx = int(self.width * 0.31) + gaze_shift_x
        right_cx = int(self.width * 0.69) + gaze_shift_x

        self._draw_eye(
            surf,
            cx=left_cx,
            cy=center_y,
            base_w=eye_w,
            base_h=eye_h,
            openness=params.left_openness,
            brow_deg=params.left_brow_deg,
            squint=params.lower_lid_squint,
            pupil_scale=params.pupil_scale,
            color_rgb=params.color_rgb,
            is_left=True,
        )
        self._draw_eye(
            surf,
            cx=right_cx,
            cy=center_y,
            base_w=eye_w,
            base_h=eye_h,
            openness=params.right_openness,
            brow_deg=params.right_brow_deg,
            squint=params.lower_lid_squint,
            pupil_scale=params.pupil_scale,
            color_rgb=params.color_rgb,
            is_left=False,
        )

        # Mouth / expressive voice bar
        self._draw_mouth(surf, params, shift_x=int(gaze_shift_x * 0.4))

        # Status badge in bottom-left of visor
        if self._font is not None:
            badge_text = f"{params.activity.upper()} | {params.emotion} ({params.intensity:.1f}) | gaze:{params.gaze_target}"
            txt_surf = self._font.render(badge_text, True, (110, 135, 165))
            surf.blit(txt_surf, (48, self.height - 52))

        if not self.headless and pygame.display.get_surface() is not None:
            pygame.display.flip()

        return surf

    def save_snapshot(self, path: Path | str) -> Path:
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.surface, str(out))
        return out

    def _draw_eye(
        self,
        surf: pygame.Surface,
        *,
        cx: int,
        cy: int,
        base_w: int,
        base_h: int,
        openness: float,
        brow_deg: float,
        squint: float,
        pupil_scale: float,
        color_rgb: tuple[int, int, int],
        is_left: bool,
    ) -> None:
        w = int(base_w * max(0.75, min(1.25, pupil_scale)))
        h = max(10, int(base_h * max(0.05, min(1.30, openness))))
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)

        # Soft outer glow
        glow_color = (
            max(0, color_rgb[0] // 4),
            max(0, color_rgb[1] // 4),
            max(0, color_rgb[2] // 4),
        )
        glow_rect = rect.inflate(20, 20)
        pygame.draw.rect(surf, glow_color, glow_rect, border_radius=max(8, h // 2))

        # Core glowing eye shape
        pygame.draw.rect(surf, color_rgb, rect, border_radius=max(6, int(min(w, h) * 0.36)))

        # Specular highlight when eye is open
        if h > 36:
            hl_r = max(6, min(18, h // 9))
            hl_color = (
                min(255, color_rgb[0] + 90),
                min(255, color_rgb[1] + 90),
                min(255, color_rgb[2] + 90),
            )
            pygame.draw.circle(surf, hl_color, (cx - w // 5, cy - h // 5), hl_r)

        # Lower eyelid happy squint mask
        if squint > 0.02 and h > 24:
            squint_h = int(h * min(0.65, squint))
            mask_rect = pygame.Rect(rect.left - 8, rect.bottom - squint_h, w + 16, squint_h + 12)
            pygame.draw.ellipse(surf, BG_COLOR, mask_rect)

        # Upper eyebrow slope polygon mask
        if abs(brow_deg) > 1.0 and h > 20:
            slope = math.tan(math.radians(max(-30.0, min(30.0, brow_deg)))) * (w * 0.5)
            # Positive brow_deg slopes down toward the center (stern/curious), negative slopes up toward center (concerned)
            inner_drop = int(slope)
            outer_drop = int(-slope)
            left_drop = outer_drop if is_left else inner_drop
            right_drop = inner_drop if is_left else outer_drop
            top_y = rect.top - 10
            poly = [
                (rect.left - 6, top_y),
                (rect.right + 6, top_y),
                (rect.right + 6, rect.top + max(0, right_drop + 14)),
                (rect.left - 6, rect.top + max(0, left_drop + 14)),
            ]
            pygame.draw.polygon(surf, BG_COLOR, poly)

    def _draw_mouth(self, surf: pygame.Surface, params: FaceFrameParams, *, shift_x: int = 0) -> None:
        mx = self.width // 2 + shift_x
        my = int(self.height * 0.77)
        mw = 110
        mh = max(6, int(68 * params.mouth_open))
        color = params.color_rgb

        if params.mouth_open > 0.18:
            # Speaking or surprised rounded mouth
            mrect = pygame.Rect(mx - mw // 2, my - mh // 2, mw, mh)
            pygame.draw.rect(surf, color, mrect, border_radius=mh // 2)
        else:
            # Curved smile / neutral / frown line built from quadratic bezier samples
            curve_px = int(params.mouth_curve * 26)
            pts: list[tuple[int, int]] = []
            steps = 16
            for i in range(steps + 1):
                t = i / steps
                x = int((mx - mw // 2) + t * mw)
                # Parabola peaking at t=0.5
                y = int(my + 4.0 * t * (1.0 - t) * curve_px)
                pts.append((x, y))
            pygame.draw.lines(surf, color, False, pts, width=6)
