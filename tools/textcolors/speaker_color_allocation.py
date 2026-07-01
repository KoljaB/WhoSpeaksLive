"""Speaker color allocation for readable, distinguishable text on black.

Dependency-free.

The allocator:
1. Samples a dense deterministic sRGB candidate set.
2. Filters candidates by WCAG contrast against black.
3. Removes greyish, muddy, and near-white candidates.
4. Converts colors to OKLab.
5. Orders colors with greedy farthest-point sampling.

The base order is generated with greedy farthest-point sampling.
After generation, the first two generated colors are moved to display
positions 07 and 08, as a final deterministic ordering adjustment.
"""

from __future__ import annotations

import colorsys
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class _Candidate:
    """Internal candidate color representation."""

    hex_color: str
    rgb: tuple[int, int, int]
    oklab: tuple[float, float, float]
    chroma: float
    contrast: float
    hls_saturation: float
    distance_from_white: float


class SpeakerColorAllocator:
    """Allocate readable, perceptually distinct speaker colors on black.

    Args:
        max_colors:
            Number of unique colors to generate. Intended range is 1..16.
        min_contrast:
            Minimum WCAG contrast ratio against black. 7.0 is stricter than
            WCAG AA and works well for normal text.
        candidate_steps:
            Number of samples per RGB channel. 24 means 24^3 raw candidates.
        seed:
            Optional deterministic tie-break seed.
        allow_reuse:
            If False, requesting more than max_colors raises IndexError.
            If True, colors cycle after max_colors.

    Notes:
        get_color(index) is zero-based.
        Speaker 1 is index 0.
    """

    _MIN_OKLAB_CHROMA = 0.10
    _MIN_HLS_SATURATION = 0.45
    _MIN_OKLAB_DISTANCE_FROM_WHITE = 0.10
    _OKLAB_WHITE = (1.0, 0.0, 0.0)

    def __init__(
        self,
        max_colors: int = 16,
        min_contrast: float = 7.0,
        candidate_steps: int = 24,
        seed: int | None = None,
        allow_reuse: bool = False,
    ) -> None:
        if max_colors <= 0:
            raise ValueError("max_colors must be greater than zero.")

        if max_colors > 16:
            raise ValueError("max_colors must be 16 or lower.")

        if not 1.0 <= min_contrast <= 21.0:
            raise ValueError("min_contrast must be between 1.0 and 21.0.")

        if not 4 <= candidate_steps <= 64:
            raise ValueError(
                "candidate_steps must be between 4 and 64. "
                "Candidate count grows cubically."
            )

        self.max_colors = max_colors
        self.min_contrast = min_contrast
        self.candidate_steps = candidate_steps
        self.seed = seed
        self.allow_reuse = allow_reuse

        self._rng = random.Random(seed)
        self._tie_breakers: dict[str, float] = {}
        self._palette = self._build_palette()
        self._next_index = 0

    def next_color(self) -> str:
        """Return the next color in allocation order as '#rrggbb'."""

        if self._next_index >= self.max_colors and not self.allow_reuse:
            raise IndexError(
                f"No speaker colors left: requested color "
                f"{self._next_index + 1}, but max_colors is {self.max_colors}. "
                "Pass allow_reuse=True to cycle through the palette."
            )

        color = self._palette[self._next_index % self.max_colors]
        self._next_index += 1
        return color

    def get_color(self, index: int) -> str:
        """Return the deterministic color for a zero-based speaker index."""

        if index < 0:
            raise IndexError("Color index must be non-negative.")

        if index >= self.max_colors and not self.allow_reuse:
            raise IndexError(
                f"Color index {index} is outside the palette of "
                f"{self.max_colors} colors."
            )

        return self._palette[index % self.max_colors]

    def palette(self) -> list[str]:
        """Return the full palette in final display order."""

        return list(self._palette)

    def reset(self) -> None:
        """Reset next_color() so the next call returns the first color again."""

        self._next_index = 0

    @staticmethod
    def contrast_ratio(hex_color: str, background: str = "#000000") -> float:
        """Return the WCAG contrast ratio between two sRGB hex colors."""

        fg_rgb = SpeakerColorAllocator._hex_to_rgb(hex_color)
        bg_rgb = SpeakerColorAllocator._hex_to_rgb(background)

        fg_lum = SpeakerColorAllocator._relative_luminance(fg_rgb)
        bg_lum = SpeakerColorAllocator._relative_luminance(bg_rgb)

        lighter = max(fg_lum, bg_lum)
        darker = min(fg_lum, bg_lum)

        return (lighter + 0.05) / (darker + 0.05)

    @staticmethod
    def distance(hex_a: str, hex_b: str) -> float:
        """Return perceptual distance between two colors in OKLab.

        The returned value is Euclidean OKLab distance multiplied by 100.
        The scaling makes the numbers easier to read.
        """

        lab_a = SpeakerColorAllocator._rgb_to_oklab(
            SpeakerColorAllocator._hex_to_rgb(hex_a)
        )
        lab_b = SpeakerColorAllocator._rgb_to_oklab(
            SpeakerColorAllocator._hex_to_rgb(hex_b)
        )

        return SpeakerColorAllocator._oklab_distance(lab_a, lab_b)

    def _build_palette(self) -> list[str]:
        candidates = self._generate_candidates()

        if len(candidates) < self.max_colors:
            raise ValueError(
                f"Only {len(candidates)} candidate colors survived filtering, "
                f"but max_colors is {self.max_colors}. Try lowering "
                "min_contrast or increasing candidate_steps."
            )

        ordered = sorted(candidates, key=lambda item: item.hex_color)

        if self.seed is None:
            self._tie_breakers = {
                item.hex_color: index / len(ordered)
                for index, item in enumerate(ordered)
            }
        else:
            self._tie_breakers = {
                item.hex_color: self._rng.random()
                for item in ordered
            }

        first = max(
            candidates,
            key=lambda item: (
                self._first_color_score(item),
                self._tie_breakers[item.hex_color],
            ),
        )

        selected = [first]
        remaining = [
            item for item in candidates
            if item.hex_color != first.hex_color
        ]

        nearest_distances = [
            self._oklab_distance(first.oklab, item.oklab)
            for item in remaining
        ]

        while len(selected) < self.max_colors:
            best_index = max(
                range(len(remaining)),
                key=lambda index: (
                    nearest_distances[index],
                    0.02 * self._first_color_score(remaining[index]),
                    self._tie_breakers[remaining[index].hex_color],
                ),
            )

            chosen = remaining.pop(best_index)
            nearest_distances.pop(best_index)
            selected.append(chosen)

            for index, candidate in enumerate(remaining):
                new_distance = self._oklab_distance(chosen.oklab, candidate.oklab)
                if new_distance < nearest_distances[index]:
                    nearest_distances[index] = new_distance

        palette = [item.hex_color for item in selected]
        return self._move_colors_01_and_02_to_positions_07_and_08(palette)

    @staticmethod
    def _move_colors_01_and_02_to_positions_07_and_08(
        colors: list[str],
    ) -> list[str]:
        """Move original colors 01 and 02 to positions 07 and 08.

        Positions are one-based here because this mirrors the speaker labels
        shown in the demo. For palettes with fewer than 8 colors, the order
        is left unchanged because positions 07 and 08 do not both exist.
        """

        if len(colors) < 8:
            return colors

        colors_01_and_02 = colors[:2]
        remaining_colors = colors[2:]

        return (
            remaining_colors[:6]
            + colors_01_and_02
            + remaining_colors[6:]
        )

    def _generate_candidates(self) -> list[_Candidate]:
        values = sorted(
            {
                round(index * 255 / (self.candidate_steps - 1))
                for index in range(self.candidate_steps)
            }
        )

        candidates: list[_Candidate] = []
        seen_hex: set[str] = set()

        for red in values:
            for green in values:
                for blue in values:
                    rgb = (red, green, blue)
                    hex_color = self._rgb_to_hex(rgb)

                    if hex_color in seen_hex:
                        continue

                    seen_hex.add(hex_color)

                    contrast = self.contrast_ratio(hex_color)
                    if contrast < self.min_contrast:
                        continue

                    red_float = red / 255.0
                    green_float = green / 255.0
                    blue_float = blue / 255.0

                    _, _, hls_saturation = colorsys.rgb_to_hls(
                        red_float,
                        green_float,
                        blue_float,
                    )

                    if hls_saturation < self._MIN_HLS_SATURATION:
                        continue

                    lab = self._rgb_to_oklab(rgb)
                    chroma = math.hypot(lab[1], lab[2])

                    if chroma < self._MIN_OKLAB_CHROMA:
                        continue

                    distance_from_white = self._oklab_distance_raw(
                        lab,
                        self._OKLAB_WHITE,
                    )

                    if distance_from_white < self._MIN_OKLAB_DISTANCE_FROM_WHITE:
                        continue

                    candidates.append(
                        _Candidate(
                            hex_color=hex_color,
                            rgb=rgb,
                            oklab=lab,
                            chroma=chroma,
                            contrast=contrast,
                            hls_saturation=hls_saturation,
                            distance_from_white=distance_from_white,
                        )
                    )

        return candidates

    def _first_color_score(self, candidate: _Candidate) -> float:
        """Score the initial anchor color.

        Later colors are not chosen by this score. They are chosen by greedy
        farthest-point sampling.
        """

        contrast_bonus = min(
            max(candidate.contrast - self.min_contrast, 0.0) / 10.0,
            1.0,
        )

        lightness_penalty = abs(candidate.oklab[0] - 0.80) * 0.15

        return (
            candidate.chroma * 3.0
            + candidate.hls_saturation * 0.20
            + candidate.distance_from_white * 0.10
            + contrast_bonus * 0.10
            - lightness_penalty
        )

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        value = hex_color.strip()

        if value.startswith("#"):
            value = value[1:]

        if len(value) == 3:
            value = "".join(channel * 2 for channel in value)

        if len(value) != 6:
            raise ValueError(f"Expected '#rrggbb' or '#rgb', got {hex_color!r}.")

        try:
            red = int(value[0:2], 16)
            green = int(value[2:4], 16)
            blue = int(value[4:6], 16)
        except ValueError as exc:
            raise ValueError(f"Invalid hex color {hex_color!r}.") from exc

        return red, green, blue

    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        red, green, blue = rgb
        return f"#{red:02x}{green:02x}{blue:02x}"

    @staticmethod
    def _relative_luminance(rgb: tuple[int, int, int]) -> float:
        red, green, blue = (channel / 255.0 for channel in rgb)

        red_linear = SpeakerColorAllocator._srgb_channel_to_linear(red)
        green_linear = SpeakerColorAllocator._srgb_channel_to_linear(green)
        blue_linear = SpeakerColorAllocator._srgb_channel_to_linear(blue)

        return (
            0.2126 * red_linear
            + 0.7152 * green_linear
            + 0.0722 * blue_linear
        )

    @staticmethod
    def _srgb_channel_to_linear(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92

        return ((channel + 0.055) / 1.055) ** 2.4

    @staticmethod
    def _rgb_to_oklab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
        """Convert 8-bit sRGB to OKLab."""

        red, green, blue = (channel / 255.0 for channel in rgb)

        red_linear = SpeakerColorAllocator._srgb_channel_to_linear(red)
        green_linear = SpeakerColorAllocator._srgb_channel_to_linear(green)
        blue_linear = SpeakerColorAllocator._srgb_channel_to_linear(blue)

        l_value = (
            0.4122214708 * red_linear
            + 0.5363325363 * green_linear
            + 0.0514459929 * blue_linear
        )
        m_value = (
            0.2119034982 * red_linear
            + 0.6806995451 * green_linear
            + 0.1073969566 * blue_linear
        )
        s_value = (
            0.0883024619 * red_linear
            + 0.2817188376 * green_linear
            + 0.6299787005 * blue_linear
        )

        l_cube_root = SpeakerColorAllocator._signed_cube_root(l_value)
        m_cube_root = SpeakerColorAllocator._signed_cube_root(m_value)
        s_cube_root = SpeakerColorAllocator._signed_cube_root(s_value)

        lightness = (
            0.2104542553 * l_cube_root
            + 0.7936177850 * m_cube_root
            - 0.0040720468 * s_cube_root
        )
        a_axis = (
            1.9779984951 * l_cube_root
            - 2.4285922050 * m_cube_root
            + 0.4505937099 * s_cube_root
        )
        b_axis = (
            0.0259040371 * l_cube_root
            + 0.7827717662 * m_cube_root
            - 0.8086757660 * s_cube_root
        )

        return lightness, a_axis, b_axis

    @staticmethod
    def _signed_cube_root(value: float) -> float:
        return math.copysign(abs(value) ** (1.0 / 3.0), value)

    @staticmethod
    def _oklab_distance(
        lab_a: tuple[float, float, float],
        lab_b: tuple[float, float, float],
    ) -> float:
        return 100.0 * SpeakerColorAllocator._oklab_distance_raw(lab_a, lab_b)

    @staticmethod
    def _oklab_distance_raw(
        lab_a: tuple[float, float, float],
        lab_b: tuple[float, float, float],
    ) -> float:
        return math.sqrt(
            (lab_a[0] - lab_b[0]) ** 2
            + (lab_a[1] - lab_b[1]) ** 2
            + (lab_a[2] - lab_b[2]) ** 2
        )


def _minimum_pairwise_distance(colors: Iterable[str]) -> float:
    color_list = list(colors)

    if len(color_list) < 2:
        return float("inf")

    return min(
        SpeakerColorAllocator.distance(left, right)
        for left_index, left in enumerate(color_list)
        for right in color_list[left_index + 1 :]
    )


def _enable_windows_ansi() -> None:
    """Enable ANSI escape codes in classic Windows consoles when possible."""

    if os.name != "nt":
        return

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE

        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return

        enable_virtual_terminal_processing = 0x0004
        kernel32.SetConsoleMode(
            handle,
            mode.value | enable_virtual_terminal_processing,
        )
    except Exception:
        return


def _ansi_truecolor_text(text: str, hex_color: str) -> str:
    red, green, blue = SpeakerColorAllocator._hex_to_rgb(hex_color)

    return (
        f"\x1b[40m"
        f"\x1b[38;2;{red};{green};{blue}m"
        f"{text}"
        f"\x1b[0m"
    )


def _run_self_tests() -> None:
    allocator = SpeakerColorAllocator()
    colors = allocator.palette()

    assert len(colors) == allocator.max_colors
    assert len(set(colors)) == len(colors), "Palette contains duplicate colors."

    for color in colors:
        contrast = allocator.contrast_ratio(color)
        assert contrast >= allocator.min_contrast, (
            f"{color} contrast {contrast:.2f} is below "
            f"{allocator.min_contrast:.2f}"
        )

    second_allocator = SpeakerColorAllocator()
    assert colors == second_allocator.palette(), (
        "Default palette is not deterministic."
    )

    allocator.reset()
    assert allocator.next_color() == colors[0]
    assert allocator.next_color() == colors[1]

    minimum_distance = _minimum_pairwise_distance(colors)
    print(f"Minimum pairwise OKLab distance: {minimum_distance:.2f}")


def _demo() -> None:
    allocator = SpeakerColorAllocator()
    colors = allocator.palette()

    _enable_windows_ansi()

    print("\x1b[40m", end="")
    print("Speaker colors rendered with ANSI truecolor escape codes:\n")

    for index, color in enumerate(colors, start=1):
        contrast = allocator.contrast_ratio(color)

        line = (
            f"Speaker {index:02d}: {color}  "
            f"contrast={contrast:5.2f}  "
            f"████████  "
            f"The quick brown fox jumps over the lazy dog."
        )

        print(_ansi_truecolor_text(line, color))

    print("\x1b[0m", end="")


if __name__ == "__main__":
    _run_self_tests()
    print()
    _demo()