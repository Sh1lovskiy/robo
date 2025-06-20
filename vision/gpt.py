# vision/gpt.py
"""Minimal wrapper for DeepSeek and Anthropic APIs."""

import requests
import json
from typing import List, Tuple

from utils.config import Config

Config.load()
DEEPSEEK_API_KEY = Config.get("gpt.deepseek_api_key", "")
ANTHROPIC_API_KEY = Config.get("gpt.anthropic_api_key", "")
DEEPSEEK_URL = Config.get(
    "gpt.deepseek_url", "https://api.deepseek.com/v1/chat/completions"
)
ANTHROPIC_URL = Config.get(
    "gpt.anthropic_url", "https://api.anthropic.com/v1/messages"
)


def build_prompt(
    center: Tuple[float, float, float], width: float, height: float, length: float
) -> str:
    """Builds a unified prompt to generate 6D robot cleaning trajectory."""
    return f"""
Generate a list of robot trajectory points for cleaning all surfaces of a rectangular cuboid.

Inputs:
- A cuboid is defined by its center coordinates (x, y, z) in meters: {center}
- Dimensions: width (x-axis): {width}, height (z-axis): {height}, and length (y-axis): {length} in meters.

Requirements:
1. The robot should start from the bottom front-left corner.
2. Use a systematic back-and-forth pattern to clean all **vertical surfaces first**, then the **top surface**.
3. Each point in the output must be a 6D tuple (x, y, z, Rx, Ry, Rz) where:
   - (x, y, z) are coordinates in meters
   - (Rx, Ry, Rz) are Euler angles in degrees
4. For vertical walls, the orientation should ensure the robot "looks" toward the surface (i.e., face left/right/front/back depending on side).
5. For the top surface, the orientation should be straight downward (Rx = 180°, Ry = 0°, Rz = 0° or similar).
6. Return the result as a Python list of 6D tuples:
   [(x1, y1, z1, Rx1, Ry1, Rz1), (x2, y2, z2, Rx2, Ry2, Rz2), ...]
7. Output ONLY the Python list. Do NOT add any text or explanation.
"""


def generate_robot_path_deepseek(
    center: Tuple[float, float, float], width: float, height: float, length: float
) -> List[Tuple[float, float, float, float, float, float]]:
    """Generate 6D robot cleaning path using DeepSeek API."""
    prompt = build_prompt(center, width, height, length)

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    response = requests.post(DEEPSEEK_URL, headers=headers, json=payload)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return parse_coordinates(content)


def generate_robot_path_claude(
    center: Tuple[float, float, float], width: float, height: float, length: float
) -> List[Tuple[float, float, float, float, float, float]]:
    """Generate 6D robot cleaning path using Claude API."""
    prompt = build_prompt(center, width, height, length)

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    response = requests.post(ANTHROPIC_URL, headers=headers, json=payload)
    response.raise_for_status()
    content = response.json()["content"][0]["text"]
    return parse_coordinates(content)


def parse_coordinates(
    content: str,
) -> List[Tuple[float, float, float, float, float, float]]:
    """
    Parse the AI response to extract 6D coordinates list.
    Expected format: [(x, y, z, Rx, Ry, Rz), ...]
    """
    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        coord_str = content[start:end]
        return eval(coord_str, {"__builtins__": {}})
    except Exception as e:
        raise ValueError(f"Failed to parse coordinates: {e}\nRaw content:\n{content}")


def save_path_to_file(
    coordinates: List[Tuple[float, float, float, float, float, float]], filename: str
):
    """Save list of 6D points to a JSON file."""
    with open(filename, "w") as f:
        json.dump(coordinates, f, indent=2)


if __name__ == "__main__":
    center = (1.5, 2.0, 0.5)
    width = 2.0
    height = 1.0
    length = 3.0

    try:
        print("Generating path with DeepSeek...")
        path = generate_robot_path_deepseek(center, width, height, length)
        save_path_to_file(path, "robot_path_deepseek.json")
        print(f"Generated {len(path)} 6D points. Saved to robot_path_deepseek.json")

        # For Claude
        # print("Generating path with Claude...")
        # path = generate_robot_path_claude(center, width, height, length)
        # save_path_to_file(path, "robot_path_claude.json")
        # print(f"Generated {len(path)} 6D points. Saved to robot_path_claude.json")

    except Exception as e:
        print(f"Error: {e}")
