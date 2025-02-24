import json
import random
import os
import math

BASE_DIR = "synthetic_data"
os.makedirs(f"{BASE_DIR}/vision", exist_ok=True)
os.makedirs(f"{BASE_DIR}/llm", exist_ok=True)
os.makedirs(f"{BASE_DIR}/labels", exist_ok=True)

num_samples = 500

# Fixed categories
TERMINAL_POSITIONS = {f"terminal_{i}": (random.randint(0, 100), random.randint(0, 100)) for i in range(10)}
WIRE_STATES = ["on_table", "held", "inserted"]
TERMINAL_STATES = ["empty", "inserted", "locked"]
WIRE_COLORS = ["red", "blue", "green", "yellow", "black"]

def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def generate_vision_file(num_wires=5):
    vision_data = {"wires": [], "terminals": {}}
    for i in range(num_wires):
        wire_color = random.choice(WIRE_COLORS)
        wire_name = f"{wire_color}_wire"
        state = "on_table"
        coords = (random.randint(0, 100), random.randint(0, 100))
        vision_data["wires"].append({
            "name": wire_name,
            "color": wire_color,
            "state": state,
            "coordinates": coords
        })
    for terminal, coords in TERMINAL_POSITIONS.items():
        vision_data["terminals"][terminal] = {
            "state": "empty",
            "coordinates": coords
        }
    return vision_data
    