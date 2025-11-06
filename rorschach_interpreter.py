"""
Rorschach Interpretation System - ENGLISH VERSION
Loads interpretations and classifies gaze patterns
"""

import pandas as pd
import random
import numpy as np
from typing import Dict, List, Tuple


class RorschachInterpreter:
    """Loads Rorschach interpretations and maps to gaze patterns"""

    CATEGORIES = {
        "architectural": ["Architectural", "Architectu"],
        "human": ["Human Fig", "Human"],
        "psychological": ["Psycholog"],
        "nature": ["Natuur"],
        "animals": ["Dieren"],
        "cosmic": ["Kosmische"]
    }

    # ENGLISH poem tones for variety
    POEM_TONES = [
        "mysterious and philosophical",
        "surreal and dreamlike",
        "intimate and personal",
        "abstract and cosmic",
        "emotional and vulnerable",
        "observational and precise",
        "haunting and evocative",
        "playful and curious"
    ]

    def __init__(self, excel_path: str = "Rorschach_Interpretations_English.xlsx"):
        self.excel_path = excel_path
        self.data = None
        self.load_data()

    def load_data(self):
        """Load and parse Excel file"""
        try:
            print(f"  Loading Rorschach data from: {self.excel_path}")

            xls = pd.ExcelFile(self.excel_path)
            df = pd.read_excel(self.excel_path, sheet_name=0)

            interpretations = []
            cols = df.columns.tolist()
            i = 0

            while i < len(cols) - 1:
                col_name = str(cols[i]).lower()

                if 'category' in col_name or 'categor' in col_name:
                    if i + 1 < len(cols):
                        category_col = cols[i]
                        interp_col = cols[i + 1]

                        for _, row in df.iterrows():
                            cat = row[category_col]
                            interp = row[interp_col]

                            if pd.notna(cat) and pd.notna(interp):
                                cat_str = str(cat).strip()
                                interp_str = str(interp).strip()

                                if len(cat_str) > 0 and len(interp_str) > 0:
                                    interpretations.append({
                                        'category': cat_str,
                                        'interpretation': interp_str
                                    })

                        i += 2
                        continue
                i += 1

            self.data = pd.DataFrame(interpretations)

            if len(self.data) == 0:
                # Fallback parsing
                for i in range(0, len(cols) - 1, 2):
                    for _, row in df.iterrows():
                        cat = row[cols[i]]
                        interp = row[cols[i + 1]]
                        if pd.notna(cat) and pd.notna(interp):
                            interpretations.append({
                                'category': str(cat).strip(),
                                'interpretation': str(interp).strip()
                            })

                self.data = pd.DataFrame(interpretations)

            print(f"  ✓ Loaded {len(self.data)} interpretations")

        except FileNotFoundError:
            print(f" File not found: {self.excel_path}")
            self.data = None
        except Exception as e:
            print(f"  Error loading: {e}")
            self.data = None

    def get_interpretation_by_category(self, category: str, n: int = 1) -> List[str]:
        """Get random interpretations from category"""
        if self.data is None or len(self.data) == 0:
            fallback = {
                'architectural': ["A cathedral shaped from mist", "An endless labyrinth"],
                'human': ["Two figures mirroring each other", "A dancing silhouette"],
                'psychological': ["A face transforming in light", "An emotion captured in abstraction"],
                'nature': ["A lightning flash in slow-motion", "Mountains that seem to breathe"],
                'animals': ["A butterfly in transformation", "A bird taking flight"],
                'cosmic': ["A meteor exploding", "A planet with strange rings"]
            }
            return fallback.get(category, ["A mysterious pattern"])[:n]

        category_patterns = self.CATEGORIES.get(category, [category])

        matches = self.data[
            self.data['category'].str.contains('|'.join(category_patterns), case=False, na=False)
        ]

        if len(matches) == 0:
            matches = self.data

        sample_size = min(n, len(matches))
        if sample_size > 0:
            selected = matches.sample(n=sample_size)
            return selected['interpretation'].tolist()
        return ["A mysterious pattern emerges"]

    def map_gaze_to_categories(self, gaze_pattern: str) -> List[str]:
        """Map gaze pattern to relevant categories"""
        GAZE_TO_CATEGORY_MAP = {
            "scattered_fast": ["psychological", "cosmic", "animals"],
            "focused_center": ["psychological", "architectural", "human"],
            "methodical_scan": ["architectural", "nature", "human"],
            "erratic_jumps": ["cosmic", "psychological", "animals"],
            "slow_drift": ["nature", "cosmic", "psychological"],
            "peripheral_avoidance": ["psychological", "architectural"],
            "rapid_return_center": ["human", "psychological", "architectural"],
            "diagonal_exploration": ["cosmic", "animals", "nature"],
        }
        return GAZE_TO_CATEGORY_MAP.get(gaze_pattern, ["psychological", "nature"])

    def get_random_tone(self) -> str:
        """Get random poem tone"""
        return random.choice(self.POEM_TONES)

    def generate_interpretation_set(self, gaze_pattern: str) -> Dict:
        """Generate interpretation set for gaze pattern"""
        categories = self.map_gaze_to_categories(gaze_pattern)

        interpretations = {}
        for cat in categories[:2]:
            interps = self.get_interpretation_by_category(cat, n=3)
            interpretations[cat] = interps

        psych_interps = self.get_interpretation_by_category('psychological', n=3)
        tone = self.get_random_tone()

        return {
            'gaze_pattern': gaze_pattern,
            'primary_categories': categories[:2],
            'interpretations': interpretations,
            'psychological_context': psych_interps,
            'all_interpretations': sum(interpretations.values(), []),
            'poem_tone': tone
        }


class GazePatternClassifier:
    """Classifies gaze patterns from 15 seconds of data"""

    def __init__(self):
        self.thresholds = {
            'saccade_speed_high': 0.25,
            'saccade_speed_low': 0.08,
            'center_bias_high': 0.65,
            'center_bias_low': 0.30,
            'direction_entropy_high': 2.5,
            'fixation_duration_long': 0.8,
        }

    def classify_gaze_pattern(self, gaze_history: List[Dict]) -> Tuple[str, Dict]:
        """Classify gaze pattern"""
        valid_data = [g for g in gaze_history if g is not None]

        if len(valid_data) < 10:
            return "insufficient_data", {}

        metrics = {
            'avg_saccade_speed': self._compute_avg_saccade_speed(valid_data),
            'center_time_ratio': self._compute_center_bias(valid_data),
            'direction_entropy': self._compute_direction_entropy(valid_data),
            'movement_variance': self._compute_movement_variance(valid_data),
            'fixation_count': self._count_fixations(valid_data),
            'total_frames': len(valid_data)
        }

        pattern = self._classify(metrics)
        return pattern, metrics

    def _compute_avg_saccade_speed(self, data: List[Dict]) -> float:
        speeds = [g.get('saccade_speed', 0) for g in data]
        return float(np.mean(speeds)) if speeds else 0.0

    def _compute_center_bias(self, data: List[Dict]) -> float:
        center_count = sum(1 for g in data if g.get('direction') == 'center')
        return center_count / len(data) if data else 0.0

    def _compute_direction_entropy(self, data: List[Dict]) -> float:
        from collections import Counter
        import math

        directions = [g.get('direction', 'center') for g in data]
        counts = Counter(directions)
        total = len(directions)

        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return float(entropy)

    def _compute_movement_variance(self, data: List[Dict]) -> float:
        positions = []
        for g in data:
            norm = g.get('smoothed_norm')
            if norm is not None:
                if isinstance(norm, (list, tuple, np.ndarray)):
                    positions.append(norm)

        if not positions:
            return 0.0

        positions = np.array(positions)
        return float(np.var(positions))

    def _count_fixations(self, data: List[Dict]) -> int:
        fixations = 0
        in_fixation = False

        for g in data:
            speed = g.get('saccade_speed', 0)
            if speed < 0.05 and not in_fixation:
                fixations += 1
                in_fixation = True
            elif speed > 0.1:
                in_fixation = False

        return fixations

    def _classify(self, metrics: Dict) -> str:
        """Main classification logic"""
        speed = metrics['avg_saccade_speed']
        center = metrics['center_time_ratio']
        entropy = metrics['direction_entropy']
        variance = metrics['movement_variance']

        if speed > self.thresholds['saccade_speed_high']:
            if entropy > self.thresholds['direction_entropy_high']:
                return "scattered_fast"
            else:
                return "erratic_jumps"

        elif center > self.thresholds['center_bias_high']:
            if speed < self.thresholds['saccade_speed_low']:
                return "focused_center"
            else:
                return "rapid_return_center"

        elif center < self.thresholds['center_bias_low']:
            return "peripheral_avoidance"

        elif speed < self.thresholds['saccade_speed_low']:
            if variance < 0.01:
                return "focused_center"
            else:
                return "slow_drift"

        elif entropy > self.thresholds['direction_entropy_high']:
            return "diagonal_exploration"

        else:
            return "methodical_scan"


PATTERN_DESCRIPTIONS = {
    "scattered_fast": {
        "essence": "rapid, restless seeking",
        "emotion": "anxious energy, urgency",
        "metaphor": "a hummingbird's flight, never still",
        "tempo": "allegro, staccato"
    },
    "focused_center": {
        "essence": "deep contemplation, inward turning",
        "emotion": "calm introspection, presence",
        "metaphor": "a still pool reflecting stars",
        "tempo": "largo, sustained"
    },
    "methodical_scan": {
        "essence": "systematic exploration, seeking order",
        "emotion": "curious precision, measured wonder",
        "metaphor": "a cartographer mapping unknown lands",
        "tempo": "moderato, even rhythm"
    },
    "erratic_jumps": {
        "essence": "fragmented attention, electric movement",
        "emotion": "excitement or overwhelm, intensity",
        "metaphor": "lightning seeking ground",
        "tempo": "presto, irregular"
    },
    "slow_drift": {
        "essence": "floating awareness, gentle wandering",
        "emotion": "dreamy detachment, reverie",
        "metaphor": "clouds drifting across twilight",
        "tempo": "adagio, flowing"
    },
    "peripheral_avoidance": {
        "essence": "cautious observation, holding back",
        "emotion": "guarded reserve, hesitation",
        "metaphor": "watching from the forest's edge",
        "tempo": "andante, tentative"
    },
    "rapid_return_center": {
        "essence": "seeking anchor, finding ground",
        "emotion": "need for stability, centering",
        "metaphor": "a compass needle finding north",
        "tempo": "allegretto, pulsing return"
    },
    "diagonal_exploration": {
        "essence": "playful discovery, unconventional paths",
        "emotion": "curious delight, openness",
        "metaphor": "a child finding secret passages",
        "tempo": "allegro, playful rhythm"
    }
}

if __name__ == "__main__":
    print("Testing Rorschach Interpreter...")
    interpreter = RorschachInterpreter()

    if interpreter.data is not None and len(interpreter.data) > 0:
        print(f"\nSystem ready! {len(interpreter.data)} interpretations loaded")
    else:
        print("\nData loading failed")