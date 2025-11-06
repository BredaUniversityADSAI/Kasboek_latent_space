"""
Export data for display systems and integrations
"""

import os
import json
from collections import Counter


class DataExporter:
    """Export poem and gaze data for other systems"""

    def __init__(self, output_dir='shared'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_poem_data(self, poem: str, pattern: str, metrics: dict,
                         categories: list, timestamp: float, gaze_history: list):
        """Export data to JSON for display systems"""
        try:
            data = {
                'poem': poem,
                'pattern': pattern,
                'timestamp': timestamp,
                'metrics': metrics,
                'categories': categories,
                'gaze_summary': self._summarize_gaze(gaze_history)
            }

            filepath = f"{self.output_dir}/current_poem.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            return filepath
        except Exception:
            return None

    def _summarize_gaze(self, gaze_history: list) -> dict:
        """Create gaze data summary"""
        valid_data = [g for g in gaze_history if g is not None]

        if not valid_data:
            return {}

        directions = [g.get('direction', 'center') for g in valid_data]
        speeds = [g.get('saccade_speed', 0) for g in valid_data]

        direction_counts = Counter(directions)

        return {
            'total_frames': len(valid_data),
            'dominant_direction': direction_counts.most_common(1)[0][0],
            'avg_speed': sum(speeds) / len(speeds),
            'direction_distribution': dict(direction_counts)
        }