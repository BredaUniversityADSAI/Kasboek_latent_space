"""
Print formatting and output module
"""

import os
from datetime import datetime


class PrintFormatter:
    """Format poems for printing"""

    def __init__(self, output_dir='print_queue'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def format_poem(self, poem: str, pattern: str, metrics: dict, timestamp: float) -> str:
        """Create formatted text for printing"""
        output = []
        output.append("=" * 60)
        output.append("INNER STATE - A Poetic Interpretation")
        output.append("=" * 60)
        output.append("")
        output.append(f"Date: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Pattern: {pattern.replace('_', ' ').title()}")
        output.append("")
        output.append("-" * 60)

        for line in poem.split('\n'):
            output.append(line)

        output.append("")
        output.append("-" * 60)
        output.append(f"Speed: {metrics.get('avg_saccade_speed', 0):.3f}")
        output.append(f"Focus: {metrics.get('center_time_ratio', 0):.1%}")
        output.append(f"Entropy: {metrics.get('direction_entropy', 0):.2f}")
        output.append("=" * 60)

        return '\n'.join(output)

    def save(self, poem: str, pattern: str, metrics: dict, timestamp: float) -> str:
        """Save formatted poem to file"""
        try:
            formatted = self.format_poem(poem, pattern, metrics, timestamp)
            filename = f"{self.output_dir}/poem_{int(timestamp)}.txt"

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted)

            return filename
        except Exception:
            return None

    def send_to_printer(self, filename: str):
        """Send file to printer if available"""
        if not filename or not os.path.exists(filename):
            return

        try:
            if os.system("which lp > /dev/null 2>&1") == 0:
                os.system(f"lp {filename}")
        except Exception:
            pass