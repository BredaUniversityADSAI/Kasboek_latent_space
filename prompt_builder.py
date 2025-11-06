"""
Build prompts for LLM poem generation
"""

from rorschach_interpreter import PATTERN_DESCRIPTIONS


class PromptBuilder:
    """Construct prompts for poem generation"""

    @staticmethod
    def build_poem_prompt(pattern: str, metrics: dict,
                          interpretations: list, tone: str) -> str:
        """Build comprehensive prompt for poem generation"""

        pattern_desc = PATTERN_DESCRIPTIONS.get(pattern, {})
        selected_interps = interpretations[:4]

        essence_translations = {
            "snel, rusteloze zoektocht": "rapid, restless seeking",
            "diepe contemplatie, naar binnen keren": "deep contemplation, inward turning",
            "systematische verkenning, orde zoeken": "systematic exploration, seeking order",
            "gefragmenteerde aandacht, elektrische beweging": "fragmented attention, electric movement",
            "zwevend bewustzijn, zacht dwalen": "floating awareness, gentle wandering",
            "voorzichtige observatie, terughoudend": "cautious observation, holding back",
            "anker zoeken, grond vinden": "seeking anchor, finding ground",
            "speelse ontdekking, onconventionele paden": "playful discovery, unconventional paths"
        }

        essence = essence_translations.get(
            pattern_desc.get('essence', ''),
            pattern_desc.get('essence', 'a unique pattern')
        )

        prompt = f"""Write a short, evocative poem in English (6-10 lines).

CONTEXT:
A person's gaze was observed for 15 seconds, revealing a pattern of attention.

GAZE PATTERN: "{pattern}"
- Essence: {essence}
- Movement speed: {metrics.get('avg_saccade_speed', 0):.3f}
- Center focus: {metrics.get('center_time_ratio', 0):.1%}
- Pattern complexity: {metrics.get('direction_entropy', 0):.2f}

SYMBOLIC IMAGES (from their gaze):
{chr(10).join(f'- {interp}' for interp in selected_interps)}

DESIRED TONE: {tone}

Create a poem that:
1. Captures this person's inner state as revealed by their gaze
2. Weaves the symbolic images into meaningful poetry
3. Uses metaphorical language
4. Feels personal yet universal
5. Reflects the tone: {tone}

Write ONLY the poem, no title, no explanation."""

        return prompt