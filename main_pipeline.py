"""
Main poem generation pipeline
"""

import asyncio
import cv2
import time
from gaze_collector import GazeCollector
from poem_orchestrator import PoemOrchestrator
from config import validate_config, COLLECTION_SETTINGS


async def run_continuous():
    """Run continuous poem generation"""

    if not validate_config():
        print("Configuration invalid. Check config.py")
        return

    collector = GazeCollector()
    orchestrator = PoemOrchestrator()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        for idx in range(1, 5):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                break

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("System running")

    poem_count = 0
    wait_time = COLLECTION_SETTINGS["wait_between_poems"]

    try:
        while True:
            poem_count += 1
            print(f"\n--- Poem #{poem_count} ---")

            print("Collecting gaze data...")
            gaze_history = collector.collect(cap)

            valid_count = sum(1 for g in gaze_history if g is not None)
            if valid_count < 30:
                print("Insufficient data, retrying...")
                await asyncio.sleep(2)
                continue

            print("Generating poem...")
            result = orchestrator.process(gaze_history)

            print(f"\nPattern: {result['pattern']}")
            print(f"\n{result['poem']}\n")

            await asyncio.sleep(wait_time)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        collector.release()


def run_single_test():
    """Test with single poem generation"""

    if not validate_config():
        print("Configuration invalid. Check config.py")
        return

    collector = GazeCollector()
    orchestrator = PoemOrchestrator()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Single test mode")
    print("Look at camera naturally for 15 seconds...")

    gaze_history = collector.collect(cap)

    valid_count = sum(1 for g in gaze_history if g is not None)
    if valid_count < 30:
        print(f"Insufficient data ({valid_count} frames)")
        cap.release()
        return

    print("Processing...")
    result = orchestrator.process(gaze_history)

    print(f"\nPattern: {result['pattern']}")
    print(f"\n{result['poem']}\n")

    cap.release()
    collector.release()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_single_test()
    else:
        asyncio.run(run_continuous())