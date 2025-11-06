"""
Run complete integrated system
"""

import asyncio
import cv2
import threading
import time
from gaze_collector import GazeCollector
from poem_orchestrator import PoemOrchestrator
from eye_visualizer import EyeVisualizer
from config import validate_config, COLLECTION_SETTINGS


class SystemRunner:
    """Run all components together"""

    def __init__(self):
        if not validate_config():
            raise ValueError("Configuration invalid")

        self.collector = GazeCollector()
        self.orchestrator = PoemOrchestrator()
        self.visualizer = EyeVisualizer()
        self.running = True
        self.current_poem = None

    def run_visuals(self, cap):
        """Visual display thread"""
        filter_types = ["neon", "thermal", "ascii"]
        current_filter = 0

        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                if self.visualizer.eye_tracker.flip:
                    frame = cv2.flip(frame, 1)

                _, gaze_info = self.visualizer.eye_tracker.process_frame(
                    frame, only_compute=True
                )

                eye_display = self.visualizer.create_eye_display(
                    frame, filter_types[current_filter]
                )
                gaze_display = self.visualizer.create_gaze_display(gaze_info)

                # Add poem to display if available
                if self.current_poem:
                    lines = self.current_poem.split('\n')
                    y_pos = gaze_display.shape[0] - 150

                    cv2.rectangle(gaze_display,
                                  (0, y_pos - 10),
                                  (gaze_display.shape[1], gaze_display.shape[0]),
                                  (0, 0, 0), -1)

                    for i, line in enumerate(lines[:3]):
                        if len(line) > 50:
                            line = line[:50] + "..."
                        cv2.putText(gaze_display, line,
                                    (10, y_pos + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)

                cv2.imshow('Eyes', eye_display)
                cv2.imshow('Gaze', gaze_display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('1'):
                    current_filter = 0
                elif key == ord('2'):
                    current_filter = 1
                elif key == ord('3'):
                    current_filter = 2
                elif key == ord('q'):
                    self.running = False
                    break

            except Exception:
                time.sleep(0.1)

    async def run_poems(self, cap):
        """Poem generation async loop"""
        wait_time = COLLECTION_SETTINGS["wait_between_poems"]
        poem_count = 0

        try:
            while self.running:
                poem_count += 1
                print(f"\n--- Poem #{poem_count} ---")

                gaze_history = self.collector.collect(cap)

                valid_count = sum(1 for g in gaze_history if g is not None)
                if valid_count < 30:
                    await asyncio.sleep(2)
                    continue

                result = self.orchestrator.process(gaze_history)

                self.current_poem = result['poem']

                print(f"Pattern: {result['pattern']}")
                print(f"\n{result['poem']}\n")

                await asyncio.sleep(wait_time)

        except asyncio.CancelledError:
            pass

    async def run(self):
        """Run complete system"""
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

        print("Complete system running")
        print("Press 'q' in window to stop, 1/2/3 to change filter")

        try:
            visual_thread = threading.Thread(
                target=self.run_visuals,
                args=(cap,),
                daemon=True
            )
            visual_thread.start()

            await self.run_poems(cap)

        except KeyboardInterrupt:
            print("\nStopping...")
            self.running = False

        finally:
            self.running = False
            time.sleep(1)

            cap.release()
            cv2.destroyAllWindows()
            self.collector.release()
            self.visualizer.release()


def main():
    """Entry point"""
    try:
        runner = SystemRunner()
        asyncio.run(runner.run())
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()