"""
MASTER ORCHESTRATOR - Runs the Complete Installation
Coordinates all components:
- Eye tracking + visualization
- Gaze analysis
- Poem generation
- TTS audio output
- Print formatting
- Display systems
"""

import asyncio
import cv2
import threading
import time
from poem_generation_pipeline import CompletePoemPipeline
from visual_display_system import EyeVisualizer
from config import validate_config


class CompleteSystemOrchestrator:
    """
    Master controller for the entire art installation
    """

    def __init__(self):
        print(" 'Through the Eye of the Algorithm'")

        if not validate_config():
            print("\n Configuration invalid")
            raise ValueError("Invalid configuration")

        # Initialize components
        print("\nInitializing components...")

        print("  1. Poem Generation Pipeline...")
        self.poem_pipeline = CompletePoemPipeline()

        print("  2. Visual Display System...")
        self.visualizer = EyeVisualizer()

        print("\n✓ ALL SYSTEMS INITIALIZED")
        print("=" * 70)

        self.running = True
        self.current_poem = None
        self.current_pattern = None

    def run_visual_displays(self, cap):
        """
        Thread function: Run visual displays continuously
        Shows cropped eyes and gaze trajectory
        """
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
                    frame, filter_type=filter_types[current_filter]
                )
                gaze_display = self.visualizer.create_gaze_display(gaze_info)

                if self.current_poem:
                    lines = self.current_poem.split('\n')
                    y_pos = gaze_display.shape[0] - 150

                    cv2.rectangle(gaze_display,
                                  (0, y_pos - 10),
                                  (gaze_display.shape[1], gaze_display.shape[0]),
                                  (0, 0, 0), -1)

                    for i, line in enumerate(lines[:3]):  # Show first 3 lines
                        if len(line) > 50:
                            line = line[:50] + "..."
                        cv2.putText(gaze_display, line,
                                    (10, y_pos + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)

                cv2.imshow('Eyes - Cropped with Filter', eye_display)
                cv2.imshow('Gaze Trajectory - Live', gaze_display)

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

            except Exception as e:
                print(f" Visual display error: {e}")
                time.sleep(0.1)

    async def run_poem_generation(self, cap):
        """
        Async function: Run poem generation loop
        """
        wait_between = 3.0
        poem_count = 0

        try:
            while self.running:
                poem_count += 1
                print(f" POEM GENERATION CYCLE #{poem_count}")
                print(f"{'=' * 70}")

                gaze_history = self.poem_pipeline.collect_gaze_data(cap)

                valid_count = sum(1 for g in gaze_history if g is not None)
                if valid_count < 30:
                    print(f"\n Insufficient data ({valid_count} frames), retrying...")
                    await asyncio.sleep(2)
                    continue

                result = self.poem_pipeline.process_and_generate(gaze_history)
                self.current_poem = result['poem']
                self.current_pattern = result['pattern']

                print(f"{'=' * 70}")
                print(f"  ✓ Poem: {len(result['poem'])} characters")
                print(f"  ✓ Pattern: {result['pattern']}")
                print(f"  ✓ Audio: Generated and played")
                print(f"  ✓ Print: Formatted and saved")
                print(f"  ✓ Display: Data exported")
                print(f"{'=' * 70}")
                print(f"\nWaiting {wait_between}s before next poem...")
                await asyncio.sleep(wait_between)

        except asyncio.CancelledError:
            print("\nPoem generation stopped")

    async def run(self):
        """
        Main execution: Run everything together
        """
        print("\nOpening webcam...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            for idx in range(1, 5):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    break

        if not cap.isOpened():
            print(" ERROR: Cannot open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✓ Webcam opened successfully")

        print("=" * 70)
        print("\nComponents Running:")
        print("  • Eye tracking with live visualization")
        print("  • Gaze trajectory display")
        print("  • Poem generation (15s cycles)")
        print("  • TTS audio output")
        print("  • Print queue")
        print("  • Display data export")
        print("\nPress 'q' in any window to stop")
        print("Press 1/2/3 to change eye filter")
        print("=" * 70 + "\n")

        try:
            # Start visual display in separate thread
            visual_thread = threading.Thread(
                target=self.run_visual_displays,
                args=(cap,),
                daemon=True
            )
            visual_thread.start()

            await self.run_poem_generation(cap)

        except KeyboardInterrupt:
            print("\n\nStopping system...")
            self.running = False

        finally:
            print("\nCleaning up...")
            self.running = False
            time.sleep(1)

            cap.release()
            cv2.destroyAllWindows()
            self.poem_pipeline.eye_tracker.release()
            self.visualizer.release()

            print("✓ System stopped cleanly\n")


def main():
    """Entry point"""
    try:
        orchestrator = CompleteSystemOrchestrator()
        asyncio.run(orchestrator.run())

    except ValueError as e:
        print(f"\n Initialization failed: {e}")
        print("fix configuration")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("=" * 70)
    print("\nThis script runs the complete system:")
    print("  ✓ Eye tracking + visualization")
    print("  ✓ Gaze pattern analysis")
    print("  ✓ Poem generation")
    print("  ✓ TTS audio output")
    print("  ✓ Print formatting")
    print("  ✓ Display systems")
    print("=" * 70 + "\n")

    input("Press Enter to start the complete system...")
    main()