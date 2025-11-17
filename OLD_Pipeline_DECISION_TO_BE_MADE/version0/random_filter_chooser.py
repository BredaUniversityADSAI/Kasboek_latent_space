import cv2
import numpy as np
from typing import List, Dict
import time
import random

# Configuration
FILTER_CHOOSE_TIME = 5.0  # Seconds between filter changes
FILTERS_AT_ONCE = 4  # Number of filters to apply at once

class FilterChooser:
    """
    Random filter chooser that picks filters randomly.
    """
    
    def __init__(self, model_name=None):
        """
        Initialize the filter chooser.
        
        Args:
            model_name: Unused, kept for compatibility
        """
        self.last_analysis_time = 0
        self.analysis_interval = FILTER_CHOOSE_TIME
        self.current_filters = []
        self.first_run = True  # Flag to force immediate first selection
        print(f"✓ Random filter chooser initialized!")
        print(f"  - {FILTERS_AT_ONCE} filters at once")
        print(f"  - Changes every {FILTER_CHOOSE_TIME} seconds")
    
    def choose_random_filters(self, available_filters: Dict[int, str]) -> List[int]:
        """
        Randomly select filters.
        
        Args:
            available_filters: Dictionary of filter_num -> filter_name
            
        Returns:
            List of randomly selected filter numbers
        """
        # Get all available filter numbers
        filter_numbers = list(available_filters.keys())
        
        # Randomly select FILTERS_AT_ONCE filters
        num_to_select = min(FILTERS_AT_ONCE, len(filter_numbers))
        selected = random.sample(filter_numbers, num_to_select)
        
        # Show what was selected
        filter_names_list = [available_filters[f] for f in selected]
        print(f"🎲 Randomly selected: {', '.join(filter_names_list)}")
        
        return selected
    
    def should_analyze(self) -> bool:
        """
        Check if enough time has passed to select new filters.
        
        Returns:
            True if should select new filters, False otherwise
        """
        # Always select on first run
        if self.first_run:
            self.first_run = False
            self.last_analysis_time = time.time()
            return True
        
        current_time = time.time()
        if current_time - self.last_analysis_time >= self.analysis_interval:
            self.last_analysis_time = current_time
            return True
        return False
    
    def get_filters_for_frame(self, frame: np.ndarray, available_filters: Dict[int, str]) -> List[int]:
        """
        Main method to get random filter selections.
        
        Args:
            frame: Input video frame (unused, kept for compatibility)
            available_filters: Dictionary of available filters
            
        Returns:
            List of filter numbers to apply
        """
        if self.should_analyze():
            print(f"\n⏱️  Selecting new filters (next change in {FILTER_CHOOSE_TIME}s)...")
            self.current_filters = self.choose_random_filters(available_filters)
        
        return self.current_filters
    
    def reset_timer(self):
        """Reset the timer to force immediate selection on next call"""
        self.first_run = True
    
    def cleanup(self):
        """Cleanup resources"""
        pass