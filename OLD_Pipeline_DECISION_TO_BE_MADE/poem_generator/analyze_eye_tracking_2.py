"""
Poem Generator - Setup Instructions

1. Get your USER_KEY from https://ai-assistants.buas.nl/ (click user icon)
2. Replace USER_KEY on line 8 with your key
3. Install: pip install pandas openpyxl requests
4. Make sure saccades.csv exists (from eye-tracking system)
5. Run: python poem_generator_2.py

Output: generated_poem.txt
"""

import pandas as pd
import numpy as np

def analyze_eye_tracking_csv(csv_filepath):
    """
    Analyzes eye-tracking CSV file and returns psychological characteristics.
    
    CSV columns: timestamp, speed, direction
    """
    # Read the CSV
    df = pd.read_csv(csv_filepath)
    
    if df.empty:
        return None
    
    # Calculate statistics
    avg_speed = df['speed'].mean()
    max_speed = df['speed'].max()
    speed_variance = df['speed'].var()
    
    # Count direction changes (compare each row to the previous one)
    # We need to handle the direction column as strings, not numbers
    direction_changes = 0
    if len(df) > 1:
        for i in range(1, len(df)):
            if df['direction'].iloc[i] != df['direction'].iloc[i-1]:
                direction_changes += 1
    
    total_samples = len(df)
    change_rate = direction_changes / total_samples if total_samples > 0 else 0
    
    # Determine movement pattern
    analysis = {
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'speed_variance': speed_variance,
        'direction_changes': direction_changes,
        'change_rate': change_rate,
        'total_samples': total_samples
    }
    
    return analysis


def match_to_rorschach_category(analysis):
    """
    Matches eye-tracking analysis to Rorschach categories.
    
    Returns: category name (string)
    """
    if analysis is None:
        return "Psychological and Emotional Interpretations"
    
    avg_speed = analysis['avg_speed']
    change_rate = analysis['change_rate']
    speed_variance = analysis['speed_variance']
    
    # Decision rules for matching to categories
    
    # High speed + high variance + many changes = Cosmic/Abstract (chaotic, scattered)
    if avg_speed > 50 and change_rate > 0.3 and speed_variance > 100:
        return "Cosmic and Abstract Patterns"
    
    # Low speed + low changes = Nature (calm, flowing)
    elif avg_speed < 20 and change_rate < 0.2:
        return "Nature and Elements"
    
    # Medium speed + structured patterns = Architectural
    elif 20 <= avg_speed <= 50 and change_rate < 0.25:
        return "Architectural and Structural Associations"
    
    # High changes + medium speed = Animals (dynamic, organic)
    elif change_rate > 0.25 and 30 <= avg_speed <= 60:
        return "Animals and Organic Forms"
    
    # Moderate patterns = Human Figures
    elif change_rate > 0.2 and avg_speed > 25:
        return "Human Figures and Physicality"
    
    # Default = Psychological/Emotional
    else:
        return "Psychological and Emotional Interpretations"


def select_interpretation_from_category(df_rorschach, category):
    """
    Selects a random interpretation from a specific category.
    
    df_rorschach: DataFrame with Rorschach data
    category: category name (string)
    
    Returns: (category, interpretation)
    """
    # Filter by category
    category_rows = df_rorschach[df_rorschach['Category ENG'] == category]
    
    if category_rows.empty:
        # Fallback to random if category not found
        random_row = df_rorschach.sample(n=1).iloc[0]
        return random_row['Category ENG'], random_row['Interpretation ENG']
    
    # Select random interpretation from this category
    selected_row = category_rows.sample(n=1).iloc[0]
    return selected_row['Category ENG'], selected_row['Interpretation ENG']


# Test function
if __name__ == "__main__":
    # Create fake test data
    test_data = pd.DataFrame({
        'timestamp': range(100),
        'speed': np.random.uniform(10, 80, 100),
        'direction': np.random.choice(['left', 'right', 'up', 'down', 'center'], 100)
    })
    
    # Save test CSV
    test_data.to_csv('test_saccades.csv', index=False)
    
    # Test the analysis
    analysis = analyze_eye_tracking_csv('test_saccades.csv')
    print("Eye-tracking analysis:")
    print(analysis)
    print()
    
    category = match_to_rorschach_category(analysis)
    print(f"Matched category: {category}")