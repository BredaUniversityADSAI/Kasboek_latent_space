# Poem Generator System

Analyzes eye-tracking data and generates personalized poems using AI.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get your USER_KEY:
   - Go to https://ai-assistants.buas.nl/
   - Login with BUAS credentials
   - Click user icon (top-right)
   - Copy your USER_KEY

3. Edit `poem_generator_2.py`:
   - Replace `"ADD_YOUR_USER_KEY_HERE"` on line 8 with your key

4. Make sure `saccades.csv` exists (from eye-tracking system)

## Usage
```bash
python poem_generator_2.py
```

## Output

- `generated_poem.txt` - The generated poem
- Console shows analysis and poem generation process

## Files

- `poem_generator_2.py` - Main program
- `analyze_eye_tracking_2.py` - Eye-tracking analysis
- `requirements.txt` - Python dependencies
- `Rorschach_Interpretations_English.xlsx` - Interpretation database