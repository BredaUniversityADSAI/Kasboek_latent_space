import pandas as pd
import random
import requests
import json

# Your BUAS API credentials
API_URL = "https://ai-assistants.buas.nl/aioda-api"
USER_KEY = "ADD_YOUR_USER_KEY_HERE"  # Get from https://ai-assistants.buas.nl/
ASSISTANT_KEY = "47555064-c8b2-43cd-b696-8e1279754657" 

# API endpoints
init_endpoint_url = f"{API_URL}/actions/init_assistant"
ask_endpoint_url = f"{API_URL}/actions/ask_assistant"
headers = {"Content-Type": "application/json"}

# Read the Rorschach file
def load_rorschach_data(filepath):
    df = pd.read_excel(filepath)
    return df

# Select a random interpretation
def get_random_interpretation(df):
    random_row = df.sample(n=1).iloc[0]
    category = random_row['Category ENG']
    interpretation = random_row['Interpretation ENG']
    return category, interpretation

# Initialize the assistant and get chat_key
def init_assistant():
    data = {
        "user_key": USER_KEY,
        "assistant_key": ASSISTANT_KEY
    }
    response = requests.post(init_endpoint_url, data=json.dumps(data), headers=headers)
    
    if response.status_code == 200:
        assistant = response.json()
        if assistant['status'] == 'failed':
            print(f"ERROR: {assistant['error']}")
            return None
        chat_key = assistant["chat_key"]
        return chat_key
    else:
        print(f"Error initializing assistant: {response.status_code}")
        return None

# Generate a poem using BUAS API
def generate_poem(chat_key, category, interpretation):
    # Construct the prompt
    message = f"""Category: {category}
Interpretation: {interpretation}

Write a 6-8 line poem."""
    
    # Send request to BUAS API
    data = {
        "user_key": USER_KEY,
        "chat_key": chat_key,
        "assistant_key": ASSISTANT_KEY,
        "message": message
    }
    
    response = requests.post(ask_endpoint_url, data=json.dumps(data), headers=headers, stream=True)
    response.raise_for_status()
    
    # Print the response as we receive it (streaming)
    print(f"\nASSISTANT:")
    full_response = ""
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            decoded_chunk = chunk.decode('utf-8')
            full_response += decoded_chunk
            print(decoded_chunk, end='', flush=True)
    
    print("\n")  # New line after streaming
    return full_response

def main():
    # Import the analysis functions
    from analyze_eye_tracking_2 import (
        analyze_eye_tracking_csv, 
        match_to_rorschach_category,
        select_interpretation_from_category
    )
    
    # Load Rorschach data
    df = load_rorschach_data('Rorschach_Interpretations_English.xlsx')
    
    # Analyze eye-tracking data
    print("Analyzing eye-tracking data...")
    csv_file = 'saccades.csv'  # This will come from Ben's system
    
    analysis = analyze_eye_tracking_csv(csv_file)
    
    if analysis:
        print(f"Average speed: {analysis['avg_speed']:.2f}")
        print(f"Direction changes: {analysis['direction_changes']}")
        print(f"Change rate: {analysis['change_rate']:.2%}")
        print()
        
        # Match to Rorschach category
        category = match_to_rorschach_category(analysis)
        print(f"Matched category: {category}")
        
        # Get specific interpretation from that category
        category, interpretation = select_interpretation_from_category(df, category)
    else:
        print("No eye-tracking data found, using random interpretation...")
        category, interpretation = get_random_interpretation(df)
    
    print(f"Category: {category}")
    print(f"Interpretation: {interpretation}")
    
    # Initialize assistant and get chat_key
    print("\nInitializing assistant...")
    chat_key = init_assistant()
    
    if not chat_key:
        print("Failed to initialize assistant")
        return None
    
    print(f"Chat initialized with key: {chat_key}")
    
    # Generate poem
    poem = generate_poem(chat_key, category, interpretation)
    
    # Save to file (for printing)
    with open('generated_poem.txt', 'w', encoding='utf-8') as f:
        f.write(poem)
    
    print("\n[Poem saved to 'generated_poem.txt']")
    
    return poem

if __name__ == "__main__":
    main()