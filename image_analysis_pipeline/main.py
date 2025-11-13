from chat import *
from analysis import run_classification
from tts import tts_setup, run_tts
from heatmap import draw_heatmap
import time
from datetime import datetime
import logging
import os

def run_analysis_pipeline():
    '''
    Function to run the pipeline: logger initialization, model initialization, classification, psychoanalysis, poem writing, \
text-to-speech, heatmap generation, and updating the website

    Params:
        None
    
    Returns:
        True/False: depends on whether the pipeline ran successfully
    '''


    # Logger initialization
    logging.basicConfig(filename='installation.log', level=logging.INFO, 
                       format='%(asctime)s;%(name)s;%(levelname)s;%(message)s')
    start_time = datetime.now()
    logging.info(f"Started script at {start_time}")

    try:
        # Logger initialization
        logging.basicConfig(filename='installation.log', level=logging.INFO, format='%(asctime)s;%(name)s;%(levelname)s;%(message)s')
        start_time = datetime.now()
        logging.info(f"Started script at {start_time}")

        # Model setup
        # dpa: (d)er (P)sycho(a)nalytiker
        # poesie: name of the poet, meaning 'poetry' in French
        logging.info("Setting up models...")
        ASSISTANT_KEY_DPA, USER_KEY, init_endpoint_url_dpa, ask_endpoint_url_dpa, headers = credentials('64a00530-4985-4289-bd2d-69dbbd2257d5')
        ASSISTANT_KEY_POESIE, USER_KEY, init_endpoint_url_poesie, ask_endpoint_url_poesie, headers = credentials('8c1cdd40-14e7-460b-acaf-0f874f50703e')
        response_dpa = initialize_ai_assistant(ASSISTANT_KEY_DPA, USER_KEY, init_endpoint_url_dpa, headers)
        response_poesie = initialize_ai_assistant(ASSISTANT_KEY_POESIE, USER_KEY, init_endpoint_url_poesie, headers)
        tts_client, voice, audio_config = tts_setup()
        logging.info("Setup complete!")

        # Run classification
        logging.info("Starting classification...")
        label = run_classification('image.png')
        logging.info("Classification compelete!")

        # Psychoanalysis
        logging.info("Starting psychoananalysis...")
        psychoanalysis = chat_with_ai_assistant(response=response_dpa, message=label, USER_KEY=USER_KEY, ask_endpoint_url=ask_endpoint_url_dpa, ASSISTANT_KEY=ASSISTANT_KEY_DPA, headers=headers)
        logging.info("Psychanalysis complete!")

        # Poem
        logging.info("Starting writing the poem...")
        poem = chat_with_ai_assistant(response=response_poesie, message=psychoanalysis, USER_KEY=USER_KEY, ask_endpoint_url=ask_endpoint_url_poesie, ASSISTANT_KEY=ASSISTANT_KEY_POESIE, headers=headers)
        logging.info("Poem complete!")

        # Text-to-speech
        logging.info("Starting text-to-speech...")
        poem_filename = run_tts(client=tts_client, voice=voice, audio_config=audio_config, text=poem, output="poem")
        logging.info("Text-to-speech complete!")

        # Generate heatmap
        logging.info("Generating heatmap...")
        heatmap_filename = draw_heatmap()
        logging.info("Heatmap complete!")

        # Generate content.json for website
        logging.info("Updating website...")
        website_data = {
            "scribblePath": "image.png",
            "heatmapPath": "heatmap.png",
            "psychoanalysis": psychoanalysis,
            "poem": poem,
            "poemAudio": poem_filename,
            "timestamp": datetime.now().isoformat()
        }
        
        # Write data to a .json file
        with open('content.json', 'w', encoding='utf-8') as f:
            json.dump(website_data, f, indent=2, ensure_ascii=False)
        
        logging.info("Website updated!")

        logging.info(f"Program concluded in {datetime.now() - start_time}.")
        return True
    
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        print(f"Error occurred: {str(e)}")
        return False
    
if __name__ == "__main__":
    image_path = 'image.png'
    last_modified_time = None
    
    print("Watching for changes in image.png... Press Ctrl+C to stop.")
    
    try:
        # Continuously check for changes in the scribble image
        # If there is a change in the image, a new pair of eyes has been detected, and rerun the analysis
        while True:
            if os.path.exists(image_path):
                current_modified_time = os.path.getmtime(image_path)
                
                if last_modified_time is None:
                    # First run
                    run_analysis_pipeline()
                    last_modified_time = current_modified_time

                elif current_modified_time != last_modified_time:
                    # File has been modified
                    print("\nDetected change in image.png, rerunning analysis...")
                    run_analysis_pipeline()
                    last_modified_time = current_modified_time

            # Check every second
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping file watch...")