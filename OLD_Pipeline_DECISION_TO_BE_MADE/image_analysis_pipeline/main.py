from chat import LLMModel
from analysis import run_classification
from tts_gc import gc_tts_setup, run_gc_tts
from tts_el import el_tts_setup, run_el_tts
from heatmap import draw_heatmap
import time
from datetime import datetime
import logging
import os
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
import json
from elevenlabs import ElevenLabs

# Logger initialization
main_logger = logging.getLogger("image_analysis_pipeline")
main_logger.setLevel(logging.INFO)

response_logger = logging.getLogger('request_responses')
response_logger.setLevel(logging.INFO)

prediction_logger = logging.getLogger("predictions")
prediction_logger.setLevel(logging.INFO)

credits_logger = logging.getLogger("credits")
credits_logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("installation.log")
formatter = logging.Formatter('%(asctime)s;%(name)s;%(levelname)s;%(message)s')
file_handler.setFormatter(formatter)
main_logger.addHandler(file_handler)

file_handler = logging.FileHandler("responses.log")
formatter = logging.Formatter('%(asctime)s;%(name)s;%(levelname)s;%(message)s')
file_handler.setFormatter(formatter)
response_logger.addHandler(file_handler)

file_handler = logging.FileHandler("predictions.csv")
formatter = logging.Formatter('%(asctime)s;%(message)s')
file_handler.setFormatter(formatter)
prediction_logger.addHandler(file_handler)

file_handler = logging.FileHandler("credits.csv")
formatter = logging.Formatter('%(asctime)s;%(name)s;%(levelname)s;%(message)s')
file_handler.setFormatter(formatter)
credits_logger.addHandler(file_handler)


# URL served by python -m http.server
URL = "http://localhost:8000/index.html"
OUT_FILE = "rendered_index.html"

# try common Chrome locations on Windows
CHROME_CANDIDATES = [
    r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

def find_chrome():
    for p in CHROME_CANDIDATES:
        p = os.path.expandvars(p)
        if p and Path(p).exists():
            return str(p)
    return None

async def fetch_rendered_html(url, out_file):
    chrome_path = find_chrome()
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=chrome_path, headless=True) if chrome_path else await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")
        # optionally wait for a specific DOM node that your JS inserts, e.g. '#content'
        # await page.wait_for_selector('#content', timeout=5000)
        html = await page.content()  # full rendered HTML
        await browser.close()
    Path(out_file).write_text(html, encoding="utf-8")
    print(f"Rendered HTML saved to {out_file}")

def save_print_pdf(url=URL, out_dir='docs'):
    """
    Render the served URL with Playwright (prefer installed Chrome) and save a PDF into out_dir.
    Returns the saved PDF path or None on error.
    """
    async def _render(url, out_dir):
        chrome_path = find_chrome()
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_path = out_dir_path / f'print_{timestamp}.pdf'

        async with async_playwright() as p:
            if chrome_path:
                browser = await p.chromium.launch(executable_path=chrome_path, headless=True)
            else:
                browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until='networkidle')
            # set margins to 0 and include backgrounds
            await page.pdf(
                path=str(pdf_path),
                format='A4',
                print_background=True,
                margin={'top': '0cm', 'right': '0cm', 'bottom': '0cm', 'left': '0cm'}
            )
            await browser.close()
        return str(pdf_path)

    try:
        return asyncio.run(_render(url, out_dir))
    except Exception as e:
        main_logger.error(f"Failed to render PDF: {e}")
        return None

def run_analysis_pipeline():
    '''
    Function to run the pipeline: main_logger initialization, model initialization, classification, psychoanalysis, poem writing, \
text-to-speech, heatmap generation, and updating the website

    Params:
        None
    
    Returns:
        True/False: depends on whether the pipeline ran successfully
    '''

    start_time = datetime.now()
    main_logger.info(f"Started analysis pipeline at {start_time}")

    image = "image.png"

    try:
        # Model setup
        main_logger.info("Setting up models")
        analysis_model = LLMModel('8a79f5ec-c5dd-448c-9611-99610792a04b')
        poem_model = LLMModel('959058ad-4417-49e3-9e71-252ee2fb033d')
        init_response_a = analysis_model.initialize()
        response_logger.info(init_response_a)
        init_response_p = poem_model.initialize()
        response_logger.info(init_response_p)
        client = el_tts_setup('kui/.env')
        main_logger.info("Setup complete")

        # Run classification
        main_logger.info("Starting classification")
        label = run_classification(f'{image}')
        prediction_logger.info(label)
        main_logger.info(f"Result of the classification: {label}")

        # Analysis
        main_logger.info("Starting analysis")
        analysis = analysis_model.chat(message=label)
        main_logger.info("Analysis complete")

        # Poem
        main_logger.info("Starting writing the poem")
        poem = poem_model.chat(analysis)
        main_logger.info("Poem complete")

        try:
            with open('.env', 'r') as env:
                api_key = env.readlines()[0].split('=')[1].strip()
            el_client = ElevenLabs(api_key=api_key)
            user_before = el_client.user.get()
            credits_before = user_before.subscription.character_count
        except Exception as e:
            credits_logger.error(f"Could not get credits before TTS: {e}")
            credits_before = None

        # Text-to-speech
        #main_logger.info("Starting text-to-speech")
        #poem_filename = run_el_tts(client=client, text=poem)
        #main_logger.info("Text-to-speech complete")

        try:
            user_after = el_client.user.get()
            credits_after = user_after.subscription.character_count
            credits_used = credits_after - credits_before if credits_before else len(poem)
            credits_logger.info(f"{credits_before};{credits_after};{credits_used}")
        except Exception as e:
            credits_logger.error(f"Could not calculate credits used: {e}")

        # Generate heatmap
        main_logger.info("Generating heatmap")
        heatmap_filename = draw_heatmap()
        main_logger.info("Heatmap complete")

        # Generate content.json for website
        main_logger.info("Updating website")
        website_data = {
            "scribblePath": f'{image}',
            "heatmapPath": heatmap_filename,
            "psychoanalysis": analysis,
            "poem": poem,
            "poemAudio": poem_filename, # variable!
            "timestamp": datetime.now().isoformat()
        }
        
        # Write data to a .json file
        with open('content.json', 'w', encoding='utf-8') as f:
            json.dump(website_data, f, indent=2, ensure_ascii=False)
        
        main_logger.info("Website updated")

        # create print-view PDF and save into docs/
        pdf_file = save_print_pdf(URL, out_dir='docs')
        if pdf_file:
            main_logger.info(f"Saved print PDF: {pdf_file}")
            website_data['printPDF'] = pdf_file.replace('\\', '/')
            with open('content.json', 'w', encoding='utf-8') as f:
                json.dump(website_data, f, indent=2, ensure_ascii=False)

        main_logger.info(f"Program concluded in {datetime.now() - start_time}")
        return True
    
    except Exception as e:
        main_logger.error(f"Error during analysis: {str(e)}")
        print(f"Error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    image_path = r'image.png'
    last_modified_time = None
    
    print("Watching for changes in image.png... Press Ctrl+C to stop.")
    
    asyncio.run(fetch_rendered_html(URL, OUT_FILE))

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