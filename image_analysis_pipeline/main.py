from chat import *
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
        logging.error(f"Failed to render PDF: {e}")
        return None

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

    image = "image.png"

    try:
        # Logger initialization
        logging.basicConfig(filename='installation.log', level=logging.INFO, format='%(asctime)s;%(name)s;%(levelname)s;%(message)s')
        start_time = datetime.now()
        logging.info(f"Started script at {start_time}")

        # Model setup
        # dpa: (d)er (P)sycho(a)nalytiker
        # poesie: name of the poet, meaning 'poetry' in French
        logging.info("Setting up models...")
        ASSISTANT_KEY_DPA, USER_KEY, init_endpoint_url_dpa, ask_endpoint_url_dpa, headers = credentials('8a79f5ec-c5dd-448c-9611-99610792a04b')
        ASSISTANT_KEY_POESIE, USER_KEY, init_endpoint_url_poesie, ask_endpoint_url_poesie, headers = credentials('959058ad-4417-49e3-9e71-252ee2fb033d')
        response_dpa = initialize_ai_assistant(ASSISTANT_KEY_DPA, USER_KEY, init_endpoint_url_dpa, headers)
        response_poesie = initialize_ai_assistant(ASSISTANT_KEY_POESIE, USER_KEY, init_endpoint_url_poesie, headers)
        #client = el_tts_setup()
        logging.info("Setup complete!")

        # Run classification
        logging.info("Starting classification...")
        label = run_classification(f'{image}')
        logging.info(f"Result of the classification: {label}")

        # Psychoanalysis
        logging.info("Starting psychoananalysis...")
        psychoanalysis = chat_with_ai_assistant(response=response_dpa, message=label, USER_KEY=USER_KEY, ask_endpoint_url=ask_endpoint_url_dpa, ASSISTANT_KEY=ASSISTANT_KEY_DPA, headers=headers)
        logging.info("Psychanalysis complete!")

        # Poem
        logging.info("Starting writing the poem...")
        poem = chat_with_ai_assistant(response=response_poesie, message=psychoanalysis, USER_KEY=USER_KEY, ask_endpoint_url=ask_endpoint_url_poesie, ASSISTANT_KEY=ASSISTANT_KEY_POESIE, headers=headers)
        logging.info("Poem complete!")

        # Text-to-speech
        #logging.info("Starting text-to-speech...")
        #poem_filename = run_el_tts(client=client, text=poem)
        #logging.info("Text-to-speech complete!")

        # Generate heatmap
        logging.info("Generating heatmap...")
        heatmap_filename = draw_heatmap()
        logging.info("Heatmap complete!")

        # Generate content.json for website
        logging.info("Updating website...")
        website_data = {
            "scribblePath": f'{image}',
            "heatmapPath": heatmap_filename,
            "psychoanalysis": psychoanalysis,
            "poem": poem,
            "poemAudio": "output.mp3", # variable!
            "timestamp": datetime.now().isoformat()
        }
        
        # Write data to a .json file
        with open('content.json', 'w', encoding='utf-8') as f:
            json.dump(website_data, f, indent=2, ensure_ascii=False)
        
        logging.info("Website updated!")

        # create print-view PDF and save into docs/
        pdf_file = save_print_pdf(URL, out_dir='docs')
        if pdf_file:
            logging.info(f"Saved print PDF: {pdf_file}")
            website_data['printPDF'] = pdf_file.replace('\\', '/')
            with open('content.json', 'w', encoding='utf-8') as f:
                json.dump(website_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Program concluded in {datetime.now() - start_time}.")
        return True
    
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
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