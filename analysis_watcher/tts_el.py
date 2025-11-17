from elevenlabs.client import ElevenLabs
import os

def el_tts_setup():
    '''
    Initialize ElevenLabs client

    Params:
        None
    
    Returns:
        client: ElevenLabs client
    '''

    with open('.env', 'r') as env:
        api_key = env.readlines()[0].split('=')[1].strip()
    client = ElevenLabs(api_key=api_key)

    return client

def run_el_tts(client, text: str, voice_id='R8MYc5Q5y0TurlOQoX88', output='output.mp3'):
    '''
    Convert text to speech

    Params:
        client: ElevenLabs client obtained from the tts_setup function
        text: response of the LLM
        voice_id: ID of the ElevenLabs vocie
        output: filename including the filetype extension

    Returns:
        output: filename including the filetype extension
    '''
    audio = client.text_to_speech.convert(text=text, voice_id=voice_id, model_id="eleven_v3", output_format="mp3_44100_128")
    with open(f'{output}', 'wb') as file:
        for chunk in audio:
            file.write(chunk)

    return output

def open_audio_file(filename: str):
    '''
    Open audio file and play
    
    Params:
        filename: name of the audio file to be played with the filetype extension
    
    Returns:
        None
    '''

    # Open the audio in the default audio player
    os.open(f"{filename}")