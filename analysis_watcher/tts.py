from google.cloud import texttospeech
import os

def tts_setup(voice_variant='A', speaking_rate=1.0, pitch=0.0):
    '''
    Configure Google Cloud credentials and TTS properties

    Params:
        voice_variant: one of the 3 variants of the en-US-Neural2 voice (either A, B or C)
        speaking_rate: speed of the speech
        pitch: pitch of the speech

    Returns:
        client: the Google Cloud client used to execute commands
        voice: the configured voice
        audio_config: properties of the audio
    '''

    # Initialize the client
    client = texttospeech.TextToSpeechClient()

    # Set up voice parameters
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=f"en-US-Chirp3-HD-Zephyr",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # Configure the audio output format
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
        pitch=pitch
    )

    return client, voice, audio_config

def run_tts(client, voice, audio_config, text: str, output: str):

    '''
    Synthetize speech from the given text

    Params:
        client: client used to execute commands
        voice: the configured voice
        audio_config: properties of the audio
        text: the response of a model
        output: name of the audio file without the filetype extension
    
    Returns:
        filename: the value of the output variable and the filetype extension appended to it
    '''

    # Define the text input
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Generate the speech
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    # Save the output
    filename = f"{output}.mp3"
    with open(filename, "wb") as out:
        out.write(response.audio_content)

    return filename

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