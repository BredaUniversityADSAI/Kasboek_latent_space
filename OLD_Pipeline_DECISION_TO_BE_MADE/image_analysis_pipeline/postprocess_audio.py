import pydub

def decrease_volume(audio_path: str, dB: int, output_path: str = None):
    """
    Decrease the volume of an audio file by dB decibels.
    
    Parameters:
    - audio_path: path to the audio file (mp3, wav, etc.)
    - dB: amount to decrease in decibels (e.g., -10)
    - output_path: where to save the result (defaults to overwriting input)
    """
    if output_path is None:
        output_path = audio_path
    
    # Load audio file
    audio = pydub.AudioSegment.from_file(audio_path)
    
    # Decrease volume
    silent_audio = audio + dB  # negative dB decreases volume
    
    # Export to file
    silent_audio.export(output_path, format='mp3')
    print(f"Volume decreased by {dB}dB. Saved to {output_path}")


def overlay_audio(audio1='output_ryan.mp3', audio2='output_rae.mp3', output_name='output.mp3'):
    audio1 = pydub.AudioSegment.from_file(audio1)
    audio2 = pydub.AudioSegment.from_file(audio2)
    output = audio2.overlay(audio1)

    output.export('output.mp3', format='mp3')

    return output_name