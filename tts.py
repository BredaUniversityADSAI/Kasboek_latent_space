import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)

prompt = """In whispers of the wind, I hear a call,
A mystery that beckons, one and all.
To seek, to find, to cherish and to share,
The meaning of life, hidden, yet beyond compare."""
description = "An elderly male speaker delivers a mysterious but welcoming speech with a slow speed and low pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)