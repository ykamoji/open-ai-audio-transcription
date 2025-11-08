from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv
from OIGenerator.utils import createChunks
import os
import requests
import time

CHUNK_LIMIT = 1000

load_dotenv(override=True)

if "OPEN_AI_KEY" not in os.environ:
    raise Exception("Load OPEN AI Key Access token first !")

client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])

instruction = (
    "You are a professional audiobook narrator. "
    "Read the following story in a warm, expressive tone, "
    "with natural pacing, emotional inflection, and pauses after sentences:\n\n"
)


def merge_audio(files, output_file):

    combined = AudioSegment.empty()
    for file in files:
        print(f"Merging {file}...")
        segment = AudioSegment.from_mp3(file)
        combined += segment

    combined.export(output_file, format="mp3")
    print(f"Merged audio saved as {output_file}")


def convert(content, title):

    chunks = createChunks(content, CHUNK_LIMIT)[:5]

    part_count = 0
    audio_files = []
    for part, chunk in enumerate(chunks):
        # response = client.audio.speech.create(
        #     model="whisper-1",
        #     voice="ash",
        #     instructions=instruction,
        #     speed=1.3,
        #     input=chunk
        # )
        print(chunk)

        mp3file = f"{title}_{part_count}.mp3"

        # with open(mp3file, "wb") as f:
        #     f.write(response.read())

        audio_files.append(mp3file)
        part_count += 1
        print(f"Part {part_count}/{len(chunks)} done")
        # time.sleep(30)

    # merge_audio(audio_files, f"{title}.mp3")


