from pydub import AudioSegment


def content_stats(content):
    count = len(content.replace(" ", "").replace("\n", ""))
    word_count = len(content.split())
    lines = len(content.splitlines())
    return f"{count} characters, {word_count} words, {lines} lines"


def createChunks(content, limit):

    chunks = []
    while len(content) >= limit:
        split_pos = content.rfind('.', 0, limit)
        if split_pos == -1:
            split_pos = limit
        chunk = content[:split_pos + 1].strip()
        chunks.append(chunk)
        content = content[split_pos + 1:].strip()

    if content:
        chunks.append(content)

    return chunks


def merge_audio(files, output_file, format="mp3"):

    combined = AudioSegment.empty()
    for file in files:
        print(f"Merging {file}...")
        segment = None
        if format == "mp3":
            segment = AudioSegment.from_mp3(file)
        elif format == "wav":
            segment = AudioSegment.from_wav(file)

        combined += segment

    combined.export(output_file, format=format)

    print(f"Merged audio saved as {output_file}")