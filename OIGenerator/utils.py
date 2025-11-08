def content_stats(content):
    count = len(content.replace(" ", "").replace("\n", ""))
    word_count = len(content.split())
    lines = len(content.splitlines())
    return f"{count} characters, {word_count} words, {lines} lines"


def createChunks(content, limit):

    chunks = []
    while len(content) >= limit:
        # Try to split on last sentence or paragraph near the limit
        split_pos = content.rfind('.', 0, limit)
        if split_pos == -1:
            split_pos = limit
        chunk = content[:split_pos + 1].strip()
        chunks.append(chunk)
        content = content[split_pos + 1:].strip()

    if content:
        chunks.append(content)

    return chunks
