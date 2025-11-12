import torch
from tqdm import tqdm
from Emotions.utils import getModelAndTokenizer, clean_output

emotions_list = [
    "<laugh>",
    "<laugh_harder>",
    "<sigh>",
    "<chuckle>",
    "<gasp>",
    "<angry>",
    "<excited>",
    "<whisper>",
    "<cry>",
    "<scream>",
    "<sing>",
    "<snort>",
    "<exhale>",
    "<gulp>",
    "<giggle>",
    "<sarcastic>",
    "<curious>",
]

prompt = """
You are an expressive text enhancer that adds emotional cues to writing.
Your task is to insert the most appropriate emotion tags **after** specific words or phrases in a paragraph.
Emotion tags you may use: <laugh>, <laugh_harder>, <sigh>, <chuckle>, <gasp>, <angry>, <excited>, <whisper>, <cry>, <scream>, <sing>, <snort>, <exhale>, <gulp>, <giggle>, <sarcastic>, <curious>.
Keep the original text exactly as written. Do not paraphrase or rewrite.
Insert emotion tags only where natural emotional cues are implied.
Add the tag **right after** the word or phrase it applies to, before punctuation if it fits better.
Use tags sparingly and only when contextually appropriate.
Reflect tone (anger, excitement, sadness, curiosity, humor, sarcasm, etc.) based on clues in the text.
Example 1:
Input: "He looked at the strange machine and tilted his head."
Answer: "He looked at the strange machine <curious> and tilted his head."
Example 2:
Input: "I can’t believe you actually did that. You’re unbelievable!"
Answer: "I can’t believe you actually did that. <gasp> You’re unbelievable! <laugh>"
Example 3:
Input: "I tried so hard, but it still wasn’t enough."
Answer: "I tried so hard, <sigh> but it still wasn’t enough <cry>."
Now process the following paragraph and add the most appropriate emotion tags:
{}
Answer:
"""


def generate_emotion_lines(model, tokenizer, paragraph):

    batch_prompts = prompt.replace("{}", paragraph)

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(
        model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return [d.strip() for d in decoded]


def addEmotions(Args, content):

    MODEL_PATH = Args.Emotions.ModelPath.__dict__[Args.Platform]

    model, tokenizer = getModelAndTokenizer(MODEL_PATH, Args.Emotions.Quantize)

    outputs = []
    for batch in tqdm(content, desc="Processing", ncols=100):

        emotion_lines = generate_emotion_lines(model, tokenizer, batch)
        outputs.extend(emotion_lines)
        torch.cuda.empty_cache()

    emotives = clean_output(outputs, [])

    return emotives

