import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import bitsandbytes
from tqdm import tqdm

from Generator.utils import createChunks


def getModelAndTokenizer(MODEL_PATH):

    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=6.0,
    #     llm_int8_enable_fp32_cpu_offload=False
    # )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        # quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=MODEL_PATH
    )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
                                              cache_dir=MODEL_PATH)

    tokenizer.pad_token = tokenizer.eos_token
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def generate_correct_lines(model, tokenizer, lines):
    # batch_prompts = "You are a grammar and spelling corrector. Answer only the corrected sentence — no explanations."\
    #         + "Example 1: this bad grammer Answer 1: This is a bad grammer." \
    #         + "Example 2: This is a good grammer. Answer 2: This is a good grammer." \
    #         + "Based on the above example and answer, correct the following sentences:" \
    #         + f"Sentences: {(",".join(lines))} Answer:"

    sentences = (",".join(lines))

    batch_prompts = "You are a precise spelling and grammar corrector. Correct only clear spelling and grammar errors." + \
                    ("Do NOT change proper nouns, names, technical terms, or uncommon words. Capitalize nouns and "
                     "proper nouns (e.g., kuret → Kuret).") + \
                    ("Do NOT replace or guess uncommon or rare words — keep them as they are. Do NOT change word "
                     "meaning, structure, or punctuation unnecessarily.") + \
                    "Example 1: this bad grammer Answer 1: This is a bad grammar." + \
                    "Example 2: This is a good grammer. Answer 2: This is a good grammar." + \
                    "Example 3: Input: kuret was in the air. Answer 3: Kuret was in the air." \
                    "Based on the above example and answer, correct the following sentences:" + \
                    f"Sentences: {sentences} Answer:"

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return [d.strip() for d in decoded]


def sanitise(Args, content):

    MODEL_PATH = Args.Emotions.ModelPath.__dict__[Args.Platform]
    BATCH_SIZE = Args.Emotions.BatchSize

    model, tokenizer = getModelAndTokenizer(MODEL_PATH)

    # torch.cuda.set_per_process_memory_fraction(0.9)

    chunks = createChunks(content)

    corrected = []
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Processing", ncols=100):
        batch = chunks[i: i + BATCH_SIZE]
        para_break = -1
        for idx, line in enumerate(batch):
            if not line.strip():
                para_break = idx
                break

        if para_break > -1:
            batch = list(batch).pop(para_break)
        corrected_lines = generate_correct_lines(model, tokenizer, batch)[0]
        corrected_lines = corrected_lines.split("Answer:", 1)[1].strip()
        corrected_lines = [line.strip() + '.' for line in corrected_lines.split('.')]
        corrected_lines.insert(para_break, " ")
        corrected.extend(corrected_lines)
        torch.cuda.empty_cache()

    return corrected
