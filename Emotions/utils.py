import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


def getModelAndTokenizer(MODEL_PATH, quantize):

    bnb_config = None
    if quantize:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False
        )

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        # dtype="float16",
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


def clean_output(outputs, chunks):

    extracted_data = []
    for idx, output in enumerate(outputs):
        if "Answer:" in output:
            clean_lines = output.split("Answer:", 1)[1].strip()
            extracted_data.append(clean_lines)
        else:
            print(f"\nFix needed for {output} !\n")
            # backup in case the Model fails
            extracted_data.append(chunks[idx])

    return extracted_data


def convert_to_sentences(content):
    lines = []
    for para in content:
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.)\s'
        for se in re.split(pattern, para):
            lines.append(re.sub(r'(^|\s)\d+\.\s*', r'\1', se))
    return lines

