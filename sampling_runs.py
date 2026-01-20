from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import json
import argparse
from tqdm import tqdm
import re
from transformers import StoppingCriteriaList, StoppingCriteria

class KeywordsStoppingCriteriaCase(StoppingCriteria):
    '''
    D: designed for early stopping with pre-defined keywords on case inference
    I: keywords ids [list]
    '''
    def __init__(self, keywords_ids: list):
        self.keywords = keywords_ids

    '''
    D: designed for early stopping with pre-defined keywords
    '''
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # only need to detect the last token (new generated)
        if input_ids[0][-1] in self.keywords:
            return True
        return False



def parse_args():
    parser = argparse.ArgumentParser(description='Translation script with configurable parameters')
    parser.add_argument('--input_name', type=str, required=True, 
                        help='Pre-defined name of the file (to automatically load model and dataset)')
    parser.add_argument('--sampling_size', type=int, default=10,
                        help='Sampling size for generation')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Temperature of generation (default: 0.5)')
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy decoding (num_return_sequences=1, do_sample=False)')
    parser.add_argument('--model_path', type=str, default="",
                        help='Local cached model path. If none, use default from huggingface or modelscope')
    parser.add_argument('--cache_dir', type=str, default="",
                        help='Cache directory for models and tokenizers')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to store the dataset')
    parser.add_argument('--storage_path', type=str, required=True,
                        help='Path to store the generation results')
    return parser.parse_args()

def determine_mode_and_quantization(input_name):
    """Determine processing mode and whether to use quantization based on input_name"""
    input_name_lower = input_name.lower()
    
    # Check if quantization is used - new logic based on last number
    numbers = re.findall(r'\d+', input_name)
    use_quantization = False
    if numbers and "mbart" not in input_name_lower:
        last_number = int(numbers[-1])
        use_quantization = last_number > 13
    
    # Determine mode
    if 'mbart' in input_name_lower:
        mode = 'mbart'
    elif 'nllb' in input_name_lower:
        mode = 'nllb'
    elif 'deepseek' in input_name_lower:
        mode = 'deepseek'
    elif 'qwen3' in input_name_lower:
        if 'reason' in input_name_lower:
            mode = 'qwen3_think'
        elif 'chat' in input_name_lower:
            mode = 'qwen3_nothink'
        else:
            mode = 'qwen3_nothink'  # Default no-think
    elif 'pre' in input_name_lower:
        mode = 'pretrain'
    elif 'chat' in input_name_lower:
        mode = 'instruct'
    else:
        mode = 'instruct'  # Default to instruct mode
    
    return mode, use_quantization

def get_model_name_from_input(input_name):
    """Extract model name from input_name"""
    parts = input_name.split('_', 1)
    if len(parts) > 1:
        return parts[1]
    return input_name

def get_dataset_info_from_input(input_name):
    """Determine dataset information from input_name"""
    dataset_part = input_name.split('_')[0]
    if '23en-zh' in dataset_part:
        return '23en-zh', 'English', 'Chinese'
    elif '23zh-en' in dataset_part:
        return '23zh-en', 'Chinese', 'English'
    elif '24en-zh' in dataset_part:
        return '24en-zh', 'English', 'Chinese'
    else:
        raise ValueError(f"Unknown dataset format in input_name: {input_name}")

def load_model_and_tokenizer(model_path, model_name, use_quantization=False, cache_dir=""):
    """Load model and tokenizer"""
    print("Loading model...")
    
    # If model_path is provided, use it; otherwise use model_name directly
   
    full_model_path = model_path
    
    # Prepare tokenizer loading parameters
    tokenizer_kwargs = { 'trust_remote_code': True}
    if cache_dir:
        tokenizer_kwargs['cache_dir'] = cache_dir
    
    tokenizer = AutoTokenizer.from_pretrained(full_model_path, **tokenizer_kwargs) #**tokenizer_kwargs, 
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model loading parameters
    model_kwargs = {
        'trust_remote_code': True,
        'device_map': "auto"
    }
    if cache_dir:
        model_kwargs['cache_dir'] = cache_dir
    
    # Decide loading method based on use_quantization parameter
    if use_quantization:
        print("Loading with 4-bit quantization...")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model_kwargs['quantization_config'] = bnb_config
        model = AutoModelForCausalLM.from_pretrained(full_model_path, **model_kwargs)
    else:
        print("Loading normally with FP16...")
        model_kwargs['torch_dtype'] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(full_model_path, **model_kwargs)
    
    return model, tokenizer

def load_nllb_model_and_tokenizer(model_path, model_name, use_quantization=False, cache_dir=""):
    """Load NLLB model and tokenizer"""
    print("Loading NLLB model...")
    
    # If model_path is provided, use it; otherwise use model_name directly
    full_model_path = model_path
    
    # Prepare tokenizer loading parameters
    tokenizer_kwargs = {
        'trust_remote_code': True,
        'src_lang': 'eng_Latn',
        'tgt_lang': 'zho_Hans'
    }
    if cache_dir:
        tokenizer_kwargs['cache_dir'] = cache_dir
    
    tokenizer_in = AutoTokenizer.from_pretrained(full_model_path, **tokenizer_kwargs)
    
    # Prepare parameters for tokenizer_out
    tokenizer_kwargs['src_lang'] = 'zho_Hans'
    tokenizer_kwargs['tgt_lang'] = 'eng_Latn'
    tokenizer_out = AutoTokenizer.from_pretrained(full_model_path, **tokenizer_kwargs)
    
    # Prepare model loading parameters
    model_kwargs = {
        'trust_remote_code': True,
        'device_map': 'auto'
    }
    if cache_dir:
        model_kwargs['cache_dir'] = cache_dir
    
    # Decide loading method based on use_quantization parameter
    if use_quantization:
        print("Loading NLLB with 4-bit quantization...")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model_kwargs['quantization_config'] = bnb_config
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_path, **model_kwargs)
    else:
        print("Loading NLLB normally with FP16...")
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_path, **model_kwargs)
    
    return model, tokenizer_in, tokenizer_out

def load_mbart_model_and_tokenizer(model_path, model_name, use_quantization=False, cache_dir=""):
    """Load mBART model and tokenizer"""
    print("Loading mBART model...")
    
    # If model_path is provided, use it; otherwise use model_name directly
    full_model_path = model_path
    
    # Prepare tokenizer loading parameters
    tokenizer_kwargs = {
        'trust_remote_code': True
    }
    if cache_dir:
        tokenizer_kwargs['cache_dir'] = cache_dir
    
    tokenizer = AutoTokenizer.from_pretrained(full_model_path, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model loading parameters
    model_kwargs = {
        'trust_remote_code': True,
        'device_map': 'auto'
    }
    if cache_dir:
        model_kwargs['cache_dir'] = cache_dir
    
    # Decide loading method based on use_quantization parameter
    if use_quantization:
        print("Loading mBART with 4-bit quantization...")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model_kwargs['quantization_config'] = bnb_config
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_path, **model_kwargs)
    else:
        print("Loading mBART normally with FP16...")
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_path, **model_kwargs)
    
    return model, tokenizer

def translate_line_instruct(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, greedy=False):
    """Instruct mode translation"""
    if not text.strip():
        return []

    messages = [{"role": "user", "content": f"Translate the following {src_lang} text to {tgt_lang}. Only provide the translation, no explanations:\n\n{text}"}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    # Set generation parameters based on greedy parameter
    if greedy:
        num_return_sequences = 1
        do_sample = False
        generate_kwargs = {
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': 2048,
            'do_sample': do_sample,
            'num_beams': 1,
            'pad_token_id': tokenizer.eos_token_id
        }
    else:
        num_return_sequences = sampling_size
        do_sample = True
        generate_kwargs = {
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': 2048,
            'temperature': temperature,
            'top_k': 0,
            'top_p': 1.0,
            'do_sample': do_sample,
            'num_beams': 1,
            'pad_token_id': tokenizer.eos_token_id
        }

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            **generate_kwargs
        )
    outputs = outputs[:, inputs.input_ids.shape[-1]:]
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    new_translations = []
    for translation in translations:
        translation = translation.strip().split('\n')[0].strip()
        if translation: # only keep the first line of the translation
            new_translations.append(translation)
    return new_translations

def translate_line_pretrain(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, greedy=False):
    """Pre-train mode translation (few-shot)"""
    if not text.strip():
        return []

    if src_lang == "English" and tgt_lang == "Chinese":
        prompt = f"""Translate the following English sentences to Chinese:

English: The weather is beautiful today. 
Chinese: 今天天气很好。 

English: How are you doing? 
Chinese: 你好吗？ 

English: I'm looking forward to our meeting tomorrow. 
Chinese: 我期待着我们明天的会议。 

English: The rapid development of technology has changed our daily lives significantly. 
Chinese: 技术的快速发展显著改变了我们的日常生活。 

English: Could you please help me with this problem? 
Chinese: 你能帮我解决这个问题吗？ 

English: {text} 
Chinese:"""

    elif src_lang == "Chinese" and tgt_lang == "English":
        prompt = f"""Translate the following Chinese sentences to English:

Chinese: 今天天气很好。 
English: The weather is beautiful today. 

Chinese: 你好吗？ 
English: How are you doing? 

Chinese: 我期待着我们明天的会议。 
English: I'm looking forward to our meeting tomorrow. 

Chinese: 技术的快速发展显著改变了我们的日常生活。 
English: The rapid development of technology has changed our daily lives significantly. 

Chinese: 你能帮我解决这个问题吗？ 
English: Could you please help me with this problem? 

Chinese: {text} 
English:"""

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # Set generation parameters based on greedy parameter
    if greedy:
        num_return_sequences = 1
        do_sample = False
        generate_kwargs = {
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': 2048,
            'do_sample': do_sample,
            'num_beams': 1,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id
        }
        print(f"Greedy mode, num_return_sequences: {num_return_sequences}")
    else:
        num_return_sequences = sampling_size
        do_sample = True
        generate_kwargs = {
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': 2048,
            'temperature': temperature,
            'top_k': 0,
            'top_p': 1.0,
            'do_sample': do_sample,
            'num_beams': 1,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id
        }
    stop_words = ['\n\n', '\n', '. \n\n', '。\n\n', ' \n\n']
    stop_ids = [tokenizer.encode(w)[-1] for w in stop_words]
    stop_criteria = KeywordsStoppingCriteriaCase(stop_ids)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            **generate_kwargs,
            stopping_criteria=StoppingCriteriaList([stop_criteria])
        )
    outputs = outputs[:, inputs.input_ids.shape[-1]:]
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    new_translations = []
    for translation in translations:
        translation = translation.strip().split('\n')[0].strip()
        if translation:
            new_translations.append(translation)
    return new_translations

def translate_line_qwen3(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, think_mode=False, greedy=False):
    """Qwen3 mode translation"""
    if not text.strip():
        return []

    messages = [{"role": "user", "content": f"Translate the following {src_lang} text to {tgt_lang}. Only provide the translation, no explanations:\n\n{text}"}]
    
    try:
        if think_mode:
            text_input = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=True
            )
        else:
            text_input = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=False
            )
    except TypeError:
        text_input = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    # Set generation parameters based on greedy parameter
    if greedy:
        num_return_sequences = 1
        do_sample = False
        generate_kwargs = {
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': 2048 if think_mode else 2048,
            'do_sample': do_sample,
            'num_beams': 1,
            'pad_token_id': tokenizer.eos_token_id
        }
    else:
        num_return_sequences = sampling_size
        do_sample = True
        generate_kwargs = {
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': 2048 if think_mode else 2048,
            'temperature': temperature,
            'top_k': 0,
            'top_p': 1.0,
            'do_sample': do_sample,
            'num_beams': 1,
            'pad_token_id': tokenizer.eos_token_id
        }

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            **generate_kwargs
        )

    translations = []
    
    for output in outputs:
        output_ids = output[len(inputs.input_ids[0]):].tolist()
        
        if think_mode:
            # If there's no thinking token, decode output directly
            if 151668 not in output_ids:
                response = tokenizer.decode(output_ids, skip_special_tokens=True)
            else:
                # Find thinking token range
                thinking_token_idx = output_ids.index(151668)
                try:
                    end_thinking_token_idx = output_ids.index(151669)  # Token id corresponding to </think>
                except ValueError:
                    end_thinking_token_idx = thinking_token_idx  # If not found, prevent error
                
                # Remove thinking range (including <think>...</think>)
                cleaned_ids = output_ids[end_thinking_token_idx + 1:]
                response = tokenizer.decode(cleaned_ids, skip_special_tokens=True)
            
            translations.append(response.strip())
        else:
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            translations.append(response.strip())

    return translations

def extract_final_translation(response):
    """Extract final translation result after </think>"""
    # Find content after </think> tag
    think_end = response.find('</think>')
    if think_end != -1:
        # Extract content after </think>
        final_result = response[think_end + len('</think>'):].strip()
        return final_result
    else:
        # If </think> tag is not found, return original response
        return response.strip()

def translate_line_deepseek(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, greedy=False):
    """DeepSeek mode translation"""
    if not text.strip():
        return [text]

    messages = [{"role": "user", "content": f"Translate the following {src_lang} text to {tgt_lang}. Only provide the translation, no explanations:\n\n{text}"}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    # Set generation parameters based on greedy parameter
    if greedy:
        num_return_sequences = 1
        do_sample = False
        generate_kwargs = {
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': 2048,
            'do_sample': do_sample,
            'num_beams': 1,
            'pad_token_id': tokenizer.eos_token_id
        }
    else:
        num_return_sequences = sampling_size
        do_sample = True
        generate_kwargs = {
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': 2048,
            'temperature': temperature,
            'top_k': 0,
            'top_p': 1.0,
            'do_sample': do_sample,
            'num_beams': 1,
            'pad_token_id': tokenizer.eos_token_id
        }

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            **generate_kwargs
        )

    # Decode all results and extract final translations
    translations = []
    for output in outputs:
        response = tokenizer.decode(output[len(inputs.input_ids[0]):], skip_special_tokens=True)
        final_translation = extract_final_translation(response)
        if final_translation:  # Only add non-empty translation results
            translations.append(final_translation)

    return translations

def translate_line_nllb(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, greedy=False):
    """NLLB mode translation"""
    if not text.strip():
        return [text]

    # Determine which tokenizer and target language code to use based on language direction
    if src_lang == "English" and tgt_lang == "Chinese":
        # Use tokenizer_in (eng_Latn -> zho_Hans)
        tgt_lang_code = 'zho_Hans'
    elif src_lang == "Chinese" and tgt_lang == "English":
        # Use tokenizer_out (zho_Hans -> eng_Latn)  
        tgt_lang_code = 'eng_Latn'
    else:
        raise ValueError(f"Unsupported language pair: {src_lang} -> {tgt_lang}")

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Set generation parameters based on greedy parameter
    if greedy:
        num_return_sequences = 1
        do_sample = False
        generate_kwargs = {
            'forced_bos_token_id': tokenizer.convert_tokens_to_ids(tgt_lang_code),
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': 2048,
            'do_sample': do_sample,
            'num_beams': 1
        }
    else:
        num_return_sequences = sampling_size
        do_sample = True
        generate_kwargs = {
            'forced_bos_token_id': tokenizer.convert_tokens_to_ids(tgt_lang_code),
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': 2048,
            'temperature': temperature,
            'top_k': 0,
            'top_p': 1.0,
            'do_sample': do_sample,
            'num_beams': 1
        }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generate_kwargs
        )

    # Decode all results
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return translations

def translate_line_mbart(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, greedy=False):
    """mBART mode translation"""
    if not text.strip():
        return [text]

    # Determine target language code based on language direction
    if src_lang == "English" and tgt_lang == "Chinese":
        tgt_lang_code = 'zh_CN'
    elif src_lang == "Chinese" and tgt_lang == "English":
        tgt_lang_code = 'en_XX'
    else:
        raise ValueError(f"Unsupported language pair: {src_lang} -> {tgt_lang}")

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Set generation parameters based on greedy parameter
    if greedy:
        num_return_sequences = 1
        do_sample = False
        generate_kwargs = {
            'forced_bos_token_id': tokenizer.lang_code_to_id[tgt_lang_code],
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': 2048,
            'do_sample': do_sample,
            'num_beams': 1
        }
    else:
        num_return_sequences = sampling_size
        do_sample = True
        generate_kwargs = {
            'forced_bos_token_id': tokenizer.lang_code_to_id[tgt_lang_code],
            'num_return_sequences': num_return_sequences,
            'max_new_tokens': 2048,
            'temperature': temperature,
            'top_k': 0,
            'top_p': 1.0,
            'do_sample': do_sample,
            'num_beams': 1
        }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generate_kwargs
        )

    # Decode all results
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return translations

def translate_line(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, mode, greedy=False, tokenizer_out=None):
    """Select translation method based on mode"""
    if mode == 'instruct':
        return translate_line_instruct(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, greedy)
    elif mode == 'pretrain':
        return translate_line_pretrain(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, greedy)
    elif mode == 'qwen3_think':
        return translate_line_qwen3(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, think_mode=True, greedy=greedy)
    elif mode == 'qwen3_nothink':
        return translate_line_qwen3(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, think_mode=False, greedy=greedy)
    elif mode == 'deepseek':
        return translate_line_deepseek(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, greedy)
    elif mode == 'nllb':
        # For NLLB, select appropriate tokenizer based on translation direction
        if src_lang == "English" and tgt_lang == "Chinese":
            return translate_line_nllb(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, greedy)
        elif src_lang == "Chinese" and tgt_lang == "English":
            return translate_line_nllb(text, model, tokenizer_out, src_lang, tgt_lang, sampling_size, temperature, greedy)
        else:
            raise ValueError(f"Unsupported language pair for NLLB: {src_lang} -> {tgt_lang}")
    elif mode == 'mbart':
        return translate_line_mbart(text, model, tokenizer, src_lang, tgt_lang, sampling_size, temperature, greedy)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def load_references(ref_file):
    """Load reference translation file"""
    if os.path.exists(ref_file):
        with open(ref_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    else:
        print(f"Warning: Reference file {ref_file} does not exist")
        return []

def get_dataset_files(dataset_name, dataset_path):
    """Get source file and reference file based on dataset name and path"""
    if dataset_name == '23en-zh':
        src_file = os.path.join(dataset_path, '23en-zh.en')
        ref_file = os.path.join(dataset_path, '23en-zh.zh')
    elif dataset_name == '23zh-en':
        src_file = os.path.join(dataset_path, '23zh-en.zh')
        ref_file = os.path.join(dataset_path, '23zh-en.en')
    elif dataset_name == '24en-zh':
        src_file = os.path.join(dataset_path, '24en-zh.en')
        ref_file = os.path.join(dataset_path, '24en-zh.zh')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return src_file, ref_file

def main():
    args = parse_args()
    
    # Determine processing mode and whether to use quantization
    mode, use_quantization = determine_mode_and_quantization(args.input_name)
    
    # Parse information from input_name
    model_name = get_model_name_from_input(args.input_name)
    dataset_name, src_lang, tgt_lang = get_dataset_info_from_input(args.input_name)
    
    # Load model based on mode
    tokenizer_out = None

    greedy_suffix = "_greedy" if args.greedy else "_sampling"
    
    # If using greedy, sampling_size is displayed as 1 in filename
    effective_sampling_size = 1 if args.greedy else args.sampling_size
    output_filename = f"{args.input_name}_{effective_sampling_size}{greedy_suffix}.json"
    output_file = os.path.join(args.storage_path, output_filename)
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping translation")
        return
    if mode == 'nllb':
        model, tokenizer, tokenizer_out = load_nllb_model_and_tokenizer(
            args.model_path, model_name, use_quantization, args.cache_dir
        )
    elif mode == 'mbart':
        model, tokenizer = load_mbart_model_and_tokenizer(
            args.model_path, model_name, use_quantization, args.cache_dir
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.model_path, model_name, use_quantization, args.cache_dir
        )
    
    # Get dataset file paths
    src_file, ref_file = get_dataset_files(dataset_name, args.dataset_path)
    
    # Generate output filename
    mode_suffix = ""
    if mode == 'qwen3_think':
        mode_suffix = "_think"
    elif mode == 'qwen3_nothink':
        mode_suffix = "_chat"
    elif mode == 'pretrain':
        mode_suffix = "_pretrain"
    elif mode == 'instruct':
        mode_suffix = "_instruct"
    elif mode == 'deepseek':
        mode_suffix = "_deepseek"
    elif mode == 'nllb':
        mode_suffix = "_nllb"
    elif mode == 'mbart':
        mode_suffix = "_mbart"
    
    quantization_suffix = "_4bit" if use_quantization else "_fp16"
    
    
    
    
    # Ensure output directory exists
    os.makedirs(args.storage_path, exist_ok=True)
    # testing
    #output_file = output_file.replace(".json", "_test.json")

    # Check if output file already exists
    
    
    # Read source file
    with open(src_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # testing
    #lines=lines[:2]
    # Load reference translations
    references = load_references(ref_file)
    
    print(f"Starting translation for {dataset_name}...")
    print(f"Model: {model_name}")
    print(f"Mode: {mode}")
    print(f"Loading mode: {'4-bit quantization' if use_quantization else 'FP16 normal loading'}")
    print(f"Decoding mode: {'Greedy' if args.greedy else 'Sampling'}")
    if not args.greedy:
        print(f"Sampling size: {args.sampling_size}")
        print(f"Temperature: {args.temp}")
    print(f"Source language: {src_lang} -> Target language: {tgt_lang}")
    if args.cache_dir:
        print(f"Cache directory: {args.cache_dir}")
    
    dataset = []
    for i, line in tqdm(enumerate(lines), total=len(lines)):
        if line.strip():
            source_text = line.strip()
            translations = translate_line(
                source_text, model, tokenizer, src_lang, tgt_lang, 
                args.sampling_size, args.temp, mode, args.greedy, tokenizer_out
            )
            # Get corresponding reference translation
            reference = references[i] if i < len(references) else ""
            dataset.append({
                "id": i + 1,
                "source": source_text,
                "reference": reference,
                "candidate_translations": translations
            })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Translation completed! Results saved to: {output_file}")

if __name__ == "__main__":
    main()