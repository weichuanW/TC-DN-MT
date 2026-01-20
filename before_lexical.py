import json
import numpy as np
import torch
import evaluate
import nltk
import re
from collections import Counter
import jieba
import os
import sys
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from rouge import Rouge
from typing import List, Dict

# Increase recursion depth limit to handle large data structures
sys.setrecursionlimit(10000)

class BeforeLexicalEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        # Load all evaluators
        self.sacrebleu = evaluate.load("sacrebleu")
        self.meteor = evaluate.load("meteor")
        self.bertscore = evaluate.load("bertscore")
        self.chrf = evaluate.load("chrf")
        self.ter = evaluate.load("ter")
        self.rouge = Rouge()
        
        # Load BLEURT
        self.bleurt_config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12') # return to the original model
        self.bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')
        self.bleurt_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
        self.bleurt_model.to(self.device)
        self.bleurt_model.eval()
    

    def simple_mixed_tokenization(self, text):
        """Simple tokenization method"""
        chinese_pattern = r'[\u4e00-\u9fff]+'
        english_pattern = r'[a-zA-Z]+'
        tokens = []
        for match in re.finditer(chinese_pattern, text):
            tokens.append(match.group())
        for match in re.finditer(english_pattern, text):
            tokens.append(match.group())
        return [token for token in tokens if token.strip()]

    def tokenize_mixed_text(self, text, type="jieba"):
        """Mixed tokenization method"""
        try:
            
            tokens = list(jieba.cut(text))
            processed_tokens = []
            for token in tokens:
                # remove punctuations
                token = re.sub(r'[^\w\s]', '', token)
                if re.search('[a-zA-Z]', token):
                    english_parts = token.split()
                    processed_tokens.extend(english_parts)
                else:
                    # recognize each character
                    # token = list(token)
                    # processed_tokens.extend(token)
                    if type == "jieba":
                        processed_tokens.append(token)
                    if type == "char":
                        processed_tokens.extend(list(token))
            return [token.strip() for token in processed_tokens if token.strip()]
        except ImportError:
            return self.simple_mixed_tokenization(text)
    
    def _build_word_vocabulary(self, texts, type="jieba"):
            """Build word-level vocabulary"""
            word_freq = Counter()
            for text in texts:
                tokens = self.tokenize_mixed_text(text.lower(), type)
                word_freq.update(tokens)
            return {word: freq for word, freq in word_freq.items()}

    '''
    penalty: whether to apply penalty to the frequency score
    penalty_source: the source of the penalty, 10 means the source length of words is 10
    '''
    def _compute_word_frequency(self, texts, vocab, type="jieba", penalty=False, penalty_source=10):
            """Calculate word frequency"""
            overall_count = 0
            for item in vocab.items():
                overall_count += item[1]
            frequency = []
            if overall_count == 0:
                overall_count = 1
            for text in texts:
                tokens = self.tokenize_mixed_text(text.lower(), type)
                # remove the duplicate tokens
                tokens = list(set(tokens))
                text_specific_count = 0
                for token in tokens:
                    if token in vocab:
                        text_specific_count += vocab[token]
                    else:
                        #print('referece is handling')
                        text_specific_count += 0
                #print(texts, vocab)
                frequency_score = text_specific_count / overall_count
                #normalized_frequency_score = frequency_score / tokens_length
                if penalty:
                    frequency_score = frequency_score * min(penalty_source / len(tokens), 1.0)
                else:
                    frequency_score = frequency_score * 1.0
                frequency.append(frequency_score)
            return frequency
    def compute_GLVS_lexical(self, texts: List[str]) -> float:
        """Compute GLVS lexical diversity"""
        vocab = self._build_word_vocabulary(texts, type="jieba")
        frequency = self._compute_word_frequency(texts, vocab, type="jieba")
        frequency = [round(value * 100, 2) for value in frequency]
        return frequency

    def tokenize_text(self, text: str) -> str:
        """Tokenize text"""
        try:
            tokens = list(jieba.cut(text.strip().lower()))
            return ' '.join(tokens)
        except:
            return ' '.join(text.strip().lower().split())
    
    def evaluate_single(self, translation: str, reference: str) -> Dict:
        """Evaluate a single translation"""
        scores = {}
        
        # Handle empty translation
        if not translation or not translation.strip():
            return {
                'BLEU': 0.0, 'METEOR': 0.0, 'BERTScore': 0.0, 'BLEURT': 0.0,
                'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0,
                'chrF++': 0.0, 'TER': 0.0
            }
        
        # BLEU
        bleu_result = self.sacrebleu.compute(predictions=[translation], references=[reference])
        scores['BLEU'] = float(bleu_result["score"])
        
        # METEOR
        trans_tok = self.tokenize_text(translation)
        ref_tok = self.tokenize_text(reference)
        meteor_result = self.meteor.compute(predictions=[trans_tok], references=[ref_tok])
        scores['METEOR'] = float(meteor_result["meteor"])
        
        # BERTScore
        bert_result = self.bertscore.compute(
            predictions=[translation], 
            references=[reference], 
            model_type="bert-base-multilingual-cased" #BERT-Base, Multilingual Cased
        )
        scores['BERTScore'] = float(bert_result["f1"][0])
        
        # BLEURT
        with torch.no_grad():
            inputs = self.bleurt_tokenizer(
                reference, translation,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding='max_length',
                return_overflowing_tokens=False  # Suppress overflow warning
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            bleurt_score = self.bleurt_model(**inputs).logits.item()
            scores['BLEURT'] = float(bleurt_score)
        
        # ROUGE
        if trans_tok and ref_tok:
            rouge_scores = self.rouge.get_scores(trans_tok, ref_tok)[0]
            scores['ROUGE-1'] = float(rouge_scores['rouge-1']['f'])
            scores['ROUGE-2'] = float(rouge_scores['rouge-2']['f'])
            scores['ROUGE-L'] = float(rouge_scores['rouge-l']['f'])
        else:
            scores['ROUGE-1'] = scores['ROUGE-2'] = scores['ROUGE-L'] = 0.0
        
        # chrF++
        chrf_result = self.chrf.compute(
            predictions=[translation],
            references=[[reference]],
            word_order=2
        )
        scores['chrF++'] = float(chrf_result["score"])
        
        # TER
        ter_result = self.ter.compute(
            predictions=[translation],
            references=[[reference]]
        )
        scores['TER'] = float(ter_result["score"])
        
        return scores
    
    def evaluate_batch(self, data: List[Dict]) -> List[Dict]:
        """Batch evaluation"""
        results = []
        from tqdm import tqdm
        # Progress bar only shows on one line, no extra output
        for i, item in tqdm(enumerate(data), total=len(data), desc="Processing", ncols=100):
            source = item['source']
            reference = item['reference']
            candidates = item['candidate_translations']

            GLVS_lexical = self.compute_GLVS_lexical(candidates)

            evaluations = {
                'GLVS': [], 'BLEU': [], 'METEOR': [], 'BERTScore': [], 'BLEURT': [],
                'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': [],
                'chrF++': [], 'TER': []
            }
            # 'GLVS' is a one-time result list for all candidates, directly assign as 1D list to avoid nesting
            evaluations['GLVS'] = GLVS_lexical

            # Evaluate each candidate translation
            for candidate in candidates:
                scores = self.evaluate_single(candidate, reference)
                for metric in evaluations:
                    # 'GLVS' has been calculated and assigned above, do not append in this loop
                    if metric == 'GLVS':
                        continue
                    evaluations[metric].append(scores[metric])
            
            # Compress float precision (keep 4 decimal places to reduce storage space)
            def round_floats(obj, decimals=3):
                """Recursively round floats to specified decimal places"""
                if isinstance(obj, float):
                    return round(obj, decimals)
                elif isinstance(obj, list):
                    return [round_floats(item, decimals) for item in obj]
                elif isinstance(obj, dict):
                    return {key: round_floats(value, decimals) for key, value in obj.items()}
                return obj
            
            # Keep only id and evaluations, remove source, reference, candidates
            result = {
                'id': item.get('id'),
                'evaluations': round_floats(evaluations, decimals=4)
            }
            results.append(result)
        
        return results
    
    def calculate_stats(self, results: List[Dict]):
        """Compute statistics"""
        all_scores = {
            'GLVS': [], 'BLEU': [], 'METEOR': [], 'BERTScore': [], 'BLEURT': [],
            'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': [],
            'chrF++': [], 'TER': []
        }
        
        for result in results:
            evaluations = result['evaluations']
            for metric in all_scores.keys():
                if metric in evaluations:
                    all_scores[metric].extend(evaluations[metric])
        
        print("\n=== Evaluation Statistics ===")
        for metric, scores in all_scores.items():
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"{metric}: mean={mean_score:.4f}, std={std_score:.4f}")
    
    def load_and_evaluate(self, json_file: str, output_file: str):
        """Load JSON file and evaluate"""
        print(f"Loading data: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Start evaluating {len(data)} samples...")
        results = self.evaluate_batch(data)
        
        print(f"Saving results: {output_file}")
        # Save in batches to avoid recursion depth issues
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Use smaller indent or no indent to reduce recursion depth
                json.dump(results, f, ensure_ascii=False, indent=None)
        except RecursionError:
            # If it still fails, try writing item by item
            print("Warning: Using fallback method to save results due to recursion depth")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('[\n')
                for i, result in enumerate(results):
                    json.dump(result, f, ensure_ascii=False)
                    if i < len(results) - 1:
                        f.write(',\n')
                    else:
                        f.write('\n')
                f.write(']')
        
        self.calculate_stats(results)
        return results

# Example usage
def main():
    evaluator = BeforeLexicalEvaluator(device="cuda:0")
    
    # Evaluate a single file
    results = evaluator.load_and_evaluate(
        json_file="input.json", 
        output_file="output.json"
    )

def batch_evaluate_folder(input_folder: str, output_folder: str, device: str = "cuda:0", skip_existing: bool = True):
    """Batch evaluate all JSON files in a folder"""
    import os
    from pathlib import Path
    
    evaluator = BeforeLexicalEvaluator(device=device)
    
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    # Count existing files
    skipped_count = 0
    processed_count = 0
    failed_count = 0
    
    for i, json_file in enumerate(json_files, 1):
        output_filename = f"{json_file.stem}-evaluation_results.json"
        output_path = output_dir / output_filename
        
        # Check if already exists
        if skip_existing and output_path.exists():
            print(f"⊘ Skipping {i}/{len(json_files)}: {json_file.name} (already exists)")
            skipped_count += 1
            continue
        
        print(f"\n→ Processing file {i}/{len(json_files)}: {json_file.name}")
        
        try:
            evaluator.load_and_evaluate(str(json_file), str(output_path))
            print(f"✓ Done: {json_file.name}")
            processed_count += 1
        except RecursionError as e:
            print(f"✗ Failed: {json_file.name} - RecursionError: {e}")
            print(f"  Tip: The file might be too large or have deeply nested structures")
            failed_count += 1
        except Exception as e:
            print(f"✗ Failed: {json_file.name} - {type(e).__name__}: {e}")
            failed_count += 1
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Total: {len(json_files)} files")
    print(f"Processed: {processed_count} files")
    print(f"Skipped: {skipped_count} files")
    print(f"Failed: {failed_count} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Evaluate a single file
    # main()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="../translation_results_check")
    parser.add_argument("--output_folder", type=str, default="../evaluation_results_check_10")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--skip_existing", type=bool, default=True)
    args = parser.parse_args()
    
    
    # Batch evaluate a folder
    batch_evaluate_folder(args.input_folder, args.output_folder, args.device, args.skip_existing)