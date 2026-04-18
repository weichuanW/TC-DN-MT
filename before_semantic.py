import json
import numpy as np
from semantic_tools import BeforeSemanticEvaluator
import os
from pathlib import Path
import argparse

# FLORES-style codes for LASER / SONAR; match your generation direction.
LANG_PAIR_PRESETS = {
    "en-zh": ("eng_Latn", "zho_Hans"),
    "zh-en": ("zho_Hans", "eng_Latn"),
    "en-de": ("eng_Latn", "deu_Latn"),
    "de-en": ("deu_Latn", "eng_Latn"),
    "en-ru": ("eng_Latn", "rus_Cyrl"),
    "ru-en": ("rus_Cyrl", "eng_Latn"),
}


class JSONEvaluationProcessor:
    def __init__(self, src_lang: str = "eng_Latn", trg_lang: str = "zho_Hans"):
        self.evaluator = BeforeSemanticEvaluator()
        self.src_lang = src_lang
        self.trg_lang = trg_lang
    
    def load_json_data(self, json_file_path):
        """Load JSON input data"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def prepare_evaluation_data(self, json_data):
        """Prepare evaluation data"""
        src_items = []
        ref_items = []
        mt_items = []
        item_ids = []
        
        for item in json_data:
            candidates = item["candidate_translations"]
            for j, candidate in enumerate(candidates):
                src_items.append(item["source"])
                ref_items.append(item["reference"])
                mt_items.append(candidate)
                item_ids.append(item["id"])   
            '''if len(candidates) > candidate_index:
                src_items.append(item["source"])
                ref_items.append(item["reference"])
                mt_items.append(candidates[candidate_index])
                item_ids.append(item["id"])'''
        
        return src_items, ref_items, mt_items, item_ids
    
    def run_evaluations(self, src_items, ref_items, mt_items):
        """Run all evaluation metrics"""
        scores = self.evaluator.compute_scores(
            src_items=src_items,
            mt_items=mt_items,
            ref_items=ref_items,
            src_lang=self.src_lang,
            trg_lang=self.trg_lang,
        )
        return scores
    
    def format_detailed_results(self, json_data, scores, item_ids, step=10):
        """Format results: keep only id and evaluation metrics, compress float precision to reduce file size"""
        results = []

        def round_floats(obj, decimals=4):
            """Recursively round floats to specified decimal places"""
            if isinstance(obj, float):
                return round(obj, decimals)
            elif isinstance(obj, list):
                return [round_floats(v, decimals) for v in obj]
            elif isinstance(obj, dict):
                return {k: round_floats(v, decimals) for k, v in obj.items()}
            return obj

        for i, item_id in enumerate(item_ids):
            evaluations = {
                'COMETKIWI': [scores['cometkiwi_score'][j] if j < len(scores['cometkiwi_score']) else None for j in range(i*step, (i+1)*step)],
                'COMETDA':   [scores['cometda_score'][j]   if j < len(scores['cometda_score'])   else None for j in range(i*step, (i+1)*step)],
                'LASER':     [scores['laser_score'][j]     if j < len(scores['laser_score'])     else None for j in range(i*step, (i+1)*step)],
                'LaBSE':     [scores['labse_score'][j]     if j < len(scores['labse_score'])     else None for j in range(i*step, (i+1)*step)],
                'SentTrans': [scores['sentrans_score'][j]  if j < len(scores['sentrans_score'])  else None for j in range(i*step, (i+1)*step)],
                'XNLI':      [scores['xnli_score'][j]      if j < len(scores['xnli_score'])      else None for j in range(i*step, (i+1)*step)],
                'SONAR':     [scores['sonar_score'][j]     if (j < len(scores.get('sonar_score', [])) and scores['sonar_score'][j] is not None) else None for j in range(i*step, (i+1)*step)],
                'BLASER':    [scores['blaser_score'][j]    if (j < len(scores.get('blaser_score', [])) and scores['blaser_score'][j] is not None) else None for j in range(i*step, (i+1)*step)],
            }

            result = {
                'id': item_id,
                'evaluations': round_floats(evaluations, decimals=4)
            }
            results.append(result)

        return results
    
    def save_detailed_results(self, input_json_path, output_json_path,  step=10):
        """Complete workflow: test existence->load data -> prepare evaluation data -> run evaluations -> format detailed results -> save detailed results"""
        
        # test existence of the output json file
        if os.path.exists(output_json_path):
            print(f"Output file already exists: {output_json_path}")
            return
        
        print("Loading JSON data...")
        json_data = self.load_json_data(input_json_path)
        
        print("Preparing evaluation data...")
        src_items, ref_items, mt_items, item_ids = self.prepare_evaluation_data(json_data)
        
        print(f"Running evaluations for {len(src_items)} items...")
        

        scores = self.run_evaluations(src_items, ref_items, mt_items)
        
        print("Formatting detailed results...")
        detailed_results = self.format_detailed_results(json_data, scores, item_ids, step=step)
        
        # Save results
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            # Use compact format to reduce file size
            json.dump(detailed_results, f, ensure_ascii=False, separators=(',', ':'))
        
        print(f"Detailed results saved to: {output_json_path}")
        print(f"Processed {len(detailed_results)} items successfully!")
        
        return detailed_results

# Usage
def process_single_file(processor: JSONEvaluationProcessor, input_file: str, output_file: str, step: int = 10):
    """Process a single JSON file and save results"""
    try:
        results = processor.save_detailed_results(input_file, output_file, step=step)
        print(f"\n=== Evaluation completed: {input_file} ===")
        print(f"Total processed: {len(results)} items")
        if results:
            first_result = results[0]
            print(f"Example ID: {first_result['id']}")
    except Exception as e:
        raise e
        print(f"Error during evaluation: {input_file} - {e}")


def process_path(
    input_path: str,
    output_dir: str,
    step: int = 10,
    src_lang: str = "eng_Latn",
    trg_lang: str = "zho_Hans",
):
    """If input_path is a file, process single file; if it's a directory, batch process all .json files in the directory"""
    processor = JSONEvaluationProcessor(src_lang=src_lang, trg_lang=trg_lang)
    in_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        out_name = f"{in_path.stem}-semantic_results.json"
        process_single_file(processor, str(in_path), str(out_dir / out_name), step=step)
        return

    if in_path.is_dir():
        json_files = sorted(in_path.glob("*.json"))
        print(f"Found {len(json_files)} JSON files")
        for i, jf in enumerate(json_files, 1):
            print(f"[{i}/{len(json_files)}] Processing: {jf.name}")
            out_name = f"{jf.stem}-semantic_results.json"
            process_single_file(processor, str(jf), str(out_dir / out_name), step=step)
        return

    print(f"Input path does not exist: {input_path}")


def main():

    parser = argparse.ArgumentParser(description="Semantic evaluation: supports single file or batch processing of directories")
    parser.add_argument("--input", default="./translation_results")
    parser.add_argument("--output", default="./evaluation_results_semantic")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument(
        "--src_lang",
        default="eng_Latn",
        help="Source language (FLORES code), e.g. eng_Latn, deu_Latn",
    )
    parser.add_argument(
        "--trg_lang",
        default="zho_Hans",
        help="Target language (FLORES code), e.g. zho_Hans, eng_Latn",
    )
    parser.add_argument(
        "--lang_pair",
        default=None,
        choices=list(LANG_PAIR_PRESETS.keys()),
        help="Optional shortcut: sets --src_lang/--trg_lang (e.g. en-de)",
    )
    args = parser.parse_args()

    src_lang, trg_lang = args.src_lang, args.trg_lang
    if args.lang_pair:
        src_lang, trg_lang = LANG_PAIR_PRESETS[args.lang_pair]

    process_path(args.input, args.output, step=args.step, src_lang=src_lang, trg_lang=trg_lang)

if __name__ == "__main__":
    main()
    print("Evaluation completed!")