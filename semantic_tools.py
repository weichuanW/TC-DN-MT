from comet import download_model, load_from_checkpoint
import numpy as np
import os
import json
from laser_encoders import LaserEncoderPipeline
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm, trange
from sklearn.metrics import roc_auc_score
import time

class BeforeSemanticEvaluator:
    def __init__(self):
        pass

    '''
    D: comet kiwi detector for data quality assessment
    I: comet_sample [list(dict)], batch_size [int], gpus [int], scores [dict]
    O: scores [dict[np.array]]
    E: data format: [{'src': str, 'mt': mt}, ...]
    '''
    def cometkiwi_detector(self, comet_sample, batch_size=32, gpus=1, scores=None):
        if not scores:
            scores = dict()
        print("Computing COMETKIWI scores...")
        model_path = download_model("Unbabel/wmt22-cometkiwi-da", saving_directory='[YOUR_MODEL_CACHE_DIR]')
        model = load_from_checkpoint(model_path)

        comet_qe = model.predict(comet_sample, batch_size=batch_size, gpus=gpus)
        # keep 4 digits after the decimal point
        comet_qe.scores = [round(score, 4) for score in comet_qe.scores]
        scores['cometkiwi_score'] = np.array(comet_qe.scores)
        return scores

    '''
    D: comet data detector for data quality assessment
    I: comet_sample [list(dict)], batch_size [int], gpus [int], scores [dict]
    O: scores [dict[np.array]]
    E: data format: [{'src': str, 'ref': ref, 'mt': mt}, ...]
    '''
    def cometdata_detector(self, comet_sample, batch_size=32, gpus=1, scores=None):
        if not scores:
            scores = dict()
        print("Computing COMETDA scores...")
        
        # Check if data is empty
        if not comet_sample or len(comet_sample) == 0:
            print("Warning: Empty comet_sample provided")
            scores['cometda_score'] = np.array([])
            return scores
            
        # Validate data format
        print(f"Processing {len(comet_sample)} samples for COMET-DA")
        for i, sample in enumerate(comet_sample[:3]):  # Check first 3 samples
            if not all(key in sample for key in ['src', 'ref', 'mt']):
                print(f"Warning: Sample {i} missing required keys: {sample.keys()}")
                
        model_path = download_model("Unbabel/wmt22-comet-da", saving_directory='[YOUR_MODEL_CACHE_DIR]')
        model = load_from_checkpoint(model_path)

        try:
            comet_da = model.predict(comet_sample, batch_size=batch_size, gpus=gpus)
            if comet_da is None or comet_da.scores is None:
                print("Warning: COMET prediction returned None")
                scores['cometda_score'] = np.array([])
            else:
                # keep 4 digits after the decimal point
                comet_da.scores = [round(score, 4) for score in comet_da.scores]
                scores['cometda_score'] = np.array(comet_da.scores)
        except Exception as e:
            print(f"Error in COMET prediction: {e}")
            scores['cometda_score'] = np.array([])
        
        return scores


    '''
    D: compute the laser detector for data quality assessment (cosine similarity) 
    - steps: 
    -- 1) extract the sentence embeddings
    -- 2) L2 normalisation
    -- 3) cosine similarity (dot product)
    I: lang_abbr [str], source_sens [list(str)], target_sens [list(str)]
    O: results_score [dict[np.array]]
    '''
    def laser_detector(self, source_lang, target_lang, source_sens, target_sens, scores=None):
        if not scores:
            scores = dict()
        print("Computing LASER scores...")
        encoder_source = LaserEncoderPipeline(lang=source_lang)
        source_embeddings = encoder_source.encode_sentences(source_sens)

        encoder_target = LaserEncoderPipeline(lang=target_lang)
        target_embeddings = encoder_target.encode_sentences(target_sens)

        source_embeddings_l2 = np.linalg.norm(source_embeddings, axis=1)
        target_embeddings_l2 = np.linalg.norm(target_embeddings, axis=1)
        norm_srctgt= source_embeddings_l2 * target_embeddings_l2
        
        # Calculate dot product between each pair of vectors (element-wise multiplication then sum)
        dot_products = np.sum(source_embeddings * target_embeddings, axis=1)       
        # Calculate cosine similarity
        cosine_similarities = dot_products / norm_srctgt        
        scores['laser_score'] = cosine_similarities
        # keep 4 digits after the decimal point
        scores['laser_score'] = [round(score, 4) for score in scores['laser_score']]
       
        return scores
    '''
    D: sentence transformer detector for data quality assessment (cosine similarity)
    I: source_sens [list(str)], target_sens [list(str)]
    O: results_score [dict[np.array]]
    '''
    def sent_transformer_detector(self, source_sens, target_sens, scores=None):
        if not scores:
            scores = dict()
        print("Computing Sentence Transformer (current best, mpnet-base-v2) scores...")
        model = SentenceTransformer("all-mpnet-base-v2", cache_folder="[YOUR_MODEL_CACHE_DIR]")

        source_embeddings = model.encode(source_sens)
        target_embeddings = model.encode(target_sens)

        source_embeddings_l2 = np.linalg.norm(source_embeddings, axis=1)
        target_embeddings_l2 = np.linalg.norm(target_embeddings, axis=1)
        norm_srctgt = source_embeddings_l2 * target_embeddings_l2
        dot_products = np.sum(source_embeddings * target_embeddings, axis=1)       
        # Calculate cosine similarity
        cosine_similarities = dot_products / norm_srctgt        
        scores['sentrans_score'] = cosine_similarities
        # keep 4 digits after the decimal point
        scores['sentrans_score'] = [round(score, 4) for score in scores['sentrans_score']]
        return scores

    def LaBSE_detector(self, source_sens, target_sens, scores=None):
        if not scores:
            scores = dict()
        print("Computing LaBSE scores...")
        model = SentenceTransformer("sentence-transformers/LaBSE", cache_folder="[YOUR_MODEL_CACHE_DIR]")

        source_embeddings = model.encode(source_sens)
        target_embeddings = model.encode(target_sens)

        source_embeddings_l2 = np.linalg.norm(source_embeddings, axis=1)
        target_embeddings_l2 = np.linalg.norm(target_embeddings, axis=1)
        norm_srctgt = source_embeddings_l2 * target_embeddings_l2
        dot_products = np.sum(source_embeddings * target_embeddings, axis=1)       
        # Calculate cosine similarity
        cosine_similarities = dot_products / norm_srctgt        
        scores['labse_score'] = cosine_similarities
        # keep 4 digits after the decimal point
        scores['labse_score'] = [round(score, 4) for score in scores['labse_score']]
        return scores
    '''
    D: compute the entailment score for the data
    I: data_src [list(str)], data_mt [list(str)], scores [dict]
    O: scores [dict[np.array]]
    '''
    def xnli_detector(self, data_src, data_mt, scores=None):
        if not scores:
            scores = dict()
        print("Computing XNLI scores...")
        model_name = "joeddav/xlm-roberta-large-xnli"
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir="[YOUR_MODEL_CACHE_DIR]").cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="[YOUR_MODEL_CACHE_DIR]")
        scores_forward = self.compute_entailment_score(
            data_src, data_mt, model, tokenizer,
        )
        scores_backward = self.compute_entailment_score(
            data_src, data_mt, model, tokenizer
        )
        scores["xnli_score"] = scores_forward * scores_backward
        # keep 4 digits after the decimal point
        scores["xnli_score"] = [round(score, 4) for score in scores["xnli_score"]]
        return scores

    '''
    D: compute the entailment score for the data
    I: data_src [list(str)], data_mt [list(str)], model [transformers.model], tokenizer [transformers.tokenizer], batch_size [int]
    O: scores [np.array]
    '''
    def compute_entailment_score(self, data_src, data_mt, model, tokenizer, batch_size=16):
        scores = []
        for i in trange(0, len(data_src), batch_size):
            batch_src = data_src[i: i + batch_size]
            batch_mt = data_mt[i: i + batch_size]

            with torch.inference_mode():
                inputs = tokenizer(
                    batch_src,
                    batch_mt,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                proba = (
                    torch.softmax(model(**inputs).logits, -1)[
                    :, model.config.label2id["entailment"]
                    ]
                    .cpu()
                    .numpy()
                )
            scores.append(proba)
        # concatenate all scores first, then round to 4 digits
        all_scores = np.concatenate(scores)
        return np.round(all_scores, 4)

    '''
    D: compute the SONAR and BLASER score for the data
    I: src_lang [str], trg_lang [str], src_text [list(str)], trg_text [list(str)], device [str], scores [dict]
    O: scores [dict[np.array]]
    E: src_lang = 'eng_Latn'
    -- tgt_lang = 'zho_Hans'
    '''
    def add_sonar_scores(self, src_lang, trg_lang, src_text, trg_text, device='cuda', scores=None):
        if not scores:
            scores = {}
        print("Computing SONAR scores...")
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        from sonar.models.blaser.loader import load_blaser_model

        # Ensure device is in correct format and a valid device string
        if isinstance(device, list):
            device = device[0] if device else 'cuda'
        elif not isinstance(device, (str, torch.device)):
            device = 'cuda'
        
        # Validate that device parameter is not text content
        if isinstance(device, str) and (len(device) > 10 or any(ord(c) > 127 for c in device)):
            print(f"Warning: Invalid device '{device}', using 'cuda' instead")
            device = 'cuda'
            
        t2vec_model = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        )
        # Ensure device is compatible with torch.device
        device_obj = torch.device(device) if isinstance(device, str) else device
        blaser_qe = load_blaser_model("blaser_2_0_qe").eval().to(device_obj)

        src_embs = t2vec_model.predict(src_text, source_lang=src_lang)
        tgt_embs = t2vec_model.predict(trg_text, source_lang=trg_lang)


        blaser_scores = blaser_qe(src=torch.tensor(src_embs), mt=torch.tensor(tgt_embs))
        scores["blaser_score"] = blaser_scores.detach().cpu().numpy()

        source_embeddings = src_embs.detach().cpu().numpy()
        target_embeddings = tgt_embs.detach().cpu().numpy()
        source_embeddings_l2 = np.linalg.norm(source_embeddings, axis=1)
        target_embeddings_l2 = np.linalg.norm(target_embeddings, axis=1)
        norm_srctgt = source_embeddings_l2 * target_embeddings_l2
        dot_products = np.sum(source_embeddings * target_embeddings, axis=1)       
        # Calculate cosine similarity
        
        cosine_similarities = dot_products / norm_srctgt        
        scores['sonar_score'] = cosine_similarities
        
        # Ensure blaser_score is an array instead of a scalar
        blaser_scores_array = scores['blaser_score'].squeeze()
        if blaser_scores_array.ndim == 0:  # If it's a scalar, convert to an array with one element
            blaser_scores_array = np.array([blaser_scores_array])
        scores['blaser_score'] = blaser_scores_array
        

        # keep 4 digits after the decimal point
        scores['sonar_score'] = [round(score, 4) for score in scores['sonar_score']]
        scores['blaser_score'] = [round(score, 4) for score in scores['blaser_score']]
        return scores

    '''
    D: Convert numpy arrays and numpy scalars to JSON-serializable types
    I: obj [np.array, np.float32, np.float64, np.int32, np.int64, list, dict]
    O: obj [list, dict]
    '''
    def convert_to_json_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return round(float(obj.item()), 4)  # Convert numpy scalar and round to 4 digits
        elif isinstance(obj, float):
            return round(obj, 4)  # Also round regular Python floats
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self.convert_to_json_serializable(v) for k, v in obj.items()}
        else:
            return obj

    def compute_scores(self, src_items, mt_items, ref_items, src_lang, trg_lang, store_prefix_file=None, type='mt'):
        scores = dict()
        kiwi_data = [{'src': src, 'mt': mt} for src, mt in zip(src_items, mt_items)]
        data_data = [{'src': src, 'ref': ref, 'mt': mt} for src, mt, ref in zip(src_items, mt_items, ref_items)]
        # update the scores dictionary with the new scores
        # 'eng_Latn',"zho_Hans"
        # overall saving the scores
        if not store_prefix_file:
            scores.update(self.cometkiwi_detector(kiwi_data))
            scores.update(self.cometdata_detector(data_data))
            scores.update(self.laser_detector(src_lang, trg_lang, src_items, mt_items))
            scores.update(self.LaBSE_detector(src_items, mt_items))  
            scores.update(self.sent_transformer_detector(src_items, mt_items))
            scores.update(self.xnli_detector(src_items, mt_items))
            # Try SONAR, skip if it fails
            try:
                scores.update(self.add_sonar_scores(src_lang, trg_lang, src_items, mt_items))
            except Exception as e:
                print(f"⚠️  Skipping SONAR evaluation (dependency library version incompatible): {str(e)[:80]}")
                # Ensure returning a list of None values instead of deleting keys
                #scores['sonar_score'] = [None] * len(src_items)
                #scores['blaser_score'] = [None] * len(src_items)
        else:
        # separate saving
            
            # Check if kiwi scores file exists
            if not os.path.exists(os.path.join(store_prefix_file, f'{type}_kiwi_scores.json')):
                kiwi_scores = self.cometkiwi_detector(kiwi_data)
                kiwi_scores = self.convert_to_json_serializable(kiwi_scores)
                with open(os.path.join(store_prefix_file, f'{type}_kiwi_scores.json'), 'w') as f:
                    json.dump(kiwi_scores, f, ensure_ascii=False, indent=2)
            else:
                print(f'{type}_kiwi_scores.json already exists')
            
            # Check if data scores file exists
            if not os.path.exists(os.path.join(store_prefix_file, f'{type}_data_scores.json')):
                data_scores = self.cometdata_detector(data_data)
                data_scores = self.convert_to_json_serializable(data_scores)
                with open(os.path.join(store_prefix_file, f'{type}_data_scores.json'), 'w') as f:
                    json.dump(data_scores, f, ensure_ascii=False, indent=2)
            else:
                print(f'{type}_data_scores.json already exists')
            
            # Check if laser scores file exists
            if not os.path.exists(os.path.join(store_prefix_file, f'{type}_laser_scores.json')):
                laser_scores = self.laser_detector(src_lang, trg_lang, src_items, mt_items)
                laser_scores = self.convert_to_json_serializable(laser_scores)
                with open(os.path.join(store_prefix_file, f'{type}_laser_scores.json'), 'w') as f:
                    json.dump(laser_scores, f, ensure_ascii=False, indent=2)
            else:
                print(f'{type}_laser_scores.json already exists')
            
            # Check if labse scores file exists
            if not os.path.exists(os.path.join(store_prefix_file, f'{type}_labse_scores.json')):
                labse_scores = self.LaBSE_detector(src_items, mt_items)
                labse_scores = self.convert_to_json_serializable(labse_scores)
                with open(os.path.join(store_prefix_file, f'{type}_labse_scores.json'), 'w') as f:
                    json.dump(labse_scores, f, ensure_ascii=False, indent=2)
            else:
                print(f'{type}_labse_scores.json already exists')
            # Check if sentrans scores file exists
            if not os.path.exists(os.path.join(store_prefix_file, f'{type}_sentrans_scores.json')):
                sentrans_scores = self.sent_transformer_detector(src_items, mt_items)
                sentrans_scores = self.convert_to_json_serializable(sentrans_scores)
                with open(os.path.join(store_prefix_file, f'{type}_sentrans_scores.json'), 'w') as f:
                    json.dump(sentrans_scores, f, ensure_ascii=False, indent=2)
            else:
                print(f'{type}_sentrans_scores.json already exists')
            # Check if xnli scores file exists
            if not os.path.exists(os.path.join(store_prefix_file, f'{type}_xnli_scores.json')):
                xnli_scores = self.xnli_detector(src_items, mt_items)
                xnli_scores = self.convert_to_json_serializable(xnli_scores)
                with open(os.path.join(store_prefix_file, f'{type}_xnli_scores.json'), 'w') as f:
                    json.dump(xnli_scores, f, ensure_ascii=False, indent=2)
            else:
                print(f'{type}_xnli_scores.json already exists')
            # Check if sonar scores file exists
            if not os.path.exists(os.path.join(store_prefix_file, f'{type}_sonar_scores.json')):
                try:
                    sonar_scores = self.add_sonar_scores(src_lang, trg_lang, src_items, mt_items)
                    sonar_scores = self.convert_to_json_serializable(sonar_scores)
                    with open(os.path.join(store_prefix_file, f'{type}_sonar_scores.json'), 'w') as f:
                        json.dump(sonar_scores, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"⚠️  Skipping SONAR evaluation (dependency library version incompatible): {str(e)[:80]}")
                    sonar_scores = {
                        'sonar_score': [None] * len(src_items),
                        'blaser_score': [None] * len(src_items)
                    }
                    with open(os.path.join(store_prefix_file, f'{type}_sonar_scores.json'), 'w') as f:
                        json.dump(sonar_scores, f, ensure_ascii=False, indent=2)
            else:
                print(f'{type}_sonar_scores.json already exists')
        return scores

    '''
    D: serve for testing only   
    '''
    def test_cases(self):
        # testing all methods in the external_detectors class
        ext = external_detectors()
        # compute the scores for these two pairs
        
        sample_src = ["This is a test sentence.", "This is a test sentence."]
        sample_ref = ["这是测试句子", "这是我的老家"]
        comet_sample_kiwi = [{'src': 'This is a test sentence.', 'mt': '这是测试句子'}, {'src': 'This is a test sentence.', 'mt': '这是我的老家'}]
        comet_sample_data = [{'src': 'This is a test sentence.', 'ref': '这是测试句子', 'mt': '这是测试用句'}, {'src': 'This is a test sentence.', 'ref': '这是测试句子', 'mt': '这是我的老家'}]
        scores_kiwi = ext.cometkiwi_detector(comet_sample_kiwi)
        
        scores_data = ext.cometdata_detector(comet_sample_data)
        
        scores_laser = ext.laser_detector('eng_Latn',"zho_Hans", sample_src, sample_ref)

        scores_labse = ext.LaBSE_detector(sample_src, sample_ref)

        scores_sentrans = ext.sent_transformer_detector(sample_src, sample_ref)

        scores_xnli = ext.xnli_detector(sample_src, sample_ref)

        scores_sonar = ext.add_sonar_scores('eng_Latn',"zho_Hans", sample_src, sample_ref)

        print('cometkiwi_score:', scores_kiwi['cometkiwi_score'])
        print('cometda_score:', scores_data['cometda_score'])
        print('sonar_score:', scores_sonar['sonar_score'])
        print('blaser_score:', scores_sonar['blaser_score'])
        print('laser_score:', scores_laser['laser_score'])
        print('labse_score:', scores_labse['labse_score'])
        print('sentrans_score:', scores_sentrans['sentrans_score'])
        print('xnli_score:', scores_xnli['xnli_score'])