"""Sample answers from LLMs on QA task."""
import gc
import os
import logging
import sys
import random
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import math
import torch.nn.functional as F
import wandb
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline


from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from sklearn.metrics import auc
import json
from uncertainty.utils import gradient_utils
import nltk

def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

def get_target_mask(tokenizer, prompt, target_answer, ner_pipeline, device='cuda'):
    """
    Identifies the key phrase (entity) in text using NER and returns a binary mask 
    aligned with tokenizer tokens.
    """
    # 1. Extract Key Phrase using NER
    key_entity_text = target_answer
    try:
        if ner_pipeline:
            # Standard pipeline output with aggregation_strategy="simple"
            entities = ner_pipeline(target_answer)
            if entities:
                # Pick the first entity as the "Subject" or key signal
                key_entity_text = entities[0]['word']
    except Exception as e:
        pass

    # 2. Create Mask by aligning
    # Ensure space handling matches gradient_utils (which joins prompt + " " + target)
    target_answer_spaced = target_answer
    if not prompt.endswith(' ') and not target_answer.startswith(' '):
         target_answer_spaced = ' ' + target_answer

    full_text = prompt + target_answer_spaced
    
    # Tokenize full prompt+answer
    full_tokens_tensor = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids[0].to(device)
    prompt_tokens_tensor = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0].to(device)
    
    # Robust BOS Check for Split
    if tokenizer.bos_token_id is not None:
        prompt_has_bos = (prompt_tokens_tensor[0] == tokenizer.bos_token_id)
        full_has_bos = (full_tokens_tensor[0] == tokenizer.bos_token_id)
        
        # Adjust prompt length relative to full_tokens
        offset = int(full_has_bos) - int(prompt_has_bos)
        prompt_len = len(prompt_tokens_tensor) + offset
    else:
        prompt_len = len(prompt_tokens_tensor)

    target_tokens = full_tokens_tensor[prompt_len:]
    
    mask = torch.zeros(len(target_tokens), device=device)
    
    # 3. Find key entity tokens within target_tokens
    # We strip spaces from key_entity_text for tokenization to avoid leading space issues if not needed
    entity_tokens = tokenizer(key_entity_text.strip(), add_special_tokens=False).input_ids
    entity_tokens_tensor = torch.tensor(entity_tokens, device=device)
    
    # Sliding window search
    target_len = len(target_tokens)
    entity_len = len(entity_tokens)
    found = False
    
    if entity_len > 0 and target_len >= entity_len:
        for i in range(target_len - entity_len + 1):
            if torch.equal(target_tokens[i : i+entity_len], entity_tokens_tensor):
                mask[i : i+entity_len] = 1
                found = True
                break
    
    if not found:
        # Fallback to full mask
        mask[:] = 1
              
    return mask

# Load Perturbations
with open('config/perturbations.json', 'r') as f:
    PERTURBATION_CONFIG = json.load(f)


utils.setup_logger()


def init_paraphrase_model(device='cuda'):
    """Initialize the Pegasus paraphrase model."""
    try:
        logging.info("Initializing Pegasus paraphrase model...")
        logging.info("Downloading safetensors version (this may take a few minutes on first run)...")
        model_name = "tuner007/pegasus_paraphrase"
        
        # Directly download and use safetensors version
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            use_safetensors=True  # Force safetensors format
        ).to(device)
        
        model.eval()
        logging.info("Pegasus paraphrase model initialized successfully.")
        return tokenizer, model
    except Exception as e:
        logging.error(f"Failed to initialize paraphrase model: {e}")
        return None, None


def get_paraphrase(text, paraphrase_tokenizer, paraphrase_model, device='cuda', num_return_sequences=1):
    """Generate paraphrase using Pegasus model."""
    try:
        # Tokenize input
        inputs = paraphrase_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=60
        ).to(device)
        
        # Generate paraphrase
        with torch.no_grad():
            outputs = paraphrase_model.generate(
                **inputs,
                max_length=60,
                num_beams=10,
                num_return_sequences=num_return_sequences,
                temperature=1.5,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
        
        # Decode output
        paraphrases = [
            paraphrase_tokenizer.decode(output, skip_special_tokens=True) 
            for output in outputs
        ]
        
        return paraphrases[0] if paraphrases else text
    except Exception as e:
        logging.warning(f"Failed to paraphrase text: {e}. Returning original text.")
        return text


                    
def predict_with_embeddings(model, prompt, temperature=1, max_new_tokens=20):
    inputs = model.tokenizer(prompt, return_tensors="pt").to('cuda')
    input_ids = inputs.input_ids

    with torch.no_grad():
        generated_outputs = model.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            output_hidden_states=True, 
            output_scores=True, 
            return_dict_in_generate=True,
            do_sample=True if temperature > 0 else False,
            pad_token_id=model.tokenizer.pad_token_id,
            top_p=0.9 if temperature > 0 else 1.0,
            top_k=50 if temperature > 0 else 1,
        )
        
        generated_ids = generated_outputs.sequences[0]
        
        new_token_ids = generated_ids[len(input_ids[0]):]
        
        predicted_answer = model.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        if '\n' in predicted_answer:
            predicted_answer = predicted_answer.split('\n')[0]
        
        token_log_likelihoods = []
        if hasattr(generated_outputs, 'scores') and generated_outputs.scores is not None:
            for i, score in enumerate(generated_outputs.scores):
                logits = score / temperature
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_likelihoods.append(log_probs[0, new_token_ids[i]].item())
        else:
            logging.warning("Warning: No scores available in generation output")
            token_log_likelihoods = [0.0] * len(new_token_ids)
        
        embeddings_per_layer = []
        if hasattr(generated_outputs, 'hidden_states') and generated_outputs.hidden_states is not None:
            last_step_hidden_states = generated_outputs.hidden_states[-1]
            for layer_idx in range(len(last_step_hidden_states)):

                embedding = last_step_hidden_states[layer_idx][0, -1, :].cpu().numpy()
                embeddings_per_layer.append(embedding)
            l_embedding = last_step_hidden_states[-1][0, -1, :].cpu().unsqueeze(0)
        else:
            logging.warning("Warning: No hidden states available in generation output")
            with torch.no_grad():
                outputs = model.model(**inputs, output_hidden_states=True)
                for layer_idx in range(len(outputs.hidden_states)):
                    embedding = outputs.hidden_states[layer_idx][0, -1, :].cpu().numpy()
                    embeddings_per_layer.append(embedding)
                l_embedding = outputs.hidden_states[-1][0, -1, :].cpu()
    
    return predicted_answer, token_log_likelihoods, embeddings_per_layer, l_embedding
    
def calculate_roc_auc(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    
    tprs = [0]
    fprs = [0]
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    tprs.append(1)
    fprs.append(1)
    
    roc_auc = auc(fprs, tprs)
    
    return roc_auc, fprs, tprs


    
def main(args):
    # Start timing and reset peak memory tracker
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Setup run.
    logging.info("Running gradient-based curvature analysis.")

    if args.dataset == 'svamp':
        if not args.use_context:
            logging.info('Forcing `use_context=True` for svamp dataset.')
            args.use_context = True
    elif args.dataset == 'squad':
        if not args.answerable_only:
            logging.info('Forcing `answerable_only=True` for squad dataset.')
            args.answerable_only = True

    experiment_details = {'args': args}
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    user = os.environ['USER']
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    wandb.init(
        entity=args.entity,
        project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
        mode="disabled",
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
        settings=wandb.Settings(init_timeout=300),
    )
    logging.info('Finished wandb init.')

    # Get accuracy metric.
    metric = utils.get_metric(args.metric)

    # Load dataset.
    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)
    if args.ood_train_dataset is not None:
        logging.warning(
            'Using OOD dataset %s to construct few-shot prompts and train p_ik.',
            args.ood_train_dataset)
        # Get indices of answerable and unanswerable questions and construct prompt.
        train_dataset, _ = load_ds(args.ood_train_dataset, add_options=args.use_mc_options)
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Get indices of answerable and unanswerable questions and construct prompt.
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    # Initialize model.
    token = "" # replace with your token if using models that require authentication
    model = utils.init_model(args, token=token)
    
    # Initialize NER Pipeline
    ner_pipeline = None
    if args.use_key_phrase_masking:
        try:
            logging.info("Initializing NER pipeline for entity extraction...")
            # Use aggregation_strategy="simple" to get word-level entities
            ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
            logging.info("NER pipeline initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize NER pipeline: {e}. Will fallback to full mask.")
    
    # Initialize Paraphrase Model
    paraphrase_tokenizer, paraphrase_model = None, None
    if args.use_paraphrase_perturbation:
        paraphrase_tokenizer, paraphrase_model = init_paraphrase_model(device='cuda')
        if paraphrase_tokenizer is None or paraphrase_model is None:
            logging.warning("Paraphrase model failed to initialize. Falling back to non-paraphrase mode.")
            args.use_paraphrase_perturbation = False



    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
     # Select perturbations consistently for the whole run
    active_perturbations = []
    if args.use_mc_dropout_perturbation:
        # MC Dropout mode: we'll run num_perturbations forward passes with dropout enabled
        active_perturbations = list(range(args.num_perturbations))
        logging.info("Selected %d MC Dropout perturbations with dropout_rate=%.2f",
                    args.num_perturbations, args.mc_dropout_rate)
        logging.info("Using MC DROPOUT mode: Weights will be stochastically perturbed via dropout.")
    elif args.use_embedding_noise_perturbation:
        # Embedding noise mode: we don't need textual perturbations
        # We'll generate num_perturbations noise samples instead
        active_perturbations = list(range(args.num_perturbations))
        logging.info("Selected %d embedding noise perturbations with epsilon=%.4f", 
                    args.num_perturbations, args.embedding_noise_epsilon)
        logging.info("Using EMBEDDING NOISE mode: Gaussian noise will be injected into input embeddings.")
    elif args.use_paraphrase_perturbation:
        # Use paraphrase prompts - these will be used to generate paraphrased questions
        available_perturbations = PERTURBATION_CONFIG.get('paraphrase_prompts', [])
        if not available_perturbations:
            logging.warning("No paraphrase_prompts found in config. Falling back to sentence_prompts.")
            available_perturbations = PERTURBATION_CONFIG.get('sentence_prompts', [])
        active_perturbations = random.sample(
            available_perturbations, 
            min(args.num_perturbations, len(available_perturbations))
        )
        logging.info("Selected perturbations for run: %s", active_perturbations)
        logging.info("Using PARAPHRASE mode: Questions will be rephrased using LLM.")
    elif args.enable_brief:
        available_perturbations = PERTURBATION_CONFIG.get('brief_prompts', [])
        active_perturbations = random.sample(
            available_perturbations, 
            min(args.num_perturbations, len(available_perturbations))
        )
        logging.info("Selected perturbations for run: %s", active_perturbations)
    else:
        available_perturbations = PERTURBATION_CONFIG.get('sentence_prompts', [])
        active_perturbations = random.sample(
            available_perturbations, 
            min(args.num_perturbations, len(available_perturbations))
        )
        logging.info("Selected perturbations for run: %s", active_perturbations)

    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # This will store all input data and model predictions.
        accuracies, generations = [], {}
        all_cosine_similarities = []  # Store cosine similarities for all examples

        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))

        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # Evaluate over random subset of the datasets.
        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
        experiment_details[dataset_split] = {'indices': indices}

        if args.num_samples > len(dataset):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

        it = 0
        y_error=[]
        curvature_scores = []
        
        for index in tqdm(indices):
            # More aggressive memory management
            if it > 0:
                # Clear gradients from previous iteration
                if hasattr(model, 'model'):
                    model.model.zero_grad(set_to_none=True)
                
                # More frequent garbage collection
                if it % 5 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Force synchronization to ensure all CUDA operations complete
                    torch.cuda.synchronize()
            it += 1

            # Grab example at index.
            example = dataset[index]
            question, context = example["question"], example['context']
            generations[example['id']] = {'question': question, 'context': context}
            correct_answer = example['answers']['text']

            current_input = make_prompt(
                context, question, None, BRIEF, args.brief_always and args.enable_brief)
            local_prompt = prompt + current_input

            logging.info('Current input: '.ljust(15) + current_input)

            full_responses = []
            embeddings = []  # Store embeddings for all responses

            # We sample one low temperature answer on which we will compute the
            # accuracy and args.num_generation high temperature answers which will
            # be used to estimate the entropy variants.

            if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = args.num_generations + 1

            # === Refactored Main Logic for Gradient-Based Uncertainty ===
            
            # 1. Generate Anchor (One Generation, Low Temp/Greedy-ish)
            # Using i=0 logic for anchor generation
            temperature = args.temperature # Low temp for anchor
            
            # Predict Anchor
            predicted_answer, token_log_likelihoods, embedding, l_embedding = predict_with_embeddings(
                    model, local_prompt, temperature)
            
            # Compute Accuracy for Anchor
            if correct_answer:
                acc = metric(predicted_answer, example, model)
            else:
                acc = 0
            
            y_error.append(1-acc)
            accuracies.append(acc)

            # Store Anchor Info
            most_likely_answer_dict = {
                'response': predicted_answer,
                'token_log_likelihoods': token_log_likelihoods,
                'embedding': l_embedding,
                'accuracy': acc}
            
            generations[example['id']].update({
                'most_likely_answer': most_likely_answer_dict,
                'reference': utils.get_reference(example)})
            
            logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
            logging.info('question: '.ljust(15) + question)
            logging.info('Anchor prediction: '.ljust(15) + predicted_answer)
            logging.info('correct answer: '.ljust(15) + str(correct_answer))
            logging.info('Accuracy: '.ljust(15) + str(acc))
            sys.stdout.flush()



            # 2. Gradient Calculation (Curvature Awareness)
            if True:
                try:
                    # Phase 1: The Anchor
                    # We already generated predicted_answer above.
                    # We need to get the exact target_ids corresponding to this answer.
                    
                    # Re-tokenize prompt to get length
                    clean_input_tokens = model.tokenizer(local_prompt, return_tensors="pt", add_special_tokens=True).to('cuda')
                    clean_input_ids = clean_input_tokens.input_ids
                    
                    # To be perfectly safe and match the generation exact tokens:
                    # We re-run a quick generate or we can trust predict_with_embeddings if it returned full ids?
                    # predict_with_embeddings returns string.
                    # Let's re-tokenize the predicted answer to get target_ids. 
                    # NOTE: This might have slight mismatch if naive tokenization differs from generation context, 
                    # but `get_gradients_fixed` stitches them manually so it checks out.
                    
                    # BETTER: Use the `new_token_ids` if you modified predict_with_embeddings to return them? 
                    # The current predict_with_embeddings (lines 121-174) doesn't return raw ids easily visible here.
                    # It returns `predicted_answer`.
                    
                    # To strictly follow the user's snippet:
                    # "generated_ids = model.generate(clean_input_ids, ...); target_ids = generated_ids[:, clean_input_ids.shape[1]:]"
                    
                    # We already did generation in predict_with_embeddings. 
                    # Let's just tokenize the answer. 
                    target_ids = model.tokenizer(predicted_answer, return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')
                    
                    # Compute mask if key phrase masking is enabled
                    target_mask = None
                    if args.use_key_phrase_masking:
                        try:
                            target_mask = get_target_mask(
                                model.tokenizer, local_prompt, predicted_answer, 
                                ner_pipeline, device='cuda'
                            )
                            logging.info(f"Key phrase mask computed: {target_mask}")
                            logging.info(f"Tokens contributing to gradient: {target_mask.sum().item()}/{len(target_mask)}")
                        except Exception as e:
                            logging.warning(f"Failed to compute mask: {e}. Using full mask.")
                            target_mask = None
                    
                    # Phase 2: Compute Anchor Gradient
                    grad_clean = gradient_utils.get_gradients_fixed(
                        model.model, model.tokenizer, local_prompt, target_ids, 
                        layer_type=args.gradient_target, device='cuda', target_mask=target_mask
                    )
                    
                    if grad_clean is not None:
                        perturbation_scores = []
                    
                        # Phase 3: Perturbation Loop
                        for pert_idx, pert_prefix in enumerate(active_perturbations):
                            
                            # Check perturbation mode
                            if args.use_mc_dropout_perturbation:
                                # MC Dropout mode: enable dropout for weight perturbation
                                logging.info(f"MC Dropout perturbation #{pert_idx + 1} with dropout_rate={args.mc_dropout_rate}")
                                
                                # Use MC Dropout gradient function (model in train mode)
                                grad_pert = gradient_utils.get_gradients_with_mc_dropout(
                                    model.model, model.tokenizer, local_prompt, target_ids,
                                    layer_type=args.gradient_target, device='cuda', target_mask=target_mask,
                                    dropout_rate=args.mc_dropout_rate
                                )
                            elif args.use_embedding_noise_perturbation:
                                # Embedding Noise mode: inject Gaussian noise into embeddings
                                logging.info(f"Embedding noise perturbation #{pert_idx + 1} with epsilon={args.embedding_noise_epsilon}")
                                
                                # Use the new embedding noise gradient function
                                grad_pert = gradient_utils.get_gradients_with_embedding_noise(
                                    model.model, model.tokenizer, local_prompt, target_ids,
                                    epsilon=args.embedding_noise_epsilon,
                                    layer_type=args.gradient_target, device='cuda', target_mask=target_mask
                                )
                            elif args.use_paraphrase_perturbation:
                                # Paraphrase mode: Use Pegasus to generate paraphrased question
                                perturbed_question = get_paraphrase(
                                    question, 
                                    paraphrase_tokenizer, 
                                    paraphrase_model, 
                                    device='cuda'
                                )
                                
                                logging.info(f"Original question: {question}")
                                logging.info(f"Paraphrased question: {perturbed_question}")
                                
                                # Use the paraphrased question with the original make_prompt format
                                current_input_perturbed = make_prompt(
                                    context, perturbed_question, None, BRIEF, args.brief_always and args.enable_brief)
                                perturbed_input = prompt + current_input_perturbed
                                
                                logging.info(f"Perturbation used: {pert_prefix}")
                                
                                # Pass the SAME target_ids AND target_mask
                                grad_pert = gradient_utils.get_gradients_fixed(
                                    model.model, model.tokenizer, perturbed_input, target_ids, 
                                    layer_type=args.gradient_target, device='cuda', target_mask=target_mask
                                )
                            elif isinstance(pert_prefix, str) and "${question}" in pert_prefix:
                                # Template mode with ${question} placeholder
                                perturbed_question = pert_prefix.replace("${question}", question)
                                current_input_perturbed = make_prompt(
                                    context, perturbed_question, None, BRIEF, args.brief_always and args.enable_brief)
                                perturbed_input = prompt + current_input_perturbed
                                
                                logging.info(f"Perturbation used: {pert_prefix}")
                                
                                grad_pert = gradient_utils.get_gradients_fixed(
                                    model.model, model.tokenizer, perturbed_input, target_ids, 
                                    layer_type=args.gradient_target, device='cuda', target_mask=target_mask
                                )
                            else:
                                # Prefix mode: add prompt before question
                                perturbed_question = str(pert_prefix) + " " + question
                                current_input_perturbed = make_prompt(
                                    context, perturbed_question, None, BRIEF, args.brief_always and args.enable_brief)
                                perturbed_input = prompt + current_input_perturbed
                                
                                logging.info(f"Perturbation used: {pert_prefix}")
                                
                                grad_pert = gradient_utils.get_gradients_fixed(
                                    model.model, model.tokenizer, perturbed_input, target_ids, 
                                    layer_type=args.gradient_target, device='cuda', target_mask=target_mask
                                )
                        
                            # Score 
                            if grad_pert is not None:
                                score = gradient_utils.calculate_hallucination_score(grad_clean, grad_pert)
                                perturbation_scores.append(score)
                                logging.info(f"Score for this perturbation: {score}")

                        # Final Score (Minimization)
                        final_curvature_score = np.min(perturbation_scores) if perturbation_scores else 100.0
                        curvature_scores.append(final_curvature_score)
                        logging.info(f"Curvature score: {final_curvature_score}")
                    else:
                        logging.warning("Gradient computation failed (None) for Anchor.")
                        curvature_scores.append(0.0)

                except Exception as e:
                    logging.error(f"Error computing curvature: {e}", exc_info=True)
                    curvature_scores.append(0.0)

        if len(curvature_scores) == len(y_error):
             auc_curvature, fprs, tprs = calculate_roc_auc(np.array(y_error), np.array(curvature_scores))
             logging.info(f"Detection Performance (AUROC) Curvature: {auc_curvature}")
             wandb.log({"auc_curvature": auc_curvature})
             print("Detection Performance (AUROC) Curvature: ", auc_curvature)
        else:
             logging.warning("Mismatch in scores and labels length, skipping AUROC.")
             print("Mismatch in scores and labels length, skipping AUROC.")
        

        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')

        # Log overall accuracy and cosine similarity statistics
        accuracy = np.mean(accuracies)
        logging.info(f"Overall {dataset_split} split accuracy: {accuracy}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})
        
        if all_cosine_similarities:
            avg_cosine_sim = np.mean(all_cosine_similarities)
            std_cosine_sim = np.std(all_cosine_similarities)
            wandb.log({
                f"{dataset_split}_avg_cosine_similarity": avg_cosine_sim,
                f"{dataset_split}_std_cosine_similarity": std_cosine_sim
            })
            logging.info(f"Average cosine similarity for {dataset_split}: {avg_cosine_sim}")
            logging.info(f"Std cosine similarity for {dataset_split}: {std_cosine_sim}")


    utils.save(experiment_details, 'experiment_details.pkl')
    
    # Log timing and peak VRAM usage
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if torch.cuda.is_available():
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_gb = peak_vram_bytes / (1024 ** 3)
        logging.info(f"Peak GPU VRAM usage: {peak_vram_gb:.2f} GB ({peak_vram_bytes:,} bytes)")
        wandb.log({"peak_vram_gb": peak_vram_gb, "peak_vram_bytes": peak_vram_bytes})
        print(f"Peak GPU VRAM usage: {peak_vram_gb:.2f} GB")
    
    logging.info(f"Total run time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    wandb.log({"run_time_seconds": elapsed_time, "run_time_minutes": elapsed_time/60})
    print(f"Total run time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    logging.info('Run complete.')
    del model


if __name__ == '__main__':

    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')

    wandb.finish()
    sys.exit(0)