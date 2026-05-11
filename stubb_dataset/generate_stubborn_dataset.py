"""Generate stubborn hallucination dataset."""
import os
import sys
import logging
import random
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to import uncertainty modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils

def main():
    parser = argparse.ArgumentParser(description="Generate stubborn hallucination dataset.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Model name")
    parser.add_argument("--dataset", type=str, default="trivia_qa", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of generations to check for consistency")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--output_file", type=str, default="stubborn_dataset.json", help="Output file path")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (None for all)")
    parser.add_argument("--model_max_new_tokens", type=int, default=50, help="Max new tokens")
    parser.add_argument("--use_context", action="store_true", default=False, help="Use context in prompt")
    parser.add_argument("--enable_brief", action="store_true", default=True, help="Use brief prompt")
    parser.add_argument("--brief_always", action="store_true", default=False, help="Always use brief prompt")
    parser.add_argument("--brief_prompt", type=str, default="default", help="Brief prompt key")
    parser.add_argument("--prompt_type", type=str, default="default", help="Prompt type")
    
    # Args expected by utils logic
    parser.add_argument("--use_mc_options", type=bool, default=True, help="Include MC options question?")
    parser.add_argument("--ood_train_dataset", type=str, default=None, help="Dataset to use to assemble few-shot prompt")
    parser.add_argument("--num_few_shot", type=int, default=5, help="Number of few shot examples to use")
    
    # Handling specific dataset quirks (copied from run.py)
    parser.add_argument("--answerable_only", action="store_true", default=False, help='Exclude unanswerable questions.')

    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Starting with args: {args}")
    
    # Apply dataset specific overrides based on run.py logic
    if args.dataset == 'svamp':
        if not args.use_context:
            logging.info('Forcing `use_context=True` for svamp dataset.')
            args.use_context = True
    elif args.dataset == 'squad':
        if not args.answerable_only:
            logging.info('Forcing `answerable_only=True` for squad dataset.')
            args.answerable_only = True

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Load dataset
    logging.info(f"Loading dataset {args.dataset}...")
    # Load train to construct few-shot prompt
    train_dataset, validation_dataset = load_ds(args.dataset, seed=args.seed, add_options=args.use_mc_options)
    
    # Split dataset
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)
    
    # Filter validation dataset if answerable_only
    if args.answerable_only:
        val_answerable, _ = utils.split_dataset(validation_dataset)
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    # Create Few-Shot prompt
    if args.ood_train_dataset:
         logging.warning('Using OOD dataset %s to construct few-shot prompts.', args.ood_train_dataset)
         train_dataset, _ = load_ds(args.ood_train_dataset, add_options=args.use_mc_options)
         answerable_indices, _ = utils.split_dataset(train_dataset)

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    few_shot_prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    
    logging.info(f"Few-shot prompt preview: {few_shot_prompt[:200]}...")

    # Initialize model
    logging.info(f"Initializing model {args.model_name}...")
    token = "" # replace with your token if using models that require authentication
    
    try:
        model = utils.init_model(args, token=token)
    except Exception as e:
        logging.error(f"Failed to initialize model: {e}")
        # Check if model name needs to be adjusted
        logging.info("Attempting to proceed anyway (maybe token issue handled externally).")
        raise e
        
    # Process validation dataset
    logging.info("Processing validation dataset...")
    stubborn_data = []
    
    indices = range(len(validation_dataset))
    if args.num_samples is not None:
        indices = list(indices)[:min(args.num_samples, len(indices))]
    else:
        indices = list(indices)
        
    logging.info(f"Generating for {len(indices)} samples...")
    
    consistent_count = 0
    
    for i in tqdm(indices):
        example = validation_dataset[i]
        context = example.get("context", "")
        question = example["question"]
        
        # Construct prompt
        current_input = make_prompt(
            context, question, None, BRIEF, args.brief_always and args.enable_brief)
        full_prompt = few_shot_prompt + current_input
        
        # Generate N responses
        responses = []
        for _ in range(args.num_generations):
            # Using model.predict which calls generate
            # Returns: predicted_answer, token_log_likelihoods, embedding
            predicted_answer, _, _ = model.predict(full_prompt, temperature=args.temperature)
            responses.append(predicted_answer.strip())
            
        # Check consistency
        if len(set(responses)) == 1:
            consistent_count += 1
            # Add consistent response to the example data if desired, 
            # but user said "save those data" (referring to the dataset input).
            # I will save the original example. 
            # Optionally I could add the generated response, but "format same with the original dataset" implies strictly original format.
            # However, collecting the "consistent response" might be useful.
            # I'll stick to saving the original example to be safe with "format same with original".
            # BUT, to verify stubbornness later, maybe I should append the stubborn response?
            # User: "save those data. so i have a set of data which the model is consistent having the same responce with."
            # "make sure the dataset is saved with format same with the original dataset".
            # I will save the original example.
            stubborn_data.append(example)
            
    logging.info(f"Finished. Found {consistent_count} stubborn examples out of {len(indices)}.")
    
    # Save to JSON
    # Ensure directory exists
    output_path = args.output_file
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(__file__), output_path)
    
    logging.info(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        # Saving as a JSON list which datasets can load
        json.dump(stubborn_data, f, indent=4)
        
    logging.info("Done.")

if __name__ == "__main__":
    main()
