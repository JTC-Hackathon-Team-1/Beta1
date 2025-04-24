# app/llms/simplify_llm.py

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
from colorama import Fore, Style, init
import time
import re

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Configure logging with colorful formatting
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s {Fore.MAGENTA}[%(levelname)s]{Style.RESET_ALL} %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("casalingua.simplify_llm")

# Model configuration
# ==================
# T5 is a text-to-text transformer model that can handle various NLP tasks
# For text simplification, we use a pre-trained model that can be fine-tuned
# T5-small is fast but less capable; consider upgrading to t5-base for better results
_MODEL_NAME = "t5-small"

# Load models once for efficiency (singleton pattern)
# ==================================================
logger.info(f"{Fore.YELLOW}Loading T5 model: {_MODEL_NAME}{Style.RESET_ALL}")
_tokenizer = T5Tokenizer.from_pretrained(_MODEL_NAME)
_model = T5ForConditionalGeneration.from_pretrained(_MODEL_NAME)
logger.info(f"{Fore.GREEN}✓ Model loaded successfully{Style.RESET_ALL}")

# Move to GPU if available for faster inference
if torch.cuda.is_available():
    _model = _model.to("cuda")
    logger.info(f"{Fore.GREEN}✓ Model moved to GPU{Style.RESET_ALL}")
else:
    logger.info(f"{Fore.YELLOW}Running on CPU - consider enabling GPU for faster performance{Style.RESET_ALL}")

# Legalese patterns to identify and replace
# ========================================
# These patterns help identify common legalese constructions that make text harder to read
LEGALESE_PATTERNS = [
    # Complex legal phrases and their simpler alternatives
    (r"notwithstanding anything to the contrary", "regardless of anything else"),
    (r"herein", "in this document"),
    (r"hereto", "to this"),
    (r"hereinafter", "from now on"),
    (r"hereby", "by this document"),
    (r"hereof", "of this"),
    (r"pursuant to", "according to"),
    (r"in accordance with", "following"),
    (r"aforementioned", "previously mentioned"),
    (r"shall", "will"),
    (r"utilize", "use"),
    (r"commence", "begin"),
    (r"terminate", "end"),
    (r"endeavor", "try"),
    (r"in the event that", "if"),
    (r"prior to", "before"),
    (r"subsequent to", "after"),
    (r"with respect to", "regarding"),
    (r"for the avoidance of doubt", "to be clear"),
    (r"set forth", "explained"),
    # Complex sentence structures
    (r"(.*) shall not (.*), unless (.*)", r"\1 will not \2, except if \3"),
    # Date references
    (r"the date hereof", "today's date"),
    # Negations
    (r"not fewer than", "at least"),
    (r"not less than", "at least"),
    (r"not later than", "by"),
]


async def simplify_text(text: str, target_grade: int = 9) -> str:
    """
    Simplify English text to approximately the target reading grade level.
    
    This function uses a two-stage approach:
    1. First, it applies pattern-based simplification for known legalese constructs
    2. Then it uses the T5 model to further simplify the text
    
    The pattern-based approach helps identify and replace common legal jargon,
    while the neural model handles more complex sentence structure simplification.
    
    Args:
        text: The text to simplify (string)
        target_grade: Target reading grade level (integer, default=9)
                     Grade 9 is the recommended level for general audience
                     comprehension according to readability guidelines
                     
    Returns:
        str: The simplified text
    """
    start_time = time.time()
    
    # Log the input text characteristics
    word_count = len(text.split())
    logger.info(f"{Fore.CYAN}Simplifying text:{Style.RESET_ALL} {word_count} words, {len(text)} chars")
    logger.info(f"{Fore.CYAN}Target grade level:{Style.RESET_ALL} {target_grade}")
    
    # Step 1: Check if the text appears to be legalese
    # ===============================================
    # This is a simple heuristic to detect legal language
    legalese_indicators = ["herein", "hereto", "shall", "pursuant", "notwithstanding", "thereof", "whereof"]
    is_legalese = any(indicator in text.lower() for indicator in legalese_indicators)
    
    # Step 2: Apply pattern-based simplification for legalese
    # ======================================================
    original_text = text
    
    if is_legalese:
        logger.info(f"{Fore.YELLOW}Legalese detected - applying specialized patterns{Style.RESET_ALL}")
        
        # Track changes for logging
        pattern_replacements = 0
        
        # Apply each pattern
        for pattern, replacement in LEGALESE_PATTERNS:
            # Case-insensitive replacement
            new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if new_text != text:
                pattern_replacements += 1
                text = new_text
        
        logger.info(f"{Fore.GREEN}✓ Applied {pattern_replacements} legalese pattern replacements{Style.RESET_ALL}")
    
    # Step 3: Prepare the prompt for T5
    # ================================
    # T5 uses a text-to-text format where the task is specified in the input
    if is_legalese:
        # Use a more specific prompt for legal text
        prompt = f"simplify legal language (grade {target_grade}): {text}"
    else:
        # Standard simplification prompt
        prompt = f"simplify (grade {target_grade}): {text}"
    
    # Step 4: Tokenize and generate simplified text
    # ===========================================
    logger.info(f"{Fore.CYAN}Tokenizing text for T5 model...{Style.RESET_ALL}")
    
    # Convert text to token IDs that the model understands
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Move inputs to the same device as the model (GPU if available)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate simplified text using the T5 model
    logger.info(f"{Fore.CYAN}Generating simplified text...{Style.RESET_ALL}")
    generation_start = time.time()
    
    with torch.no_grad():  # Disable gradient calculation for inference
        # Generate with improved parameters for better quality
        outputs = _model.generate(
            **inputs,
            max_length=512,  # Limit output length
            min_length=10,   # Ensure it's not too short
            num_beams=4,     # Beam search for better quality
            no_repeat_ngram_size=2,  # Avoid repeating phrases
            early_stopping=True      # Stop when done
        )
    
    generation_time = time.time() - generation_start
    logger.info(f"{Fore.GREEN}✓ Generation completed in {generation_time:.3f}s{Style.RESET_ALL}")
    
    # Decode the generated token IDs back to text
    simplified_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Step 5: Post-processing and cleanup
    # =================================
    # Remove any "(grade X):" prefix that might have been generated
    simplified_text = re.sub(r'^\(grade \d+\):\s*', '', simplified_text)
    
    # Clean up extra whitespace
    simplified_text = re.sub(r'\s+', ' ', simplified_text).strip()
    
    # If T5 didn't change much but we detected legalese and made pattern replacements,
    # at least return the pattern-replaced version
    if simplified_text.strip() == original_text.strip() and original_text != text:
        logger.info(f"{Fore.YELLOW}T5 didn't simplify; using pattern-replaced version{Style.RESET_ALL}")
        simplified_text = text
    
    # Step 6: Log results and return
    # ============================
    total_time = time.time() - start_time
    
    # Calculate stats for before and after
    original_words = len(original_text.split())
    simplified_words = len(simplified_text.split())
    reduction_percent = (1 - (simplified_words / original_words)) * 100 if original_words > 0 else 0
    
    logger.info(f"{Fore.GREEN}✓ Simplification completed in {total_time:.3f}s{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}Original:{Style.RESET_ALL} {original_words} words")
    logger.info(f"{Fore.CYAN}Simplified:{Style.RESET_ALL} {simplified_words} words")
    
    if reduction_percent > 0:
        logger.info(f"{Fore.GREEN}Text reduced by {reduction_percent:.1f}%{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.YELLOW}Text not reduced in length{Style.RESET_ALL}")
    
    # Sample of simplified text (first 50 chars)
    logger.info(f"{Fore.CYAN}Sample output:{Style.RESET_ALL} \"{simplified_text[:50]}...\"")
    
    return simplified_text


# Example usage for testing (only runs when executed directly)
if __name__ == "__main__":
    import asyncio
    
    # Test legalese text
    test_text = """
    Notwithstanding anything to the contrary herein, the Party of the First Part shall, 
    in accordance with the terms set forth in Exhibit A attached hereto and incorporated 
    herein by reference, remit payment to the Party of the Second Part no later than 
    thirty (30) days subsequent to the receipt of an invoice from the Party of the Second Part.
    """
    
    # Run the async function
    simplified = asyncio.run(simplify_text(test_text, target_grade=7))
    
    print("\nOriginal:")
    print(test_text)
    print("\nSimplified:")
    print(simplified)