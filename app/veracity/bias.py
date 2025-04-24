# app/veracity/bias.py

import logging
import random
import time
import asyncio
from colorama import Fore, Style, init
from typing import List, Dict, Any, Optional, Tuple, Set

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Configure logging with colorful formatting
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s {Fore.MAGENTA}[%(levelname)s]{Style.RESET_ALL} %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("casalingua.bias")

# Define bias categories that might be detected
# In a real system, these would be more comprehensive
BIAS_CATEGORIES = {
    "political": "Political bias (left/right/centrist)",
    "gender": "Gender-related bias",
    "racial": "Racial or ethnic bias",
    "age": "Age-related bias",
    "socioeconomic": "Socioeconomic bias",
    "cultural": "Cultural bias",
    "religious": "Religious bias",
    "corporate": "Corporate/commercial bias",
}

async def check_bias(text: str) -> float:
    """
    Analyzes text for bias and returns a bias score.
    
    In a production environment, this would use a machine learning model
    trained to detect various forms of bias. For now, this is a stub 
    implementation that simulates the behavior of a bias checker.
    
    The bias score represents the estimated level of bias:
    - 0.0: No detectable bias
    - 1.0: Extremely biased
    
    Args:
        text (str): The text to analyze for bias
        
    Returns:
        float: A bias score between 0.0 and 1.0
    """
    # Log the bias check process
    logger.info(f"{Fore.YELLOW}Bias analysis initiated{Style.RESET_ALL}")
    logger.info(f"  {Fore.BLUE}Input length:{Style.RESET_ALL} {len(text)} characters")
    
    # Start a timer to simulate processing time
    start_time = time.time()
    
    # Simulate bias analysis work
    # In a real implementation, this would:
    # 1. Use NLP to understand the text's structure and sentiment
    # 2. Identify potentially biased language or framing
    # 3. Compare against a corpus of known biased/neutral content
    # 4. Calculate a quantifiable bias score
    
    # For this stub implementation:
    # 1. We assume a non-biased result (0.0) for demonstration
    # 2. Add a small amount of randomness for variation
    # 3. Add a delay to simulate processing time
    
    # Simulate processing time (randomly between 0.1 and 0.3 seconds)
    processing_time = 0.1 + (random.random() * 0.2)
    await asyncio.sleep(processing_time)
    
    # For demo purposes, mostly neutral content with low bias score
    bias_score = 0.0 + (random.random() * 0.05)
    
    # Ensure score is in valid range
    bias_score = max(0.0, min(1.0, bias_score))
    
    # Simulate detecting bias categories (randomly choose 0-2 categories)
    detected_categories = []
    if random.random() > 0.7:  # 30% chance of detecting some bias
        possible_categories = list(BIAS_CATEGORIES.keys())
        num_categories = random.randint(1, 2)
        detected_categories = random.sample(possible_categories, min(num_categories, len(possible_categories)))
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    
    # Determine color based on bias score (lower is better)
    if bias_score < 0.3:
        score_color = Fore.GREEN
    elif bias_score < 0.6:
        score_color = Fore.YELLOW
    else:
        score_color = Fore.RED
    
    # Log the result
    logger.info(f"{Fore.GREEN}✓ Bias analysis completed{Style.RESET_ALL} in {elapsed_time:.3f}s")
    logger.info(f"  {Fore.BLUE}Score:{Style.RESET_ALL} {score_color}{bias_score:.2f}{Style.RESET_ALL}")
    
    # Log detected bias categories if any
    if detected_categories:
        categories_str = ", ".join(detected_categories)
        logger.info(f"  {Fore.YELLOW}Detected bias categories:{Style.RESET_ALL} {categories_str}")
    else:
        logger.info(f"  {Fore.GREEN}No significant bias detected{Style.RESET_ALL}")
    
    # Add a note about this being a stub implementation
    logger.info(f"  {Fore.YELLOW}Note:{Style.RESET_ALL} Using stub implementation. Replace with real model.")
    
    return bias_score


async def get_bias_details(text: str) -> Dict[str, Any]:
    """
    Provides detailed bias analysis with category breakdown.
    
    This extends the basic bias score with category-specific analysis.
    In a production environment, this would return detailed metrics
    from a sophisticated bias detection model.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        Dict[str, Any]: Detailed bias information including:
            - overall_score: Overall bias score (0.0-1.0)
            - categories: Dictionary of category-specific scores
            - suggestions: List of suggestions to reduce bias
    """
    # Get the overall bias score
    overall_score = await check_bias(text)
    
    # Simulate category-specific analysis
    # In a real implementation, this would analyze different types of bias
    
    # Generate random category scores that average to the overall score
    categories = {}
    for category in BIAS_CATEGORIES:
        # Generate a score within ±0.2 of overall score, but within bounds
        category_score = max(0.0, min(1.0, overall_score + (random.random() * 0.4 - 0.2)))
        categories[category] = category_score
    
    # Generate mock suggestions
    suggestions = []
    if overall_score > 0.3:
        suggestions.append("Consider using more inclusive language")
    if overall_score > 0.5:
        suggestions.append("Review content for potential slant or framing bias")
    if overall_score > 0.7:
        suggestions.append("Multiple strong bias indicators detected, consider revising")
    
    # Log the detailed analysis
    logger.info(f"{Fore.YELLOW}Detailed bias analysis completed{Style.RESET_ALL}")
    logger.info(f"  {Fore.BLUE}Categories analyzed:{Style.RESET_ALL} {len(categories)}")
    logger.info(f"  {Fore.BLUE}Suggestions:{Style.RESET_ALL} {len(suggestions)}")
    
    # Return the detailed report
    return {
        "overall_score": overall_score,
        "categories": categories,
        "suggestions": suggestions
    }


# For demonstration/testing purposes
if __name__ == "__main__":
    async def test_bias_checker():
        """Test function to demonstrate the bias checker."""
        print(f"\n{Fore.YELLOW}Testing Bias Checker{Style.RESET_ALL}\n")
        
        # Test with some sample text
        test_text = "This is a sample text for bias analysis. It aims to be neutral and factual."
        
        # Run the basic bias check
        score = await check_bias(test_text)
        
        print(f"\n{Fore.YELLOW}Basic Bias Check Results:{Style.RESET_ALL}")
        print(f"  {Fore.BLUE}Text:{Style.RESET_ALL} \"{test_text}\"")
        print(f"  {Fore.BLUE}Bias Score:{Style.RESET_ALL} {score:.2f}")
        
        # Run the detailed bias check
        print(f"\n{Fore.YELLOW}Detailed Bias Analysis:{Style.RESET_ALL}")
        details = await get_bias_details(test_text)
        
        print(f"  {Fore.BLUE}Overall Score:{Style.RESET_ALL} {details['overall_score']:.2f}")
        
        print(f"  {Fore.BLUE}Category Breakdown:{Style.RESET_ALL}")
        for category, score in details['categories'].items():
            # Color code the category scores
            if score < 0.3:
                color = Fore.GREEN
            elif score < 0.6:
                color = Fore.YELLOW
            else:
                color = Fore.RED
                
            print(f"    - {category}: {color}{score:.2f}{Style.RESET_ALL}")
        
        print(f"  {Fore.BLUE}Suggestions:{Style.RESET_ALL}")
        if details['suggestions']:
            for suggestion in details['suggestions']:
                print(f"    - {suggestion}")
        else:
            print(f"    {Fore.GREEN}No suggestions needed{Style.RESET_ALL}")
    
    # Run the test
    asyncio.run(test_bias_checker())