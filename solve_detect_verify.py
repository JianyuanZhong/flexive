import numpy as np
import json
import re
import os
import argparse
import multiprocessing
import logging
import sys
import time
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import math

from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
from evaluate import load 

from math_verify import parse

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# Load math evaluator
math_evaluator = load("competition_math")

# Constants for FLOPS calculation
FLOPS_PER_TOKEN = {
    "DeepSeek-R1-Distill-Qwen-14B": 1,  # Estimated flops per token for a LLM model
    "flexive": 1,  # Estimated flops per token for verification model
}


from collections import Counter


def majority_vote(answers: list, prob: dict) -> tuple:
    """
    Extract answers and use majority voting to determine the final answer.
    Returns (final_answer, is_correct, all_extracted_answers, chosen_solution)
    """
    extracted_answers = []
    extracted_indices = []  # 存储每个 extracted_answer 对应的原始 answers 的索引
    
    for idx, ans in enumerate(answers):
        try:
            extracted = parse(ans)[-1]
            if extracted is not None:  # Only include valid parsed answers
                extracted_answers.append(extracted)
                extracted_indices.append(idx)  # 记录原始索引
        except:
            continue
    
    if not extracted_answers:
        return None, 0, [], None, 0
    
    # Get the most common answer
    answer_counts = Counter(extracted_answers)
    final_answer = answer_counts.most_common(1)[0][0]
    final_answer = str(final_answer)
    chosen_index = 0
    for i, ans in enumerate(extracted_answers):
        if ans == final_answer:
            chosen_index = extracted_indices[i]  # 获取原始 answers 的索引
            break
    
    expected_answer = str(prob['expected_answer'])

    # Check if the majority answer is correct
    is_correct = 1 if math_evaluator.compute(references=[expected_answer], predictions=[final_answer])["accuracy"] > 0.99 else 0
    
    return final_answer, is_correct, extracted_answers, answers[chosen_index], chosen_index

def setup_logger(log_level: str) -> logging.Logger:
    """Configure and return a logger."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"math_solver_{timestamp}.log")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger("math_solver")

@dataclass
class Config:
    """Configuration for the application."""
    model_name: str = "DeepSeek-R1-Distill-Qwen-14B"
    api_key: str = ""  # Placeholder for API key, not used if empty
    base_url: str = "http://localhost:8018/v1"
    # List of LM model base URLs to pick from
    lm_base_urls: List[str] = field(default_factory=lambda: ["http://localhost:8001/v1", 
                                                          "http://localhost:8002/v1", 
                                                          "http://localhost:8003/v1","http://localhost:8004/v1","http://localhost:8005/v1","http://localhost:8006/v1"])
    tokenizer_path: str = "./DeepSeek-R1-Distill-Qwen-14B"
    critique_template_path: str = "./flexive_templates/critique_template.txt"
    problems_path: str = "./aime_2024.jsonl"
    max_attempts: int = 1
    max_tokens: int = 32768
    temperature: float = 0.6
    log_level: str = "INFO"
    # Dyve-specific configuration
    flexive_model_name: str = "flexive"
    majority_voting_n : int = 4
    flexive_base_url: str = "http://localhost:8017/v1"
    # Adaptive verification parameters
    adaptive_verification: bool = True
    verification_k: int = 4  # Number of fast verifications
    verification_threshold: float = 0.9  # Agreement threshold (90%)
    slow_thinking: bool = True  # Flag for slow thinking mode
    # Best-of-n parameters
    best_of_n: int = 4  # Number of solutions to generate in second round
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create a Config instance from command-line arguments."""
        config = cls()
        for key, value in vars(args).items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        return config


class BaseClient:
    """Base class for API clients."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.logger.debug(f"Initializing {self.__class__.__name__} with model {config.model_name}")
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=300
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path, 
            trust_remote_code=True
        )
        self.logger.debug(f"Tokenizer loaded from {config.tokenizer_path}")
    

class LLMClient(BaseClient):
    """Client for interacting with the LLM API."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        # Pick a random LM base URL for this client instance
        if hasattr(config, 'lm_base_urls') and config.lm_base_urls:
            selected_url = random.choice(config.lm_base_urls)
            logger.info(f"Selected LM model URL: {selected_url}")
            config.base_url = selected_url
        
        super().__init__(config, logger)
    
    def get_response(self, num_attempt: int, steps_so_far: str, feedback: str, prompt: str):
        """Generate a response from the LLM."""
        self.logger.debug(f"Getting response for attempt {num_attempt}")
        
        if num_attempt == 0:  # first step
            messages = [{"role": "user", "content": prompt}]
            self.logger.debug("First attempt, using chat completions")
            
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                n=1,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                logprobs=True,
                # seed = 8
            )
        else:
            prompt_text = prompt
            self.logger.debug(f"Subsequent attempt {num_attempt}, using completions with feedback")
            completion = self.client.completions.create(
                model=self.config.model_name,
                prompt=prompt_text,
                n=1,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                logprobs=5,
                # seed = 42
            )
    
        return completion
        
    def continue_after_detected(self, messages):
        """Continue generation after hesitation is detected."""
        self.logger.debug("Continuing generation after hesitation detection")
        
        if isinstance(messages, list):
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = messages
            
        prompt = prompt + '\n</think>\n\n'
        
        extra_body = {"add_generation_prompt": False, "continue_final_message": True}
        
        self.logger.debug("Sending continuation request")
        completion = self.client.completions.create(
            model=self.config.model_name,
            prompt=prompt,
            n=1,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            extra_body=extra_body,
            seed = 42
        )
        
        self.logger.debug("Received continuation response")
        return completion.choices[0].text
        
    def get_detected_response(self, prompt: str):
        """Generate a response with detection of completion."""
        self.logger.debug("Getting detected response for completeness check")
        
        if isinstance(prompt, list):
            messages = prompt
            prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            messages = [{"role": "user", "content": prompt}]
            prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
        prompt_text = prompt_text + '\nOkay, I think I have finished thinking.\n</think>\n\n'

        completion = self.client.completions.create(
            model=self.config.model_name,
            prompt=prompt_text,
            n=1,
            temperature=self.config.temperature,
            max_tokens=1,
            stream=True,
            logprobs=10,
            # seed = 42
        )
        
        return completion


class FlexiVeVerifier(BaseClient):
    """Client for interacting with the FlexiVe verification API."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.logger.debug(f"Initializing {self.__class__.__name__} with FlexiVe model at {config.flexive_base_url}")
        # Create a client specifically for the FlexiVe model
        self.base_url = random.choice(["http://localhost:8017/v1","http://localhost:8018/v1"])
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=config.api_key,
            timeout=300
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path, 
            trust_remote_code=True
        )
        self.logger.debug(f"Tokenizer loaded from {config.tokenizer_path}")
        
        with open(config.critique_template_path) as f:
            self.template = f.read().strip()
            self.logger.debug(f"Loaded critique template from {config.critique_template_path}")

    def prepare_input_boxed(self, problem: str, solution: str) -> str:
        """Prepare input with tagged paragraphs based on hesitation keywords."""
        # Get the solver instance to access hesitation keyword splitting
        solver = MathProblemSolver(self.config)
        steps = solver.split_solution_by_hesitation(solution)
        self.logger.info(f"num steps: {len(steps)}")
        self.logger.info(f"Steps: {steps}")
        
        tagged_response = ''
        for sdx, step in enumerate(steps):
            tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
        tagged_response = tagged_response.strip()
        prompt = self.template.format(problem=problem, tagged_response=tagged_response)
        self.logger.debug(f"Prepared input with {len(steps)} tagged paragraphs split by hesitation keywords")
        return prompt

    def get_response(self, problem: str, solution: str, slow_thinking: bool = False) -> str:
        """Get verification response for a solution."""
        self.logger.debug(f"Getting verification response from FlexiVe with slow_thinking={slow_thinking}")
        prompt = self.prepare_input_boxed(problem, solution)

        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        if slow_thinking:
            # Add slow thinking instruction to prompt
            prompt_text = prompt_text
            # Don't close the thinking tag to encourage more detailed verification
            self.logger.debug("Using slow thinking mode for verification")
        else:
            # Standard mode - close thinking tag immediately
            prompt_text = prompt_text + '\nOkay, I think I have finished thinking.\n</think>\n\n'
        
        self.logger.debug("Sending prompt to FlexiVe API")
        completion = self.client.completions.create(
            model=self.config.flexive_model_name,
            prompt=prompt_text,
            n=1,
            temperature=0.6,
            # top_p=0.9,
            max_tokens=self.config.max_tokens,
            # seed=8
        )
        
        self.logger.debug("Received verification response from FlexiVe")
        return completion.choices[0].text


class MathProblemSolver:
    """Main class for solving and evaluating math problems."""
    
    def __init__(self, config: Config):
        self.config = config
        # Set up logger
        self.logger = setup_logger(config.log_level)
        self.logger.info(f"Initializing MathProblemSolver with config: {config}")
        
        self.llm_client = LLMClient(config, self.logger)
        self.flexive_verifier = FlexiVeVerifier(config, self.logger)
        
        # Set of words that indicate hesitation in model responses
        self.hesitation_keywords = {
            "wait", "double-check", "alternatively", "hmm", "let me check", 
            "let me double-check", "make sure", "another way", "let me verify", "to confirm"
        }
        self.logger.debug(f"Hesitation keywords: {self.hesitation_keywords}")

    def extract_answer_judge(self, solution_text: str) -> Optional[str]:
        """Extract the answer from a boxed solution."""
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(boxed_pattern, solution_text)
        if matches:
            answer = matches[-1].strip()
            self.logger.debug(f"Extracted answer: {answer}")
            return answer
        self.logger.warning("Could not extract boxed answer")
        return None

    def split_solution_by_hesitation(self, solution_text: str) -> List[str]:
        """Split solution text by hesitation keywords."""
        # Create a regex pattern from hesitation keywords
        # Use lookahead to keep the hesitation keyword with the following step
        hesitation_pattern = r'(?:^|(?<=\.\s)|(?<=\n))(' + '|'.join(re.escape(kw) for kw in self.hesitation_keywords) + r')'
        
        # Split by hesitation keywords
        parts = re.split(hesitation_pattern, solution_text, flags=re.IGNORECASE)
        
        # Recombine the parts to keep hesitation keywords with the following text
        steps = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and any(parts[i].lower().strip() == kw.lower() for kw in self.hesitation_keywords):
                steps.append(parts[i] + (parts[i+1] if i+1 < len(parts) else ""))
                i += 2
            else:
                if parts[i].strip():  # Only add non-empty steps
                    steps.append(parts[i])
                i += 1
                
        # Remove any empty steps that might result from the split
        steps = [step.strip() for step in steps if step.strip()]

        # Check if the last step contains only a hesitation keyword
        if steps and any(steps[-1].lower().strip() == kw.lower() for kw in self.hesitation_keywords):
            self.logger.debug(f"Removing last step that contains only a hesitation keyword: '{steps[-1]}'")
            steps = steps[:-1]
        
        self.logger.debug(f"Split solution into {len(steps)} steps by hesitation keywords")
        return steps

    def collect_current_steps(self, solution_text: str, error_step_index: int) -> str:
        """Collect steps up to the error step using hesitation keywords for splitting."""
        # Split the solution text using hesitation keywords
        steps = self.split_solution_by_hesitation(solution_text)
        
        # Check if error_step_index is valid
        if error_step_index < 0 or error_step_index >= len(steps):
            self.logger.warning(f"Error step index {error_step_index} is out of range (0-{len(steps)-1})")
            return ' '.join(steps)
            
        # Select all steps including the error step
        correct_steps = steps[:error_step_index+1]
        
        # Join the correct steps back together
        result = ' '.join(correct_steps)
        self.logger.debug(f"Collected {len(correct_steps)} correct steps including error at step {error_step_index}")
        return result

    def verify_solution(self, problem: str, solution: str) -> Tuple[Optional[bool], Optional[int], Optional[str]]:
        """Verify a solution using the Dyve client."""
        self.logger.info("Verifying solution")
        
        if not self.config.adaptive_verification:
            # Original behavior - single verification
            feedback = self.flexive_verifier.get_response(problem, solution)
            return self._process_verification_result(feedback)
        
        # Adaptive verification scheme
        self.logger.info(f"Using adaptive verification with k={self.config.verification_k}")
        
        # First phase: k fast verifications
        fast_results = []
        for i in range(self.config.verification_k):
            self.logger.info(f"Fast verification {i+1}/{self.config.verification_k}")
            feedback = self.flexive_verifier.get_response(problem, solution, slow_thinking=False)
            # log the feedback
            self.logger.info(f"Fast verification {i+1}/{self.config.verification_k}\nfeedback: {feedback}")
            is_valid, error_step, feedback = self._process_verification_result(feedback)
            if is_valid is not None:  # Only count valid results
                fast_results.append((is_valid, error_step, feedback))
                
        # Check if we have enough results and calculate agreement
        if not fast_results:
            self.logger.warning("No valid results from fast verification")
            return None, None, "No valid results from verification"
            
        # Modified agreement calculation to consider error steps within 5 steps as agreement
        # Group results by is_valid first
        valid_groups = {}
        for is_valid, error_step, feedback in fast_results:
            if is_valid not in valid_groups:
                valid_groups[is_valid] = []
            valid_groups[is_valid].append((error_step, feedback))
        
        # Count agreements with exact matching on error_step - no fuzzy matching
        agreement_count = {}
        for is_valid, group in valid_groups.items():
            for error_step, feedback in group:
                key = (is_valid, error_step)
                if key in agreement_count:
                    agreement_count[key] += 1
                else:
                    agreement_count[key] = 1
        
        # Find the most common result
        if not agreement_count:
            self.logger.warning("No valid agreement count calculated")
            return None, None, "No valid agreement count"
            
        most_common_result = max(agreement_count.items(), key=lambda x: x[1])
        most_common_count = most_common_result[1]
        agreement_ratio = most_common_count / len(fast_results)
        
        self.logger.info(f"Fast verification agreement ratio: {agreement_ratio:.2f} with exact error step matching")
        
        # If agreement is above threshold, return the most common result
        if agreement_ratio >= self.config.verification_threshold:
            is_valid, error_step = most_common_result[0]
            # Find a feedback that corresponds to these values or close enough
            feedback = None
            closest_match = float('inf')
            
            for result_tuple in fast_results: # Renamed 'result' to 'result_tuple' to avoid conflict with outer scope 'result'
                if result_tuple[0] == is_valid:
                    # Find the closest error_step match
                    distance = abs(result_tuple[1] - error_step)
                    if distance < closest_match:
                        closest_match = distance
                        feedback = result_tuple[2]
                    
                    # If exact match, break immediately
                    if distance == 0:
                        break
            
            self.logger.debug(f"Fast verification agreement ratio: {agreement_ratio:.2f}, feedback: {feedback}")
            return is_valid, error_step, feedback
        
        # Second phase: k/4 slow thinking verifications
        slow_k = math.ceil(self.config.verification_k / 4)
        self.logger.info(f"Agreement below threshold, using slow thinking for {slow_k} verifications")
        
        slow_results = []
        for i in range(slow_k):
            self.logger.debug(f"Slow verification {i+1}/{slow_k}")
            feedback = self.flexive_verifier.get_response(problem, solution, slow_thinking=True)
            # log the feedback
            # self.logger.info(f"Slow verification {i+1}/{slow_k}\nfeedback: {feedback}")
            is_valid, error_step, feedback = self._process_verification_result(feedback)
            if is_valid is not None:  # Only count valid results
                slow_results.append((is_valid, error_step, feedback))
                
        # Check if we have enough results
        if not slow_results:
            self.logger.warning("No valid results from slow verification")
            return None, None, "No valid results from verification"
            
        # Group results by is_valid first
        valid_groups = {}
        for is_valid, error_step, feedback in slow_results:
            if is_valid not in valid_groups:
                valid_groups[is_valid] = []
            valid_groups[is_valid].append((error_step, feedback))
        
        # Count agreements with exact matching on error_step
        slow_agreement_count = {}
        for is_valid, group in valid_groups.items():
            # Skip if there's only one result in this validity group
            if len(group) <= 1:
                key = (is_valid, group[0][0])  # (is_valid, error_step)
                slow_agreement_count[key] = 1
                continue
                
            # Process each error_step in the group with exact matching
            for error_step, feedback in group:
                key = (is_valid, error_step)
                if key in slow_agreement_count:
                    slow_agreement_count[key] += 1
                else:
                    slow_agreement_count[key] = 1
        
        # Find the most common result from slow thinking
        if not slow_agreement_count:
            self.logger.warning("No valid slow agreement count calculated")
            return None, None, "No valid slow agreement count"
            
        most_common_slow = max(slow_agreement_count.items(), key=lambda x: x[1])
        is_valid, error_step = most_common_slow[0]
        
        # Get feedback for the most common slow result or closest match
        feedback = None
        closest_match = float('inf')
        
        for result_tuple in slow_results: # Renamed 'result' to 'result_tuple'
            if result_tuple[0] == is_valid:
                # Find the closest error_step match
                distance = abs(result_tuple[1] - error_step)
                if distance < closest_match:
                    closest_match = distance
                    feedback = result_tuple[2]
                    
                    # If exact match, break immediately
                    if distance == 0:
                        break
        
        self.logger.info(f"Using result from slow verification with exact error step matching")
        return is_valid, error_step, feedback
        
    def _process_verification_result(self, feedback: str) -> Tuple[Optional[bool], Optional[int], Optional[str]]:
        """Process the verification response to extract the result."""
        error_step = self.extract_answer_judge(feedback)
        
        try:
            error_step_int = int(error_step)
            if error_step_int > -1:
                self.logger.info(f"Solution verification found error at step {error_step_int}")
                return False, error_step_int, feedback
            elif error_step_int == -1:
                self.logger.info("Solution verification passed")
                return True, error_step_int, feedback
            else:
                self.logger.warning(f"Invalid error step index: {error_step_int}")
                return None, None, None
        except (ValueError, TypeError):
            self.logger.warning(f"Could not parse error step: {error_step}")
            return None, None, None

    def prepare_problem_prompt(self, problem: Dict[str, str]) -> str:
        """Prepare the prompt for a math problem."""
        self.logger.debug("Preparing problem prompt")
        proposer_prompt = f"""
The following is a math problem:

[Math Problem]

{problem["question"]}

Your task is to solve it step by step. 
Please put your final answer (i.e., the index) in \\boxed{{}}.
"""
        return proposer_prompt

    def evaluate_answer(self, problem: Dict[str, str], response: str) -> Dict[str, Any]:
        """Evaluate an answer against the expected answer."""
        self.logger.info("Evaluating answer")
        if response is None:
            self.logger.warning("No response to evaluate")
            return None
            
        extracted_ans = parse(response)
        if isinstance(extracted_ans, list):
            if not extracted_ans:
                extracted_ans = 'No answer'
                self.logger.warning("Empty list of answers extracted")
            else:
                extracted_ans = str(extracted_ans[-1])
                self.logger.debug(f"Last answer from list: {extracted_ans}")  
        else:
            extracted_ans = str(extracted_ans)
            self.logger.debug(f"Extracted answer: {extracted_ans}")
        
        expected_answer = str(problem["expected_answer"])
        accuracy = math_evaluator.compute(
            references=[expected_answer], 
            predictions=[extracted_ans]
        )["accuracy"]
        
        pass_at_1 = 1 if accuracy > 0.99 else 0
        self.logger.error(f"Answer evaluation: extracted={extracted_ans}, expected={problem['expected_answer']}, accuracy={accuracy}, pass@1={pass_at_1}")
        
        return {
            "extracted_answer": extracted_ans,
            "pass@1": pass_at_1,
            "accuracy": accuracy
        }

    def _process_stream(self, response_stream, problem: Dict[str, str], apply_detection: bool, skip_verification: bool = False) -> Union[str, Tuple[str, bool, Optional[int], Optional[str]]]:
        """Process a response stream and handle hesitation detection if enabled. 
        If verification is performed, returns (response, is_valid, error_step_index, verifier_feedback).
        Set skip_verification=True to bypass early verification on final attempts."""
        self.logger.debug("Processing response stream")
        stream_response = ""
        should_break = False
        
        for chunk in response_stream:
            if hasattr(chunk.choices[0], 'delta'):  # Chat completion
                chunk_content = chunk.choices[0].delta.content if hasattr(chunk.choices[0].delta, 'content') else ""
            else:  # Regular completion
                chunk_content = chunk.choices[0].text if hasattr(chunk.choices[0], 'text') else ""
                
            if chunk_content:  
                stream_response += chunk_content
            
            if apply_detection:
                current_response_lower = stream_response.lower()
                for keyword in self.hesitation_keywords:
                    if current_response_lower.rstrip().endswith(keyword):
                        self.logger.info(f"Detected hesitation keyword: '{keyword}'")
                        
                        # Check if the solution is complete
                        check_complete_prompt = f"""Given current solution to a math problem, check if it is a complete solution (i.e., contains the final answer). 

You should respond with only one word `Yes` if the current solution is complete, or `No` if it is not.
"""
                        
                        messages = [
                            {"role": "user", "content": problem["question"]},
                            {"role": "assistant", "content": stream_response},
                            {"role": "user", "content": check_complete_prompt},
                        ]
                        
                        detect_stream = self.llm_client.get_detected_response(messages)
                        for chunk_inner in detect_stream: # Renamed 'chunk' to 'chunk_inner'
                            if 'Yes' in chunk_inner.choices[0].logprobs.top_logprobs[0]:
                                if chunk_inner.choices[0].logprobs.top_logprobs[0]['Yes'] > chunk_inner.choices[0].logprobs.top_logprobs[0].get('No', float('-inf')):
                                    should_break = True
                                    break
                        if should_break:
                            break
            
            if should_break:
                break
        
        if should_break:
            self.logger.info("Hesitation detected, continuing generation after hesitation")
            messages = [
                {"role": "user", "content": self.prepare_problem_prompt(problem)},
                {"role": "assistant", "content": stream_response},
            ]
            
            # Continue generation after detection
            continuation = self.llm_client.continue_after_detected(messages)
            self.logger.debug(f"Generated continuation with length: {len(continuation)}")
            stream_response += continuation

        clean_response = stream_response.replace('<think>', '').replace('</think>', '')
        self.logger.debug(f"Final response length: {len(clean_response)} characters")
        
        return clean_response

    def process_with_verification(self, problem: Dict[str, str]) -> Dict[str, Any]:
        """Process a problem with verification."""
        problem_id = problem.get("id", "unknown")
        self.logger.info(f"Processing problem with verification: {problem_id}")
        self.logger.debug(f"Problem question: {problem['question'][:100]}...")
        
        attempts = 0
        prompt = self.prepare_problem_prompt(problem)
        response = "" # Initialize response
        verifier_feedback = "" # Initialize verifier_feedback
        
        while attempts < self.config.max_attempts:
            self.logger.info(f"Attempt {attempts+1}/{self.config.max_attempts}")
            
            if attempts == 0:
                # First attempt uses just the original prompt
                
                response_stream = self.llm_client.get_response(0, '', '', prompt)
                # Always skip early verification here - we'll do it after getting the full response
                response = self._process_stream(response_stream, problem, apply_detection=True, skip_verification=True)

                is_valid, error_step_index, verifier_feedback = self.verify_solution(problem["question"], response)
                self.logger.debug(f"Verification result: is_valid={is_valid}, error_step_index={error_step_index}")

                if is_valid:
                    break
                else:
                    # For subsequent attempts, create a message with the feedback
                    messages = [
                        {"role": "user", "content": self.prepare_problem_prompt(problem)},
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": f"""Your solution has an error. Here's the feedback from verification:

    {verifier_feedback}

    Note: When analyzing your solution, the verifier breaks it down into paragraphs based on hesitation keywords like "wait", "hmm", "let me check", "double-check", etc. Each paragraph corresponds to a step in your reasoning between these hesitation points.

    Please correct your solution and try again. Provide a complete solution with clear reasoning steps."""}
                    ]
                    # Convert messages to prompt text using the tokenizer
                    prompt_text = self.llm_client.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    
                    # Implement best-of-n approach for second round generation
                    self.logger.info(f"Implementing best-of-n with n={self.config.best_of_n} for second attempt")
                    candidate_responses = []
                    candidate_answers = []
                    
                    for i in range(self.config.best_of_n):
                        self.logger.info(f"Generating candidate solution {i+1}/{self.config.best_of_n}")
                        response_stream = self.llm_client.get_response(1, '', '', prompt_text)
                        candidate = self._process_stream(response_stream, problem, apply_detection=True, skip_verification=True)
                        candidate_responses.append(candidate)
                        
                        # Extract answer from each candidate
                        extracted_answer = parse(candidate)
                        if isinstance(extracted_answer, list):
                            if not extracted_answer:
                                extracted_answer = 'No answer'
                            else:
                                extracted_answer = str(extracted_answer[-1])
                        else:
                            extracted_answer = str(extracted_answer)
                        
                        candidate_answers.append(extracted_answer)
                        self.logger.info(f"Candidate {i+1} extracted answer: {extracted_answer}")
                    
                    # Majority voting on answers
                    if candidate_answers:
                        # Count occurrences of each answer
                        answer_counts = {}
                        for ans in candidate_answers:
                            if ans in answer_counts:
                                answer_counts[ans] += 1
                            else:
                                answer_counts[ans] = 1
                                
                        # Find majority answer
                        majority_answer, max_count = max(answer_counts.items(), key=lambda x: x[1])
                        self.logger.error(f"Majority answer: {majority_answer} (count: {max_count}/{self.config.best_of_n})")
                        
                        # Find the first response that has the majority answer
                        best_idx = 0 # Default to 0
                        for i, ans in enumerate(candidate_answers):
                            if ans == majority_answer:
                                best_idx = i
                                break
                    else:
                        # If somehow we have no answers, take the first response
                        best_idx = 0
                        self.logger.warning("No answers extracted, selecting first candidate by default")
                    
                    # Use the best candidate as the response
                    response = candidate_responses[best_idx]
                    
                    # Since we're not doing verification, assume it's valid to proceed
                    is_valid = True
                    error_step_index = -1
                    verifier_feedback = ""

                if is_valid:
                    self.logger.info("Solution is valid, evaluating answer")
                    evaluation = self.evaluate_answer(problem, response)
                    if evaluation is None:
                        return None
                        
                    result = {
                        "attempt_num": attempts,
                        "question": problem["question"],
                        "expected_answer": problem["expected_answer"],
                        "generated_answer": response,
                        "pass@1": evaluation["pass@1"],
                        "accuracy": evaluation["accuracy"],
                        "extracted_answer": evaluation["extracted_answer"],
                        "feedback": verifier_feedback
                    }
                    self.logger.info(f"Problem solved successfully after {attempts+1} attempts")
                    return result
                
            elif is_valid is None: # This condition was part of the original logic, kept for consistency
                self.logger.warning("Verification failed, treating as final attempt")
                # attempts = -1 # This was in the original, but seems to make attempts inconsistent. Removing.
                evaluation = self.evaluate_answer(problem, response)
                if evaluation is None:
                    return None
                    
                result = {
                    "attempt_num": attempts, # Use current attempts
                    "question": problem["question"],
                    "expected_answer": problem["expected_answer"],
                    "generated_answer": response,
                    "pass@1": evaluation["pass@1"],
                    "accuracy": evaluation["accuracy"],
                    "extracted_answer": evaluation["extracted_answer"],
                    "feedback": ""
                }
                self.logger.info("Problem evaluation completed with verification failure")
                return result # Exit after verification failure
                
            else: # is_valid is False
                self.logger.info(f"Solution invalid at step {error_step_index}, preparing for next attempt")
                # Store the verifier feedback for the next attempt
                feedback_text = verifier_feedback if verifier_feedback is not None else "" # Renamed feedback to feedback_text
                # Process feedback to make it more user-friendly
                if feedback_text:
                    # Remove \\boxed{} notation and content inside it
                    cleaned_feedback = re.sub(r'\\boxed\{[^}]*\}', '', feedback_text)
                    
                    # Extract only content before </think> if present
                    think_match = re.search(r'(.*?)</think>', cleaned_feedback, re.DOTALL)
                    if think_match:
                        cleaned_feedback = think_match.group(1).strip()
                        cleaned_feedback = cleaned_feedback.replace('<think>', '')
                    
                    verifier_feedback = cleaned_feedback # Update verifier_feedback for the next loop iteration's message
                
                attempts += 1
        
        # If we've exhausted attempts
        self.logger.warning(f"Exhausted all {self.config.max_attempts} attempts, evaluating best result")
        evaluation = self.evaluate_answer(problem, response)
        if evaluation is None:
            return None
            
        result = {
            "attempt_num": attempts,
            "question": problem["question"],
            "expected_answer": problem["expected_answer"],
            "generated_answer": response,
            "pass@1": evaluation["pass@1"],
            "accuracy": evaluation["accuracy"],
            "extracted_answer": evaluation["extracted_answer"],
            "feedback": verifier_feedback
        }
        self.logger.info("Completed all attempts without finding valid solution")
        return result

    def process_without_verification(self, problem: Dict[str, str], apply_detection: bool) -> Dict[str, Any]:
        """Process a problem without verification."""
        problem_id = problem.get("id", "unknown")
        self.logger.info(f"Processing problem without verification: {problem_id}")
        self.logger.debug(f"Problem question: {problem['question'][:100]}...")
        
        prompt = self.prepare_problem_prompt(problem)
        response_stream = self.llm_client.get_response(0, '', '', prompt)
        
        self.logger.info(f"Processing response stream with detection={apply_detection}")
        # Always skip verification in this method
        stream_result = self._process_stream(response_stream, problem, apply_detection, skip_verification=True)
        
        response = stream_result
        
        evaluation = self.evaluate_answer(problem, response)
        if evaluation is None:
            return None
            
        result = {
            "attempt_num": 0,
            "question": problem["question"],
            "expected_answer": problem["expected_answer"],
            "generated_answer": response,
            "pass@1": evaluation["pass@1"],
            "accuracy": evaluation["accuracy"],
            "extracted_answer": evaluation["extracted_answer"],
            "feedback": ""
        }
        self.logger.info("Problem processing completed without verification")
        return result

    def solve_problem(self, problem: Dict[str, str], apply_detection: bool, apply_verification: bool) -> Dict[str, Any]:
        """Solve a math problem with optional verification and hesitation detection."""
        problem_id = problem.get("id", "unknown")
        self.logger.info(f"Solving problem {problem_id} with detection={apply_detection}, verification={apply_verification}")
        
        start_time = time.time()
        final_result = None # Initialize final_result
        try:
            if apply_verification:

                final_result = self.process_with_verification(problem)

            else:
                final_result = self.process_without_verification(problem, apply_detection)
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if final_result:
                final_result["solving_time"] = elapsed_time
                self.logger.error(f"Problem {problem_id} solved in {elapsed_time:.2f} seconds, pass@1={final_result.get('pass@1', 'N/A')}")
            
            return final_result
        except Exception as e:
            self.logger.exception(f"Error processing problem {problem_id}: {str(e)}")
            return None


def load_problems(file_path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Load problems from a JSONL file."""
    logger = logging.getLogger("math_solver")
    logger.info(f"Loading problems from {file_path}")
    
    problems = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                problem_data = json.loads(line) # Renamed 'problem' to 'problem_data'
                # Add an ID if not present
                if "id" not in problem_data:
                    problem_data["id"] = f"problem_{i+1}"
                if isinstance(problem_data['answer'], list):
                    problem_data['answer'] = problem_data['answer'][0]
                else:
                    problem_data['answer'] = problem_data['answer']
                problems.append({
                    'id': problem_data.get("id", f"problem_{i+1}"),
                    'question': problem_data['problem'],
                    'expected_answer': problem_data['answer']
                })
        
        logger.info(f"Loaded {len(problems)} problems from file")
        
        if limit is not None:
            problems = problems[:limit]
            logger.info(f"Limited to {len(problems)} problems")
            
        return problems
    except Exception as e:
        logger.exception(f"Error loading problems from {file_path}: {str(e)}")
        return []


def process_problem(problem: Dict[str, str], config_dict: Dict[str, Any],
                    apply_detection: bool, apply_verification: bool, majority_voting_n : int,
                    result_file: str, lock: multiprocessing.Lock) -> Optional[Dict[str, Any]]:
    """Process a single problem and save the result."""
    try:
        # Create a Config object from the dictionary
        config = Config(**config_dict)
        
        # Create the solver inside the worker process
        # This will trigger random selection of an LM model URL for this problem
        solver = MathProblemSolver(config)
        
        # Log which model URL was selected
        logger = logging.getLogger("math_solver") # Define logger here
        logger.info(f"Processing problem {problem.get('id', 'unknown')} with LM model at {solver.llm_client.config.base_url}")
        

        answer_in_results = []
        results_list= [] # Renamed 'results' to 'results_list'

        final_result_single = None # Initialize final_result_single

        if majority_voting_n == 1 :
            final_result_single = solver.solve_problem(problem, apply_detection, apply_verification)
        else:
            for i in range(majority_voting_n):
                current_result = solver.solve_problem(problem, apply_detection, apply_verification) # Renamed 'result' to 'current_result'
                if current_result: # Ensure result is not None
                    results_list.append(current_result)
                    answer_in_result = current_result.get('generated_answer', "") # Use .get for safety
                    answer_in_results.append(answer_in_result)
                
            if results_list: # Proceed only if we have some results
                final_answer_1, is_correct_1, extracted_answers_1, candidate_voted_answer, index = majority_vote(answer_in_results, problem)
                if is_correct_1 == 1 and index < len(results_list): # Check index boundary
                    results_list[index]['pass@1'] = 1
                    results_list[index]['generated_answer'] = candidate_voted_answer
                    final_result_single = results_list[index] 
                    print('voting_bon can get a candidate answer')
                else:
                    print('voting_bon does not work or index out of bounds')
                    import random
                    if results_list: # Ensure results_list is not empty
                        random_index = random.randint(0, len(results_list)-1)
                        final_result_single = results_list[random_index]
                    else: # Handle case where results_list is empty after loop
                         final_result_single = None
            else: # Handle case where no results were generated in the loop
                final_result_single = None


        if final_result_single is not None:
            # Use a lock to safely write to the file
            with lock:
                with open(result_file, 'a', encoding='utf-8') as f:
                    json.dump(final_result_single, f, ensure_ascii=False)
                    f.write('\n')
            return final_result_single
    except Exception as e:
        logger = logging.getLogger("math_solver") # Define logger for exception block
        logger.exception(f"Error in worker process: {str(e)}")
    return None


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Math problem solver with multiprocessing')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to use')
    parser.add_argument('--debug', default=False, help='Debug mode with fewer problems')
    parser.add_argument('--output', type=str, default='result_flexive_seq_debug_verification.json', help='Output file path')
    parser.add_argument('--model_name', type=str, help='Model name to use')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer')
    parser.add_argument('--apply_detection', action='store_true', help='Apply hesitation detection')
    parser.add_argument('--apply_verification', action='store_true', help='Apply solution verification')
    parser.add_argument('--limit', type=int, default=2, help='Number of problems to process in debug mode')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help='Set the logging level')
    parser.add_argument('--flexive_model_name', type=str, help='Model name for FlexiVe verification')
    parser.add_argument('--flexive_base_url', type=str, help='Base URL for FlexiVe API')
    # Adaptive verification arguments
    parser.add_argument('--adaptive_verification', action='store_true', 
                       help='Enable adaptive verification scheme')
    parser.add_argument('--verification_k', type=int, default=4,
                       help='Number of fast verifications to run initially')
    parser.add_argument('--problems_path', type=str, default="./aime_2024.jsonl",
                       help='Number of fast verifications to run initially') #This help message seems duplicated/misplaced
    parser.add_argument('--majority_voting_n', type=int, default=1,
                        )
    parser.add_argument('--verification_threshold', type=float, default=0.8,
                       help='Agreement threshold (0.0-1.0) to determine if slow thinking is needed')
    parser.add_argument('--lm_base_urls', type=str, nargs='+', 
            default=["http://localhost:8001/v1", "http://localhost:8002/v1", "http://localhost:8003/v1", "http://localhost:8004/v1","http://localhost:8005/v1","http://localhost:8006/v1"],
                        help='List of base URLs for LM models to pick from')
    parser.add_argument('--best_of_n', type=int, default=1,
                       help='Number of solutions to generate in second round')
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    # Parse arguments first to get log level before setting up logger
    args = parse_arguments()
    
    # Set up global logger for the main process
    logger = setup_logger(args.log_level)
    logger.info(f"Starting math problem solver with args: {args}")
    
    config = Config.from_args(args)
    
    # Ensure lm_base_urls is properly set from args
    if hasattr(args, 'lm_base_urls') and args.lm_base_urls:
        config.lm_base_urls = args.lm_base_urls
        logger.info(f"Using multiple LM models at: {config.lm_base_urls}")
    
    # Log full configuration
    logger.info(f"Configuration: {config}")
    
    # Load problems
    problems = load_problems(config.problems_path)
    problems = problems[20:]
    if args.debug:
        problems = problems[-args.limit:]
        logger.info(f"DEBUG MODE: processing only the last {len(problems)} problems")
    
    # Set the detection and verification flags
    apply_detection = args.apply_detection if hasattr(args, 'apply_detection') else True
    apply_verification = args.apply_verification if hasattr(args, 'apply_verification') else False
    
    # Log verification settings
    if apply_verification:
        if config.adaptive_verification:
            logger.info(f"Using ADAPTIVE verification with k={config.verification_k}, threshold={config.verification_threshold}")
        else:
            logger.info("Using standard verification (single run)")
    else:
        logger.info("Verification is disabled")
        
    logger.info(f"Processing with detection={apply_detection}, verification={apply_verification}")
    
    # Record the start time for overall process
    start_time = time.time()
    
    # Create or clear the output file
    with open(args.output, 'w', encoding='utf-8') as f:
        pass
    logger.info(f"Created output file: {args.output}")
    
    # Create a lock for file access synchronization
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    
    # Convert Config to a dictionary for pickling
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')} # Ensure internal fields are not included

    
    # Create a partial function with fixed arguments
    process_fn = partial(process_problem, 
                         config_dict=config_dict,
                         apply_detection=apply_detection, 
                         apply_verification=apply_verification, 
                         majority_voting_n = args.majority_voting_n,
                         result_file=args.output, 
                         lock=lock)
    
    # Use a Pool of workers to process problems in parallel
    logger.info(f"Starting processing with {args.num_processes} processes and {len(config.lm_base_urls)} LM models")
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        # Process problems and collect non-None results
        results = list(tqdm(
            pool.imap(process_fn, problems),
            total=len(problems),
            desc='Processing problems'
        ))
        results = [r for r in results if r is not None]
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    
    # Print summary statistics
    pass_at_1_rate = 0 # Initialize
    avg_solving_time = 0 # Initialize
    if results:
        total_pass_at_1 = sum(result["pass@1"] for result in results if "pass@1" in result) # ensure key exists
        if len(results) > 0 : # Avoid division by zero
             pass_at_1_rate = total_pass_at_1 / len(results) * 100
        logger.info(f"Final Pass@1 Rate: {pass_at_1_rate:.2f}%")
        print(f"\nFinal Pass@1 Rate: {pass_at_1_rate:.2f}%")
        
        # Calculate average solving time
        if len(results) > 0: # Avoid division by zero
            avg_solving_time = sum(result.get("solving_time", 0) for result in results) / len(results)
        logger.info(f"Average solving time per problem: {avg_solving_time:.2f} seconds")
        print(f"Average solving time per problem: {avg_solving_time:.2f} seconds")
        

    logger.info(f"Evaluation complete. Processed {len(results)} problems successfully.")
    print(f"Evaluation complete. Processed {len(results)} problems successfully.")
    
    dataset_name = 'UnknownDataset' # Default
    if '2024' in args.problems_path:
        dataset_name = 'AIME2024'
    elif '2025' in args.problems_path:
        dataset_name = 'AIME2025'
    elif 'AMC' in args.problems_path:
        dataset_name = 'AMC2023'
    elif 'OlympiadBench' in args.problems_path:
        dataset_name = 'OlympiadBench'
    
    # Save summary stats to a separate file
    summary_file_name_parts = ["summary", "7B"]
    if apply_detection:
        summary_file_name_parts.append("apply_detection")
    if apply_verification:
        summary_file_name_parts.append("apply_verification")
    if not apply_detection and not apply_verification:
        summary_file_name_parts.append("no_detection_no_verification")

    summary_file_name_parts.append(dataset_name)
    summary_file = "_".join(summary_file_name_parts) + ".json"
    
    summary = {
        "total_problems": len(problems),
        "successful_problems": len(results),
        "pass_at_1_rate": pass_at_1_rate,
        "avg_solving_time": avg_solving_time,
        "total_elapsed_time": elapsed_time,
        "config": config_dict
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary statistics saved to {summary_file}")


if __name__ == "__main__":
    # This is required for Windows support
    multiprocessing.freeze_support()
    main()
