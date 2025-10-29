# app/evaluation/metrics.py
"""
Evaluation metrics for summarization quality
Includes ROUGE, BLEU, BARTScore, and custom metrics for structured output
"""

from typing import Dict, List, Any
import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
import torch
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. BARTScore will not be calculated.")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class SummarizationEvaluator:
    """
    Evaluates summarization quality using multiple metrics
    Includes ROUGE, BLEU, and BARTScore metrics
    """
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction()
        
        # Initialize BARTScore components if available
        self.bart_tokenizer = None
        self.bart_model = None
        self.bart_device = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                model_name = 'facebook/bart-large-cnn'
                print(f"Loading BARTScore model: {model_name}")
                self.bart_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.bart_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
                # Set device
                if torch.cuda.is_available():
                    self.bart_device = "cuda"
                    self.bart_model = self.bart_model.to(self.bart_device)
                else:
                    self.bart_device = "cpu"
                
                self.bart_model.eval()  # Set to evaluation mode
                print("BARTScore model loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize BARTScore: {e}")
                self.bart_tokenizer = None
                self.bart_model = None
    
    def evaluate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        
        Args:
            generated: Generated summary text
            reference: Reference summary text
            
        Returns:
            Dictionary with ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, generated)
        
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_fmeasure': scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rouge2_fmeasure': scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_fmeasure': scores['rougeL'].fmeasure,
        }
    
    def evaluate_bleu(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate BLEU score
        
        Args:
            generated: Generated summary text
            reference: Reference summary text
            
        Returns:
            Dictionary with BLEU scores
        """
        generated_tokens = word_tokenize(generated.lower())
        reference_tokens = word_tokenize(reference.lower())
        
        # Calculate BLEU scores with different n-gram weights
        bleu1 = sentence_bleu(
            [reference_tokens], 
            generated_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=self.smoothing.method1
        )
        
        bleu2 = sentence_bleu(
            [reference_tokens], 
            generated_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=self.smoothing.method1
        )
        
        bleu4 = sentence_bleu(
            [reference_tokens], 
            generated_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothing.method1
        )
        
        return {
            'bleu1': bleu1,
            'bleu2': bleu2,
            'bleu4': bleu4
        }
    
    def evaluate_bart(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate BARTScore using direct transformers implementation
        BARTScore computes the log-likelihood of generating the reference given the generated text
        
        Args:
            generated: Generated summary text
            reference: Reference summary text
            
        Returns:
            Dictionary with BARTScore
        """
        if not TRANSFORMERS_AVAILABLE or self.bart_model is None or self.bart_tokenizer is None:
            return {'bart_score': 0.0}
        
        try:
            # Tokenize inputs (source = generated, target = reference)
            # BARTScore: P(reference | generated)
            source = self.bart_tokenizer(
                generated,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False
            )
            target = self.bart_tokenizer(
                reference,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False
            )

            source = {k: v.to(self.bart_device) for k, v in source.items()}
            target_ids = target["input_ids"].to(self.bart_device)

            # Calculate negative log-likelihood loss for target given source
            with torch.no_grad():
                outputs = self.bart_model(
                    input_ids=source["input_ids"],
                    attention_mask=source.get("attention_mask"),
                    labels=target_ids,
                    return_dict=True
                )
                loss = outputs.loss  # averaged over tokens (ignores -100)

            # Define BARTScore as negative loss (higher is better)
            bart_score = float(-loss.item())
            return {'bart_score': bart_score}
        except Exception as e:
            print(f"Error calculating BARTScore: {e}")
            import traceback
            traceback.print_exc()
            return {'bart_score': 0.0}
    
    def evaluate_structured_output(
        self, 
        generated: Dict[str, Any], 
        reference: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate structured output quality by comparing fields
        
        Args:
            generated: Generated structured summary
            reference: Reference structured summary
            
        Returns:
            Dictionary with field-level accuracy scores
        """
        scores = {}
        
        # Check if key fields are present
        required_fields = ['product_name', 'category', 'key_specs', 'pros', 'cons', 'best_for']
        fields_present = sum(1 for field in required_fields if field in generated)
        scores['field_completeness'] = fields_present / len(required_fields)
        
        # Check product name match
        if 'product_name' in generated and 'product_name' in reference:
            scores['product_name_match'] = 1.0 if generated['product_name'] == reference['product_name'] else 0.0
        
        # Check category match
        if 'category' in generated and 'category' in reference:
            scores['category_match'] = 1.0 if generated['category'] == reference['category'] else 0.0
        
        # Check key_specs field match rate
        if 'key_specs' in generated and 'key_specs' in reference:
            ref_specs = set(reference['key_specs'].keys())
            gen_specs = set(generated['key_specs'].keys())
            if ref_specs:
                scores['specs_field_coverage'] = len(gen_specs & ref_specs) / len(ref_specs)
            else:
                scores['specs_field_coverage'] = 0.0
        
        # Check pros/cons count similarity
        if 'pros' in generated and 'pros' in reference:
            ref_pros_count = len(reference['pros'])
            gen_pros_count = len(generated['pros'])
            if ref_pros_count > 0:
                scores['pros_count_ratio'] = min(gen_pros_count / ref_pros_count, 1.0)
            else:
                scores['pros_count_ratio'] = 0.0
        
        if 'cons' in generated and 'cons' in reference:
            ref_cons_count = len(reference['cons'])
            gen_cons_count = len(generated['cons'])
            if ref_cons_count > 0:
                scores['cons_count_ratio'] = min(gen_cons_count / ref_cons_count, 1.0)
            else:
                scores['cons_count_ratio'] = 0.0
        
        # Calculate overall score
        if scores:
            scores['overall_structural_score'] = sum(scores.values()) / len(scores)
        else:
            scores['overall_structural_score'] = 0.0
        
        return scores
    
    def evaluate_text_summary(self, generated: str, reference: str) -> Dict[str, Any]:
        """
        Comprehensive evaluation combining ROUGE, BLEU, and BARTScore
        
        Args:
            generated: Generated summary text
            reference: Reference summary text
            
        Returns:
            Dictionary with all metric scores (ROUGE, BLEU, BARTScore, and overall score)
        """
        rouge_scores = self.evaluate_rouge(generated, reference)
        bleu_scores = self.evaluate_bleu(generated, reference)
        bart_scores = self.evaluate_bart(generated, reference)
        
        # Combine all scores
        all_scores = {**rouge_scores, **bleu_scores, **bart_scores}
        
        # Calculate overall score (weighted average)
        # Normalize BARTScore if available (BARTScore can be negative, we'll normalize it)
        overall_score = (
            rouge_scores['rouge1_fmeasure'] * 0.25 +
            rouge_scores['rouge2_fmeasure'] * 0.25 +
            rouge_scores['rougeL_fmeasure'] * 0.2 +
            bleu_scores['bleu4'] * 0.15
        )
        
        # Add BARTScore if present (normalize to 0-1 range if negative values exist)
        if 'bart_score' in bart_scores:
            bart_score = bart_scores['bart_score']
            # Normalize BARTScore (typically ranges from -inf to 0, we'll use sigmoid-like normalization)
            # Or simply add it with a weight if it's already normalized
            # Assuming BARTScore is typically negative, we'll use exp to normalize
            if bart_score < 0:
                # Normalize negative BARTScore to 0-1 range using exponential
                normalized_bart = 1 / (1 + abs(bart_score))
            else:
                # If positive, use as is but cap at 1.0
                normalized_bart = min(bart_score, 1.0)
            
            # Update weights to include BARTScore
            overall_score = (
                rouge_scores['rouge1_fmeasure'] * 0.2 +
                rouge_scores['rouge2_fmeasure'] * 0.2 +
                rouge_scores['rougeL_fmeasure'] * 0.15 +
                bleu_scores['bleu4'] * 0.15 +
                normalized_bart * 0.3
            )
        else:
            # Revert to original weights if BARTScore not available
            overall_score = (
                rouge_scores['rouge1_fmeasure'] * 0.3 +
                rouge_scores['rouge2_fmeasure'] * 0.3 +
                rouge_scores['rougeL_fmeasure'] * 0.2 +
                bleu_scores['bleu4'] * 0.2
            )
        
        all_scores['overall_score'] = overall_score
        
        return all_scores
    
    def evaluate_batch(
        self, 
        generated_summaries: List[str], 
        reference_summaries: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate multiple summaries and return average scores
        
        Args:
            generated_summaries: List of generated summaries
            reference_summaries: List of reference summaries
            
        Returns:
            Dictionary with averaged metric scores
        """
        if len(generated_summaries) != len(reference_summaries):
            raise ValueError("Number of generated and reference summaries must match")
        
        all_scores = []
        for gen, ref in zip(generated_summaries, reference_summaries):
            scores = self.evaluate_text_summary(gen, ref)
            all_scores.append(scores)
        
        # Calculate averages
        avg_scores = {}
        if all_scores:
            keys = all_scores[0].keys()
            for key in keys:
                avg_scores[f'avg_{key}'] = sum(s[key] for s in all_scores) / len(all_scores)
        
        return avg_scores


def format_evaluation_report(scores: Dict[str, float]) -> str:
    """
    Format evaluation scores into a readable report
    
    Args:
        scores: Dictionary of metric scores
        
    Returns:
        Formatted string report
    """
    report = "📊 Evaluation Report\n"
    report += "=" * 50 + "\n\n"
    
    if 'rouge1_fmeasure' in scores:
        report += "ROUGE Scores:\n"
        report += f"  ROUGE-1 F1: {scores['rouge1_fmeasure']:.4f}\n"
        report += f"  ROUGE-2 F1: {scores['rouge2_fmeasure']:.4f}\n"
        report += f"  ROUGE-L F1: {scores['rougeL_fmeasure']:.4f}\n\n"
    
    if 'bleu4' in scores:
        report += "BLEU Scores:\n"
        report += f"  BLEU-1: {scores['bleu1']:.4f}\n"
        report += f"  BLEU-2: {scores['bleu2']:.4f}\n"
        report += f"  BLEU-4: {scores['bleu4']:.4f}\n\n"
    
    if 'bart_score' in scores:
        report += "BART Score:\n"
        report += f"  BARTScore: {scores['bart_score']:.4f}\n\n"
    
    if 'overall_score' in scores:
        report += f"Overall Score: {scores['overall_score']:.4f}\n"
    
    if 'overall_structural_score' in scores:
        report += f"\nStructural Quality Score: {scores['overall_structural_score']:.4f}\n"
    
    return report

