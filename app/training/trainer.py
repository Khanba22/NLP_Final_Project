# app/training/trainer.py
"""
Training module for fine-tuning summarization models on custom datasets
"""

import os
from typing import Dict, List, Optional, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import json
from pathlib import Path


class SummarizationTrainer:
    """
    Trainer for fine-tuning summarization models on technical product data
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        output_dir: str = "./trained_models"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.training_args = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def prepare_model(self):
        """Load and prepare the model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        print("Model loaded successfully")
    
    def prepare_dataset(
        self,
        training_data: List[Dict[str, Any]],
        max_input_length: int = 1024,
        max_target_length: int = 256
    ) -> Dataset:
        """
        Prepare dataset for training
        
        Args:
            training_data: List of dicts with 'full_description' and 'target_summary'
            max_input_length: Maximum length for input sequences
            max_target_length: Maximum length for target sequences
            
        Returns:
            Prepared HuggingFace Dataset
        """
        # Convert target_summary dict to JSON string for structured output
        processed_data = []
        for item in training_data:
            target_text = item['target_summary']
            if isinstance(target_text, dict):
                # Convert structured summary to formatted text
                target_text = self._format_structured_summary(target_text)
            
            processed_data.append({
                'input_text': item['full_description'],
                'target_text': target_text
            })
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(processed_data)
        
        # Tokenize the dataset
        def tokenize_function(examples):
            # Tokenize inputs
            model_inputs = self.tokenizer(
                examples['input_text'],
                max_length=max_input_length,
                truncation=True,
                padding='max_length'
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                examples['target_text'],
                max_length=max_target_length,
                truncation=True,
                padding='max_length'
            )
            
            model_inputs['labels'] = labels['input_ids']
            return model_inputs
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def _format_structured_summary(self, summary_dict: Dict[str, Any]) -> str:
        """
        Format structured summary dictionary into a training-friendly text format
        """
        formatted_parts = []
        
        if 'product_name' in summary_dict:
            formatted_parts.append(f"Product: {summary_dict['product_name']}")
        
        if 'category' in summary_dict:
            formatted_parts.append(f"Category: {summary_dict['category']}")
        
        if 'key_specs' in summary_dict and isinstance(summary_dict['key_specs'], dict):
            specs_text = ', '.join([f"{k}: {v}" for k, v in summary_dict['key_specs'].items()])
            formatted_parts.append(f"Specifications: {specs_text}")
        
        if 'pros' in summary_dict and isinstance(summary_dict['pros'], list):
            pros_text = '; '.join(summary_dict['pros'])
            formatted_parts.append(f"Advantages: {pros_text}")
        
        if 'cons' in summary_dict and isinstance(summary_dict['cons'], list):
            cons_text = '; '.join(summary_dict['cons'])
            formatted_parts.append(f"Disadvantages: {cons_text}")
        
        if 'best_for' in summary_dict:
            formatted_parts.append(f"Best for: {summary_dict['best_for']}")
        
        if 'price_range' in summary_dict:
            formatted_parts.append(f"Price: {summary_dict['price_range']}")
        
        return '. '.join(formatted_parts) + '.'
    
    def setup_training_args(
        self,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        learning_rate: float = 2e-5,
        save_steps: int = 500,
        eval_steps: int = 500,
        warmup_steps: int = 100
    ) -> Seq2SeqTrainingArguments:
        """
        Configure training arguments
        """
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            save_total_limit=3,
            save_steps=save_steps,
            eval_steps=eval_steps,
            warmup_steps=warmup_steps,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            push_to_hub=False,
        )
        
        return self.training_args
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            
        Returns:
            Training metrics
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")
        
        if self.training_args is None:
            self.setup_training_args()
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting training...")
        train_result = trainer.train()
        
        # Save the final model
        print(f"Saving model to {self.output_dir}/final")
        trainer.save_model(f"{self.output_dir}/final")
        self.tokenizer.save_pretrained(f"{self.output_dir}/final")
        
        # Save training metrics
        metrics = train_result.metrics
        metrics_path = f"{self.output_dir}/training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("Training completed!")
        print(f"Metrics: {metrics}")
        
        return metrics
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate the model on a dataset
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Evaluation metrics
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")
        
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args or self.setup_training_args(),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        metrics = trainer.evaluate(eval_dataset)
        return metrics
    
    def generate_summary(
        self,
        text: str,
        max_length: int = 256,
        min_length: int = 50
    ) -> str:
        """
        Generate a summary using the trained model
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Generated summary
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")
        
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        summary_ids = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


class TrainingPipeline:
    """
    End-to-end training pipeline for technical product summarization
    """
    
    def __init__(self, dataset, model_name: str = "facebook/bart-large-cnn"):
        self.dataset = dataset
        self.trainer = SummarizationTrainer(model_name=model_name)
    
    def run_training(
        self,
        num_epochs: int = 3,
        batch_size: int = 2,
        train_test_split: float = 0.8
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            train_test_split: Ratio of train/test split
            
        Returns:
            Training results and metrics
        """
        print("=" * 50)
        print("Technical Product Summarizer Training Pipeline")
        print("=" * 50)
        
        # Step 1: Prepare model
        print("\n[1/5] Preparing model...")
        self.trainer.prepare_model()
        
        # Step 2: Prepare dataset
        print("\n[2/5] Preparing dataset...")
        training_pairs = self.dataset.get_training_pairs()
        
        # Split into train and eval
        split_idx = int(len(training_pairs) * train_test_split)
        train_data = [
            {'full_description': desc, 'target_summary': summary}
            for desc, summary in training_pairs[:split_idx]
        ]
        eval_data = [
            {'full_description': desc, 'target_summary': summary}
            for desc, summary in training_pairs[split_idx:]
        ]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Evaluation samples: {len(eval_data)}")
        
        train_dataset = self.trainer.prepare_dataset(train_data)
        eval_dataset = self.trainer.prepare_dataset(eval_data) if eval_data else None
        
        # Step 3: Setup training arguments
        print("\n[3/5] Configuring training...")
        self.trainer.setup_training_args(
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size
        )
        
        # Step 4: Train
        print("\n[4/5] Training model...")
        metrics = self.trainer.train(train_dataset, eval_dataset)
        
        # Step 5: Evaluate
        print("\n[5/5] Final evaluation...")
        if eval_dataset:
            eval_metrics = self.trainer.evaluate(eval_dataset)
            print(f"Evaluation metrics: {eval_metrics}")
        
        results = {
            "training_metrics": metrics,
            "model_path": f"{self.trainer.output_dir}/final",
            "num_train_samples": len(train_data),
            "num_eval_samples": len(eval_data)
        }
        
        print("\n" + "=" * 50)
        print("Training pipeline completed!")
        print("=" * 50)
        
        return results

