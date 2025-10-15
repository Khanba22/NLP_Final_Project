# app/services/summarizer.py
import torch
import string
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from app.core.config import settings

class SummarizerService:
    def __init__(self):
        self.model_name = settings.MODEL_NAME
        self.min_length = settings.MIN_SUMMARY_LENGTH
        self.max_length = settings.MAX_SUMMARY_LENGTH
        self.model, self.tokenizer = self._load_model()
        self.abstractive_pipeline = self._create_pipeline()
        self.stop_words = set(stopwords.words('english'))

    def _load_model(self):
        """Loads the pre-trained model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        return model, tokenizer

    def _create_pipeline(self):
        """Creates the summarization pipeline."""
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )

    def abstractive_summarize(self, text: str) -> str:
        """
        Generates an abstractive summary.
        Handles long texts by splitting them into chunks.
        """
        if not text or len(text.strip()) == 0:
            return "No text provided for summarization."
        
        # Use tokenizer to get actual token count instead of word count
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        max_input_length = self.tokenizer.model_max_length - 50  # Leave room for special tokens
        
        # Adjust chunk length based on model (typically 1024 for BART, 512 for T5-small)
        max_chunk_length = min(max_input_length, 1024)
        
        if len(tokens) <= max_chunk_length:
            try:
                # Calculate dynamic max_length based on input
                # For longer summaries, use 30-40% of input length or configured max
                input_length = len(tokens)
                dynamic_max_length = min(
                    max(self.max_length, int(input_length * 0.4)),
                    self.tokenizer.model_max_length // 2
                )
                dynamic_min_length = min(self.min_length, dynamic_max_length - 10)
                
                result = self.abstractive_pipeline(
                    text,
                    max_length=dynamic_max_length,
                    min_length=dynamic_min_length,
                    do_sample=False,
                    truncation=True
                )
                
                # Safely extract summary
                if result and len(result) > 0 and 'summary_text' in result[0]:
                    return result[0]['summary_text']
                else:
                    return "Error: Unable to generate summary."
                    
            except Exception as e:
                print(f"Error in abstractive summarization: {e}")
                return f"Error generating summary: {str(e)}"
        
        # Handle long texts by chunking based on sentences
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            
            if current_length + len(sentence_tokens) <= max_chunk_length:
                current_chunk.append(sentence)
                current_length += len(sentence_tokens)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence_tokens)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                chunk_tokens = len(self.tokenizer.encode(chunk, add_special_tokens=False))
                chunk_max_length = min(
                    max(self.max_length, int(chunk_tokens * 0.4)),
                    self.tokenizer.model_max_length // 2
                )
                chunk_min_length = min(self.min_length, chunk_max_length - 10)
                
                chunk_summary = self.abstractive_pipeline(
                    chunk,
                    max_length=chunk_max_length,
                    min_length=chunk_min_length,
                    do_sample=False,
                    truncation=True
                )
                
                if chunk_summary and len(chunk_summary) > 0 and 'summary_text' in chunk_summary[0]:
                    chunk_summaries.append(chunk_summary[0]['summary_text'])
                    
            except Exception as e:
                print(f"Error summarizing chunk {i}: {e}")
                continue
        
        if not chunk_summaries:
            return "Error: Unable to generate summary from chunks."
        
        combined_summary = " ".join(chunk_summaries)
        
        # If combined summary is still too long, summarize it again
        combined_tokens = self.tokenizer.encode(combined_summary, add_special_tokens=False)
        if len(combined_tokens) > max_chunk_length:
            try:
                final_max_length = min(
                    max(self.max_length, int(len(combined_tokens) * 0.5)),
                    self.tokenizer.model_max_length // 2
                )
                final_min_length = min(self.min_length, final_max_length - 10)
                
                final_summary = self.abstractive_pipeline(
                    combined_summary,
                    max_length=final_max_length,
                    min_length=final_min_length,
                    do_sample=False,
                    truncation=True
                )
                
                if final_summary and len(final_summary) > 0 and 'summary_text' in final_summary[0]:
                    return final_summary[0]['summary_text']
                    
            except Exception as e:
                print(f"Error in final summarization: {e}")
                return combined_summary

        return combined_summary


    def extractive_summarize(self, text: str, num_sentences: int = 10) -> str:
        """
        Generates an extractive summary using sentence scoring.
        """
        if not text or len(text.strip()) == 0:
            return "No text provided for summarization."
            
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        words = word_tokenize(text.lower())
        filtered_words = [
            word for word in words 
            if word.isalnum() and word not in self.stop_words and word not in string.punctuation
        ]
        
        word_freq = Counter(filtered_words)
        
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence.lower())
            score = sum(word_freq[word] for word in sentence_words if word in word_freq)
            
            # Normalize for sentence length
            if len(sentence_words) > 0:
                sentence_scores[i] = score / len(sentence_words)
            else:
                sentence_scores[i] = 0

        # Get top sentences and sort them by their original order
        top_sentence_indices = sorted(
            sentence_scores, 
            key=sentence_scores.get, 
            reverse=True
        )[:num_sentences]
        sorted_indices = sorted(top_sentence_indices)
        
        summary = ' '.join([sentences[i] for i in sorted_indices])
        return summary

    def compare_summaries(self, summaries: list) -> dict:
        """
        Compare multiple summaries to find common themes and unique points
        """
        # Extract keywords from each summary
        all_keywords = []
        summary_keywords = []
        
        for summary in summaries:
            words = word_tokenize(summary.lower())
            filtered = [
                word for word in words 
                if word.isalnum() and word not in self.stop_words and len(word) > 3
            ]
            freq = Counter(filtered)
            top_keywords = [word for word, _ in freq.most_common(10)]
            summary_keywords.append(set(top_keywords))
            all_keywords.extend(top_keywords)
        
        # Find common themes (keywords appearing in multiple summaries)
        keyword_freq = Counter(all_keywords)
        common_themes = [
            word for word, count in keyword_freq.most_common(10) 
            if count >= 2  # Appears in at least 2 documents
        ]
        
        # Find unique points for each document
        unique_points = {}
        for i, keywords in enumerate(summary_keywords):
            # Keywords unique to this document or appearing in only this and one other
            unique = []
            for keyword in keywords:
                if keyword_freq[keyword] <= 2 and keyword not in common_themes:
                    unique.append(keyword)
            unique_points[f"Document {i+1}"] = unique[:5]  # Top 5 unique keywords
        
        return {
            "common_themes": common_themes[:8],  # Top 8 common themes
            "unique_points": unique_points
        }