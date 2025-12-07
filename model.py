"""
📌 TURNI-LAB COMPLETE SYSTEM - Local Training & Deployment
Enhanced with advanced features for realistic AI detection simulation and bypass research
"""

import nltk
import numpy as np
import random
import torch
import json
import pickle
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

# Optional: lexicalrichness - if not available, we'll implement basic version
try:
    from lexicalrichness import LexicalRichness
    HAS_LEXRICH = True
except ImportError:
    HAS_LEXRICH = False
    print("Note: lexicalrichness not installed. Using basic lexical diversity metrics.")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Optional: transformers for advanced perplexity
try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Note: transformers not installed. Using basic perplexity approximation.")

# --------------------------- MODULE 1: Enhanced Linguistic Pattern Analyzer ---------------------------

class LinguisticAnalyzer:
    def __init__(self, language: str = 'english'):
        self.language = language
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
    def sentence_stats(self, text: str) -> Dict[str, float]:
        """Calculate comprehensive sentence statistics"""
        sentences = nltk.sent_tokenize(text, language=self.language)
        if not sentences:
            return {
                "sentence_count": 0,
                "avg_sentence_length": 0,
                "sentence_length_std": 0,
                "max_sentence_length": 0,
                "min_sentence_length": 0,
                "sentence_length_cv": 0  # Coefficient of variation
            }
        
        lengths = [len(s.split()) for s in sentences]
        words_per_sentence = lengths
        
        return {
            "sentence_count": len(sentences),
            "avg_sentence_length": np.mean(lengths),
            "sentence_length_std": np.std(lengths),
            "max_sentence_length": np.max(lengths),
            "min_sentence_length": np.min(lengths),
            "sentence_length_cv": np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0,
            "words_per_sentence": words_per_sentence
        }
    
    def lexical_diversity(self, text: str) -> Dict[str, float]:
        """Calculate lexical diversity metrics"""
        words = nltk.word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        if not words:
            return {
                "ttr": 0,
                "msttr": 0,
                "hdd": 0,
                "mattr": 0,
                "vocd": 0,
                "unique_words": 0,
                "total_words": 0
            }
        
        total_words = len(words)
        unique_words = len(set(words))
        
        # Type-Token Ratio
        ttr = unique_words / total_words if total_words > 0 else 0
        
        # Moving Average TTR (simplified)
        def calculate_msttr(words, window=25):
            if len(words) <= window:
                return ttr
            segments = [words[i:i+window] for i in range(0, len(words), window)]
            segment_ttrs = [len(set(seg)) / len(seg) for seg in segments if seg]
            return np.mean(segment_ttrs) if segment_ttrs else 0
        
        msttr = calculate_msttr(words)
        
        # HD-D (simplified)
        def calculate_hdd(words, sample_size=42):
            if len(words) < sample_size:
                return 0
            sample = random.sample(words, min(sample_size, len(words)))
            unique_in_sample = len(set(sample))
            return unique_in_sample / len(sample)
        
        hdd = calculate_hdd(words)
        
        # Mean Segmental TTR (simplified)
        mattr = msttr  # Approximation
        
        # Vocd (simplified - uses curve fitting approximation)
        def calculate_vocd(words):
            # Simplified version using TTR curve
            if len(words) < 100:
                return ttr * 100
            # Calculate TTR at different token counts
            points = []
            for i in range(1, min(100, len(words)), 10):
                segment = words[:i]
                points.append(len(set(segment)) / i)
            return np.mean(points) * 100 if points else 0
        
        vocd = calculate_vocd(words)
        
        return {
            "ttr": ttr,
            "msttr": msttr,
            "hdd": hdd,
            "mattr": mattr,
            "vocd": vocd,
            "unique_words": unique_words,
            "total_words": total_words,
            "lexical_density": sum(1 for w in words if w not in self.stopwords) / total_words
        }
    
    def syntactic_complexity(self, text: str) -> Dict[str, float]:
        """Analyze syntactic complexity"""
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return {
                "avg_clauses_per_sentence": 0,
                "avg_tree_depth": 0,
                "coordination_ratio": 0,
                "subordination_ratio": 0
            }
        
        clauses_per_sentence = []
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            pos_tags = nltk.pos_tag(words)
            # Simple clause detection (based on verbs)
            verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
            clauses_per_sentence.append(max(1, verb_count))
        
        return {
            "avg_clauses_per_sentence": np.mean(clauses_per_sentence),
            "clause_variation": np.std(clauses_per_sentence),
            "coordination_ratio": sum(1 for sent in sentences if ' and ' in sent.lower() or ' but ' in sent.lower()) / len(sentences),
            "subordination_ratio": sum(1 for sent in sentences if any(word in sent.lower() for word in ['that', 'which', 'who', 'when', 'where', 'because', 'although'])) / len(sentences)
        }
    
    def punctuation_analysis(self, text: str) -> Dict[str, float]:
        """Analyze punctuation patterns"""
        total_chars = len(text)
        if total_chars == 0:
            return {
                "comma_frequency": 0,
                "period_frequency": 0,
                "question_frequency": 0,
                "exclamation_frequency": 0,
                "colon_frequency": 0,
                "semicolon_frequency": 0,
                "punctuation_variety": 0
            }
        
        punct_counts = {
            ',': text.count(','),
            '.': text.count('.'),
            '?': text.count('?'),
            '!': text.count('!'),
            ':': text.count(':'),
            ';': text.count(';'),
            '-': text.count('-'),
            '(': text.count('('),
            ')': text.count(')'),
            '"': text.count('"'),
            "'": text.count("'")
        }
        
        # Normalize by 1000 characters
        punct_freq = {f"{k}_frequency": (v / total_chars) * 1000 for k, v in punct_counts.items()}
        
        # Punctuation variety (unique punctuation types)
        used_punct = sum(1 for v in punct_counts.values() if v > 0)
        punct_freq["punctuation_variety"] = used_punct
        
        return punct_freq
    
    def run(self, text: str) -> Dict[str, Any]:
        """Run complete linguistic analysis"""
        return {
            "sentence_stats": self.sentence_stats(text),
            "lexical_diversity": self.lexical_diversity(text),
            "syntactic_complexity": self.syntactic_complexity(text),
            "punctuation_analysis": self.punctuation_analysis(text)
        }


# --------------------------- MODULE 2: Enhanced Perplexity/Burstiness Engine ---------------------------

class PerplexityEngine:
    def __init__(self, model_name: str = "gpt2", use_simple: bool = False):
        self.use_simple = use_simple or not HAS_TRANSFORMERS
        
        if not self.use_simple:
            try:
                self.model = GPT2LMHeadModel.from_pretrained(model_name)
                self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
                self.model.eval()
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
            except Exception as e:
                print(f"Could not load transformers model: {e}. Using simple perplexity.")
                self.use_simple = True
        
        # For simple perplexity calculation
        self.word_freq_cache = {}
    
    def _simple_perplexity(self, text: str) -> float:
        """Simple perplexity approximation using word frequencies"""
        words = nltk.word_tokenize(text.lower())
        if len(words) < 2:
            return 1.0
        
        # Build simple language model (unigram)
        if not self.word_freq_cache:
            # Initialize with some common words
            common_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i']
            for word in common_words:
                self.word_freq_cache[word] = 1.0
        
        # Calculate perplexity
        log_prob_sum = 0
        for word in words:
            prob = self.word_freq_cache.get(word, 0.0001)  # Small probability for unknown words
            log_prob_sum += np.log(prob)
        
        avg_log_prob = log_prob_sum / len(words)
        perplexity = np.exp(-avg_log_prob)
        return perplexity
    
    def calc_perplexity(self, text: str) -> float:
        """Calculate perplexity of text"""
        if self.use_simple:
            return self._simple_perplexity(text)
        
        try:
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            
            if torch.cuda.is_available():
                tokens = {k: v.cuda() for k, v in tokens.items()}
            
            with torch.no_grad():
                outputs = self.model(**tokens, labels=tokens["input_ids"])
                loss = outputs.loss
            
            if torch.cuda.is_available():
                loss = loss.cpu()
            
            perplexity = torch.exp(loss).item()
            return perplexity
            
        except Exception as e:
            print(f"Error calculating perplexity with GPT-2: {e}. Falling back to simple method.")
            return self._simple_perplexity(text)
    
    def burstiness(self, text: str) -> Dict[str, float]:
        """Calculate burstiness metrics"""
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return {
                "mean_ppl": self.calc_perplexity(text),
                "std_ppl": 0,
                "burstiness_score": 0,
                "sentence_count": len(sentences)
            }
        
        # Calculate perplexity for each sentence
        ppls = []
        for sent in sentences:
            if len(sent.strip()) > 10:  # Only calculate for meaningful sentences
                ppl = self.calc_perplexity(sent)
                ppls.append(ppl)
        
        if not ppls:
            return {
                "mean_ppl": 0,
                "std_ppl": 0,
                "burstiness_score": 0,
                "sentence_count": 0
            }
        
        # Calculate burstiness score (normalized std)
        mean_ppl = np.mean(ppls)
        std_ppl = np.std(ppls)
        burstiness_score = (std_ppl - mean_ppl) / (std_ppl + mean_ppl) if (std_ppl + mean_ppl) > 0 else 0
        
        return {
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
            "burstiness_score": burstiness_score,
            "sentence_count": len(sentences),
            "perplexity_values": ppls
        }
    
    def analyze_consistency(self, text: str, window_size: int = 100) -> Dict[str, float]:
        """Analyze consistency of perplexity across text windows"""
        words = text.split()
        if len(words) < window_size:
            return {"consistency_score": 1.0, "windows_analyzed": 1}
        
        windows = [' '.join(words[i:i+window_size]) for i in range(0, len(words), window_size)]
        window_ppls = [self.calc_perplexity(win) for win in windows if len(win.strip()) > 50]
        
        if len(window_ppls) < 2:
            return {"consistency_score": 1.0, "windows_analyzed": len(window_ppls)}
        
        consistency_score = 1 - (np.std(window_ppls) / np.mean(window_ppls) if np.mean(window_ppls) > 0 else 1)
        
        return {
            "consistency_score": max(0, consistency_score),
            "windows_analyzed": len(window_ppls),
            "window_perplexities": window_ppls
        }


# --------------------------- MODULE 3: Enhanced Transformation Engine ---------------------------

class TransformEngine:
    def __init__(self, transformation_intensity: float = 0.5):
        self.intensity = transformation_intensity
        self.transition_phrases = [
            "Interestingly, ", "On the other hand, ", "Furthermore, ", 
            "However, ", "In contrast, ", "Moreover, ", "Additionally, ",
            "Consequently, ", "Therefore, ", "Nevertheless, ", "Specifically, ",
            "For instance, ", "In particular, ", "As a result, "
        ]
        
        self.sentence_endings = [
            " This often varies in practice.",
            " In my experience, this tends to differ.",
            " However, exceptions are common.",
            " This perspective deserves consideration.",
            " The implications are worth noting.",
            " Further research might reveal more.",
            " This aspect is frequently debated.",
            " Context plays an important role here."
        ]
        
        self.humanizing_patterns = [
            ("it is", "it's"), ("that is", "that's"), ("cannot", "can't"),
            ("do not", "don't"), ("does not", "doesn't"), ("is not", "isn't"),
            ("are not", "aren't"), ("was not", "wasn't"), ("were not", "weren't"),
            ("have not", "haven't"), ("has not", "hasn't"), ("had not", "hadn't"),
            ("will not", "won't"), ("would not", "wouldn't"), ("should not", "shouldn't"),
            ("could not", "couldn't"), ("might not", "mightn't"), ("must not", "mustn't")
        ]
    
    def break_uniformity(self, text: str) -> str:
        """Break sentence uniformity with natural variations"""
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return text
        
        modified = []
        for i, sent in enumerate(sentences):
            current_sent = sent
            
            # Add transition phrases
            if i > 0 and random.random() < (0.3 * self.intensity):
                transition = random.choice(self.transition_phrases)
                current_sent = transition + current_sent[0].lower() + current_sent[1:] if current_sent else current_sent
            
            # Add reflective endings
            if random.random() < (0.25 * self.intensity):
                ending = random.choice(self.sentence_endings)
                current_sent = current_sent.rstrip('.!?') + '.' + ending
            
            # Vary sentence openings
            if random.random() < (0.2 * self.intensity):
                words = current_sent.split()
                if len(words) > 3:
                    # Sometimes reorder the beginning
                    if random.random() < 0.5:
                        current_sent = "Actually, " + current_sent
                    else:
                        # Move a clause to the front
                        words = words[-2:] + words[:-2]
                        current_sent = ' '.join(words).capitalize()
            
            modified.append(current_sent)
        
        return ' '.join(modified)
    
    def vary_sentence_structure(self, text: str) -> str:
        """Vary sentence structures for natural flow"""
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 3:
            return text
        
        # Don't completely shuffle - use intelligent restructuring
        structured = []
        
        # Group sentences and vary their presentation
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                # Combine two sentences with conjunction
                conjunctions = ['and', 'but', 'while', 'although', 'since', 'because']
                conjunction = random.choice(conjunctions)
                combined = f"{sentences[i].rstrip('.!?')} {conjunction} {sentences[i+1][0].lower()}{sentences[i+1][1:].rstrip('.!?')}."
                structured.append(combined)
            else:
                structured.append(sentences[i])
        
        # Occasionally add a question or exclamation
        if random.random() < (0.15 * self.intensity) and len(structured) > 1:
            idx = random.randint(0, len(structured)-1)
            structured[idx] = structured[idx].rstrip('.') + '?' if random.random() < 0.5 else structured[idx].rstrip('.') + '!'
        
        return ' '.join(structured)
    
    def add_human_touch(self, text: str) -> str:
        """Add human-like imperfections and contractions"""
        words = text.split()
        
        # Add contractions
        for i in range(len(words) - 1):
            if random.random() < (0.05 * self.intensity):
                bigram = f"{words[i]} {words[i+1]}".lower()
                for formal, contraction in self.humanizing_patterns:
                    if bigram == formal:
                        words[i] = contraction
                        words[i+1] = ""
        
        # Remove empty strings and clean up
        words = [w for w in words if w]
        
        # Add occasional filler words (sparingly)
        if random.random() < (0.1 * self.intensity) and len(words) > 10:
            filler_pos = random.randint(3, len(words) - 3)
            fillers = ['well,', 'you know,', 'I mean,', 'actually,', 'basically,']
            words.insert(filler_pos, random.choice(fillers))
        
        # Add mild repetition for emphasis
        if random.random() < (0.08 * self.intensity) and len(words) > 15:
            repeat_pos = random.randint(5, len(words) - 5)
            word_to_repeat = words[repeat_pos]
            if len(word_to_repeat) > 3 and word_to_repeat.isalpha():
                words.insert(repeat_pos + 1, word_to_repeat)
        
        return ' '.join(words)
    
    def adjust_punctuation(self, text: str) -> str:
        """Adjust punctuation for more natural flow"""
        # Add occasional ellipses, dashes, or parentheses
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return text
        
        modified = []
        for sent in sentences:
            if random.random() < (0.1 * self.intensity) and len(sent) > 20:
                # Add parenthetical remark
                words = sent.split()
                if len(words) > 5:
                    insert_pos = random.randint(2, len(words) - 3)
                    remarks = ['(or so it seems)', '(in my view)', '(generally speaking)', '(at least typically)']
                    words.insert(insert_pos, random.choice(remarks))
                    sent = ' '.join(words)
            
            if random.random() < (0.08 * self.intensity):
                # Replace some commas with dashes or semicolons
                sent = sent.replace(',', random.choice([' —', ';']), 1)
            
            modified.append(sent)
        
        result = ' '.join(modified)
        
        # Add occasional ellipsis
        if random.random() < (0.05 * self.intensity):
            result = result.rstrip('.') + '...'
        
        return result
    
    def run(self, text: str, apply_all: bool = True) -> str:
        """Apply all transformations for humanization"""
        if not apply_all:
            # Randomly select one transformation
            transformations = [
                self.break_uniformity,
                self.vary_sentence_structure,
                self.add_human_touch,
                self.adjust_punctuation
            ]
            return random.choice(transformations)(text)
        
        # Apply all transformations in a sensible order
        transformed = text
        
        # 1. Break uniformity
        transformed = self.break_uniformity(transformed)
        
        # 2. Vary structure
        transformed = self.vary_sentence_structure(transformed)
        
        # 3. Add human touch
        transformed = self.add_human_touch(transformed)
        
        # 4. Adjust punctuation
        transformed = self.adjust_punctuation(transformed)
        
        return transformed


# --------------------------- MODULE 4: Enhanced Local Classifier ---------------------------

class TurnitinClassifier:
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize classifier with choice of model type:
        - 'logistic': Logistic Regression
        - 'random_forest': Random Forest
        - 'gradient_boost': Gradient Boosting
        - 'svm': Support Vector Machine
        - 'ensemble': Voting classifier of all
        """
        self.model_type = model_type
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        # Initialize models based on type
        if model_type == 'logistic':
            self.models['main'] = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.models['main'] = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boost':
            self.models['main'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.models['main'] = SVC(probability=True, random_state=42)
        elif model_type == 'ensemble':
            from sklearn.ensemble import VotingClassifier
            self.models['logistic'] = LogisticRegression(max_iter=1000, random_state=42)
            self.models['random_forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
            self.models['gradient_boost'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
            self.models['main'] = VotingClassifier(
                estimators=[
                    ('lr', self.models['logistic']),
                    ('rf', self.models['random_forest']),
                    ('gb', self.models['gradient_boost'])
                ],
                voting='soft'
            )
    
    def extract_features(self, analysis_result: Dict[str, Any]) -> List[float]:
        """Extract feature vector from analysis result"""
        features = []
        
        # Sentence stats features
        sent_stats = analysis_result.get('sentence_stats', {})
        features.extend([
            sent_stats.get('avg_sentence_length', 0),
            sent_stats.get('sentence_length_std', 0),
            sent_stats.get('sentence_length_cv', 0),
            sent_stats.get('max_sentence_length', 0) / max(sent_stats.get('avg_sentence_length', 1), 1)
        ])
        
        # Lexical diversity features
        lex_div = analysis_result.get('lexical_diversity', {})
        features.extend([
            lex_div.get('ttr', 0),
            lex_div.get('msttr', 0),
            lex_div.get('hdd', 0),
            lex_div.get('lexical_density', 0)
        ])
        
        # Syntactic complexity features
        syn_comp = analysis_result.get('syntactic_complexity', {})
        features.extend([
            syn_comp.get('avg_clauses_per_sentence', 0),
            syn_comp.get('coordination_ratio', 0),
            syn_comp.get('subordination_ratio', 0)
        ])
        
        # Perplexity features
        perplexity = analysis_result.get('perplexity', {})
        features.extend([
            perplexity.get('mean_ppl', 0),
            perplexity.get('std_ppl', 0),
            perplexity.get('burstiness_score', 0)
        ])
        
        # Punctuation features
        punct = analysis_result.get('punctuation_analysis', {})
        features.extend([
            punct.get('comma_frequency', 0),
            punct.get('period_frequency', 0),
            punct.get('question_frequency', 0),
            punct.get('punctuation_variety', 0)
        ])
        
        # Consistency feature
        consistency = analysis_result.get('consistency', {})
        features.append(consistency.get('consistency_score', 0))
        
        return features
    
    def train(self, X, y, test_size: float = 0.2, cv_folds: int = 5):
        """Train the classifier with cross-validation"""
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train main model
        self.models['main'].fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.models['main'], X_train_scaled, y_train, 
            cv=cv_folds, scoring='roc_auc'
        )
        
        # Test performance
        y_pred = self.models['main'].predict(X_test_scaled)
        y_pred_proba = self.models['main'].predict_proba(X_test_scaled)[:, 1]
        
        self.is_trained = True
        
        # Store performance metrics
        self.metrics = {
            'cv_mean_auc': np.mean(cv_scores),
            'cv_std_auc': np.std(cv_scores),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return self.metrics
    
    def predict(self, features: List[float]) -> float:
        """Predict AI-likelihood score (0-1)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features_scaled = self.scaler.transform([features])
        
        if hasattr(self.models['main'], 'predict_proba'):
            proba = self.models['main'].predict_proba(features_scaled)[0]
            # Return probability of being AI-generated (assuming class 1 is AI)
            return proba[1] if len(proba) > 1 else proba[0]
        else:
            return float(self.models['main'].predict(features_scaled)[0])
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model_type': self.model_type,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'feature_names': self.feature_names
        }
        
        # Save the actual model(s)
        if self.model_type == 'ensemble':
            # For ensemble, we need to save all models
            model_data['models'] = {}
            for name, model in self.models.items():
                model_data['models'][name] = model
        else:
            model_data['model'] = self.models['main']
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.metrics = model_data.get('metrics', {})
        self.feature_names = model_data.get('feature_names', [])
        
        if self.model_type == 'ensemble':
            self.models = model_data['models']
        else:
            self.models['main'] = model_data['model']
        
        print(f"Model loaded from {filepath}")
        return self


# --------------------------- MODULE 5: Complete TurniLab System ---------------------------

class TurniLab:
    def __init__(self, model_type: str = 'ensemble', use_advanced_ppl: bool = True):
        self.linguistic = LinguisticAnalyzer()
        self.ppl = PerplexityEngine(use_simple=not use_advanced_ppl)
        self.transform = TransformEngine()
        self.classifier = TurnitinClassifier(model_type)
        self.is_trained = False
        
    def extract_all_features(self, text: str) -> Dict[str, Any]:
        """Extract all features from text"""
        # Linguistic analysis
        linguistic_features = self.linguistic.run(text)
        
        # Perplexity analysis
        perplexity_features = self.ppl.burstiness(text)
        
        # Consistency analysis
        consistency_features = self.ppl.analyze_consistency(text)
        
        # Combine all features
        all_features = {
            **linguistic_features,
            'perplexity': perplexity_features,
            'consistency': consistency_features
        }
        
        return all_features
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text and return comprehensive report"""
        features = self.extract_all_features(text)
        
        # Extract feature vector for classification
        feature_vector = self.classifier.extract_features(features)
        
        # Get AI score if model is trained
        ai_score = None
        if self.is_trained:
            try:
                ai_score = self.classifier.predict(feature_vector)
            except:
                pass
        
        return {
            "features": features,
            "feature_vector": feature_vector,
            "ai_score": ai_score,
            "text_length": len(text),
            "word_count": len(text.split())
        }
    
    def experiment(self, text: str, iterations: int = 3) -> Dict[str, Any]:
        """Run transformation experiment with multiple iterations"""
        original_analysis = self.analyze(text)
        
        transformations = []
        for i in range(iterations):
            # Vary transformation intensity slightly
            self.transform.intensity = 0.4 + (0.2 * i / iterations)
            transformed = self.transform.run(text)
            trans_analysis = self.analyze(transformed)
            
            transformations.append({
                "iteration": i + 1,
                "transformed_text": transformed,
                "analysis": trans_analysis,
                "intensity_used": self.transform.intensity
            })
        
        # Find best transformation (lowest AI score if trained)
        if self.is_trained:
            best_idx = None
            best_score = float('inf')
            for i, trans in enumerate(transformations):
                if trans["analysis"]["ai_score"] is not None and trans["analysis"]["ai_score"] < best_score:
                    best_score = trans["analysis"]["ai_score"]
                    best_idx = i
        
        return {
            "original": {
                "text": text,
                "analysis": original_analysis
            },
            "transformations": transformations,
            "best_transformation": transformations[best_idx] if best_idx is not None else None
        }
    
    def train_on_dataset(self, human_texts: List[str], ai_texts: List[str], 
                         save_path: str = None) -> Dict[str, float]:
        """Train classifier on provided datasets"""
        print(f"Training on {len(human_texts)} human texts and {len(ai_texts)} AI texts")
        
        # Prepare data
        X = []
        y = []
        
        # Process human texts (label 0)
        print("Processing human texts...")
        for i, text in enumerate(human_texts):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(human_texts)} human texts")
            features = self.extract_all_features(text)
            feature_vector = self.classifier.extract_features(features)
            X.append(feature_vector)
            y.append(0)  # Human
        
        # Process AI texts (label 1)
        print("Processing AI texts...")
        for i, text in enumerate(ai_texts):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(ai_texts)} AI texts")
            features = self.extract_all_features(text)
            feature_vector = self.classifier.extract_features(features)
            X.append(feature_vector)
            y.append(1)  # AI
        
        print(f"Total samples: {len(X)}")
        print(f"Feature vector length: {len(X[0]) if X else 0}")
        
        # Train classifier
        metrics = self.classifier.train(X, y)
        self.is_trained = True
        
        # Save if path provided
        if save_path:
            self.classifier.save_model(save_path)
        
        return metrics
    
    def save_system(self, directory: str = "turnilab_model"):
        """Save entire system state"""
        Path(directory).mkdir(exist_ok=True)
        
        # Save classifier
        if self.is_trained:
            self.classifier.save_model(f"{directory}/classifier.pkl")
        
        # Save configuration
        config = {
            "model_type": self.classifier.model_type,
            "is_trained": self.is_trained,
            "transform_intensity": self.transform.intensity
        }
        
        with open(f"{directory}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"System saved to {directory}/")
    
    def load_system(self, directory: str = "turnilab_model"):
        """Load system state"""
        # Load configuration
        with open(f"{directory}/config.json", 'r') as f:
            config = json.load(f)
        
        # Load classifier if trained
        if config.get('is_trained', False):
            self.classifier.load_model(f"{directory}/classifier.pkl")
            self.is_trained = True
        
        self.transform.intensity = config.get('transform_intensity', 0.5)
        
        print(f"System loaded from {directory}/")
        return self


# --------------------------- TRAINING AND EVALUATION UTILITIES ---------------------------

class TurniLabTrainer:
    """Utility class for training and evaluating TurniLab"""
    
    @staticmethod
    def create_synthetic_dataset(human_samples: int = 500, ai_samples: int = 500, 
                                 avg_length: int = 200) -> Tuple[List[str], List[str]]:
        """Create synthetic dataset for testing/training"""
        human_texts = []
        ai_texts = []
        
        # Simple patterns for human-like text
        human_patterns = [
            "I think that {topic} is really interesting because {reason}.",
            "From my perspective, {topic} has several aspects worth considering.",
            "Well, it's hard to say for sure, but {observation}.",
            "You know, {topic} can be quite complex when you look at it closely.",
            "Actually, I've found that {topic} varies a lot depending on {factor}.",
            "In my experience, {topic} is not as straightforward as it seems.",
            "{Topic}? That's a good question. I'd say {opinion}.",
            "To be honest, {topic} has its pros and cons.",
            "If you ask me, {topic} needs more research in {area}.",
            "Let me think... {topic} reminds me of {analogy}."
        ]
        
        # More uniform patterns for AI-like text
        ai_patterns = [
            "{Topic} is characterized by several key features including {feature1}, {feature2}, and {feature3}.",
            "The analysis of {topic} reveals important considerations regarding {aspect}.",
            "There are multiple factors that contribute to {topic}, such as {factor1} and {factor2}.",
            "{Topic} represents a significant area of study with implications for {field}.",
            "The examination of {topic} demonstrates the relationship between {concept1} and {concept2}.",
            "Several studies have investigated {topic} and found evidence supporting {conclusion}.",
            "The primary components of {topic} include {component1}, {component2}, and {component3}.",
            "Research on {topic} indicates that {finding} with potential applications in {application}.",
            "The theoretical framework for {topic} incorporates elements from {theory1} and {theory2}.",
            "Analysis of {topic} suggests that {implication} with consequences for {outcome}."
        ]
        
        topics = ["artificial intelligence", "climate change", "economic theory", 
                 "philosophical concepts", "scientific research", "literary analysis",
                 "historical events", "technological innovation", "social dynamics",
                 "educational methods"]
        
        import random
        
        # Generate human texts
        for _ in range(human_samples):
            pattern = random.choice(human_patterns)
            topic = random.choice(topics)
            reason = random.choice(["different people see it differently", 
                                   "it connects to many other fields",
                                   "there's always more to learn about it"])
            observation = random.choice(["things aren't always what they seem",
                                       "context matters a lot",
                                       "it depends on various factors"])
            opinion = random.choice(["it's more nuanced than people think",
                                   "we should approach it carefully",
                                   "there's value in multiple perspectives"])
            
            text = pattern.format(topic=topic, reason=reason, observation=observation,
                                opinion=opinion, factor="the situation", 
                                area="several directions", analogy="something similar")
            
            # Add some variation
            if random.random() > 0.7:
                text = "Hmm... " + text.lower()
            if random.random() > 0.8:
                text = text + " Or at least that's my take on it."
            
            human_texts.append(text)
        
        # Generate AI texts
        for _ in range(ai_samples):
            pattern = random.choice(ai_patterns)
            topic = random.choice(topics)
            feature1, feature2, feature3 = random.sample(["complexity", "variability", 
                                                        "adaptability", "scalability",
                                                        "efficiency", "robustness"], 3)
            aspect = random.choice(["implementation", "analysis", "evaluation"])
            factor1, factor2 = random.sample(["methodology", "context", "resources",
                                            "timing", "expertise"], 2)
            
            text = pattern.format(topic=topic, feature1=feature1, feature2=feature2,
                                feature3=feature3, aspect=aspect, factor1=factor1,
                                factor2=factor2, field="related disciplines",
                                concept1="theory", concept2="practice",
                                conclusion="the hypothesis", component1="theory",
                                component2="methodology", component3="application",
                                finding="a correlation exists", application="various fields",
                                theory1="established frameworks", theory2="empirical evidence",
                                implication="a need for further study", outcome="future research")
            
            ai_texts.append(text)
        
        return human_texts, ai_texts
    
    @staticmethod
    def evaluate_system(turnilab: TurniLab, test_human: List[str], test_ai: List[str]) -> Dict[str, Any]:
        """Evaluate system performance on test data"""
        results = {
            "human_predictions": [],
            "ai_predictions": [],
            "confusion_matrix": {"true_human": 0, "true_ai": 0, "false_human": 0, "false_ai": 0}
        }
        
        # Test on human texts
        for text in test_human:
            analysis = turnilab.analyze(text)
            score = analysis.get("ai_score", 0.5)
            results["human_predictions"].append(score)
            if score < 0.5:
                results["confusion_matrix"]["true_human"] += 1
            else:
                results["confusion_matrix"]["false_ai"] += 1
        
        # Test on AI texts
        for text in test_ai:
            analysis = turnilab.analyze(text)
            score = analysis.get("ai_score", 0.5)
            results["ai_predictions"].append(score)
            if score >= 0.5:
                results["confusion_matrix"]["true_ai"] += 1
            else:
                results["confusion_matrix"]["false_human"] += 1
        
        # Calculate metrics
        total = len(test_human) + len(test_ai)
        accuracy = (results["confusion_matrix"]["true_human"] + 
                   results["confusion_matrix"]["true_ai"]) / total if total > 0 else 0
        
        precision = (results["confusion_matrix"]["true_ai"] / 
                    (results["confusion_matrix"]["true_ai"] + results["confusion_matrix"]["false_ai"])) if (
                    results["confusion_matrix"]["true_ai"] + results["confusion_matrix"]["false_ai"]) > 0 else 0
        
        recall = (results["confusion_matrix"]["true_ai"] / 
                 (results["confusion_matrix"]["true_ai"] + results["confusion_matrix"]["false_human"])) if (
                 results["confusion_matrix"]["true_ai"] + results["confusion_matrix"]["false_human"]) > 0 else 0
        
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        results["metrics"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "human_mean_score": np.mean(results["human_predictions"]) if results["human_predictions"] else 0,
            "ai_mean_score": np.mean(results["ai_predictions"]) if results["ai_predictions"] else 0
        }
        
        return results


# --------------------------- MAIN EXECUTION AND EXAMPLE USAGE ---------------------------

def main():
    """Example usage of the complete TurniLab system"""
    
    print("=" * 60)
    print("TURNI-LAB COMPLETE SYSTEM - Training Demo")
    print("=" * 60)
    
    # 1. Initialize system
    print("\n1. Initializing TurniLab system...")
    turnilab = TurniLab(model_type='ensemble', use_advanced_ppl=True)
    
    # 2. Create synthetic dataset for demonstration
    print("\n2. Creating synthetic dataset...")
    human_texts, ai_texts = TurniLabTrainer.create_synthetic_dataset(
        human_samples=200, 
        ai_samples=200
    )
    
    # Split into train/test
    train_human = human_texts[:150]
    test_human = human_texts[150:]
    train_ai = ai_texts[:150]
    test_ai = ai_texts[150:]
    
    print(f"   Training: {len(train_human)} human, {len(train_ai)} AI texts")
    print(f"   Testing: {len(test_human)} human, {len(test_ai)} AI texts")
    
    # 3. Train the system
    print("\n3. Training classifier...")
    metrics = turnilab.train_on_dataset(train_human, train_ai, save_path="turnilab_model/classifier.pkl")
    
    print(f"\n   Training Results:")
    print(f"   - Cross-validation AUC: {metrics['cv_mean_auc']:.3f} (+/- {metrics['cv_std_auc']:.3f})")
    print(f"   - Test Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"   - Test AUC: {metrics['test_auc']:.3f}")
    
    # 4. Evaluate on test data
    print("\n4. Evaluating on test data...")
    results = TurniLabTrainer.evaluate_system(turnilab, test_human, test_ai)
    
    print(f"\n   Test Evaluation:")
    print(f"   - Accuracy: {results['metrics']['accuracy']:.3f}")
    print(f"   - Precision: {results['metrics']['precision']:.3f}")
    print(f"   - Recall: {results['metrics']['recall']:.3f}")
    print(f"   - F1 Score: {results['metrics']['f1_score']:.3f}")
    print(f"   - Human mean score: {results['metrics']['human_mean_score']:.3f}")
    print(f"   - AI mean score: {results['metrics']['ai_mean_score']:.3f}")
    
    # 5. Demonstrate text analysis
    print("\n5. Demonstrating text analysis...")
    
    sample_human = "Well, I think AI is fascinating but also kind of scary, you know? Like, it's amazing what it can do, but sometimes I worry about the consequences."
    sample_ai = "Artificial intelligence represents a transformative technology with significant implications for various sectors including healthcare, finance, and transportation. The integration of machine learning algorithms enables enhanced decision-making capabilities."
    
    print(f"\n   Sample Human Text:")
    print(f"   '{sample_human[:80]}...'")
    analysis_human = turnilab.analyze(sample_human)
    print(f"   AI Score: {analysis_human['ai_score']:.3f}")
    
    print(f"\n   Sample AI Text:")
    print(f"   '{sample_ai[:80]}...'")
    analysis_ai = turnilab.analyze(sample_ai)
    print(f"   AI Score: {analysis_ai['ai_score']:.3f}")
    
    # 6. Demonstrate transformation
    print("\n6. Demonstrating text transformation...")
    experiment = turnilab.experiment(sample_ai, iterations=2)
    
    if experiment["best_transformation"]:
        original_score = experiment["original"]["analysis"]["ai_score"]
        transformed_score = experiment["best_transformation"]["analysis"]["ai_score"]
        
        print(f"\n   Original AI score: {original_score:.3f}")
        print(f"   Transformed AI score: {transformed_score:.3f}")
        print(f"   Improvement: {(original_score - transformed_score):.3f}")
        
        print(f"\n   Transformed text (first 150 chars):")
        print(f"   '{experiment['best_transformation']['transformed_text'][:150]}...'")
    
    # 7. Save complete system
    print("\n7. Saving complete system...")
    turnilab.save_system("turnilab_model")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Load your own dataset (human and AI texts)")
    print("2. Use turnilab.train_on_dataset(your_human_texts, your_ai_texts)")
    print("3. Save model: turnilab.save_system('your_model_path')")
    print("4. Load model: TurniLab().load_system('your_model_path')")
    print("5. Analyze texts: turnilab.analyze('your_text_here')")
    print("6. Transform texts: turnilab.experiment('your_text_here')")


def quick_start_example():
    """Quick start example for immediate use"""
    
    # Initialize system
    tl = TurniLab()
    
    # Analyze a text
    text = "This is an example text to analyze for AI detection patterns."
    result = tl.analyze(text)
    
    print("Analysis Results:")
    print(f"Text length: {result['text_length']} characters")
    print(f"Word count: {result['word_count']} words")
    print(f"AI Score: {result['ai_score'] if result['ai_score'] is not None else 'Model not trained'}")
    
    # Transform text (humanize)
    transformed = tl.transform.run(text)
    print(f"\nTransformed text: {transformed}")
    
    return tl


# --------------------------- COMMAND LINE INTERFACE ---------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TurniLab - AI Detection Research System")
    parser.add_argument("--mode", choices=["demo", "analyze", "train", "transform"], 
                       default="demo", help="Operation mode")
    parser.add_argument("--text", type=str, help="Text to analyze or transform")
    parser.add_argument("--human-file", type=str, help="File with human-written texts (one per line)")
    parser.add_argument("--ai-file", type=str, help="File with AI-generated texts (one per line)")
    parser.add_argument("--model-path", type=str, default="turnilab_model", 
                       help="Path to save/load model")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        main()
    
    elif args.mode == "analyze" and args.text:
        tl = TurniLab()
        try:
            tl.load_system(args.model_path)
        except:
            print("No trained model found, using untrained system")
        
        result = tl.analyze(args.text)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
    
    elif args.mode == "train" and args.human_file and args.ai_file:
        # Load datasets
        with open(args.human_file, 'r') as f:
            human_texts = [line.strip() for line in f if line.strip()]
        
        with open(args.ai_file, 'r') as f:
            ai_texts = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(human_texts)} human texts and {len(ai_texts)} AI texts")
        
        # Train system
        tl = TurniLab()
        metrics = tl.train_on_dataset(human_texts, ai_texts, save_path=f"{args.model_path}/classifier.pkl")
        tl.save_system(args.model_path)
        
        print("\nTraining complete!")
        print(f"Model saved to {args.model_path}/")
        print(f"Test AUC: {metrics['test_auc']:.3f}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
    
    elif args.mode == "transform" and args.text:
        tl = TurniLab()
        try:
            tl.load_system(args.model_path)
        except:
            print("No trained model found, proceeding anyway")
        
        experiment = tl.experiment(args.text, iterations=3)
        
        if args.output:
            with open(args.output, 'w') as f:
                # Simplify for JSON serialization
                output_data = {
                    "original_text": experiment["original"]["text"],
                    "original_ai_score": experiment["original"]["analysis"]["ai_score"],
                    "transformations": []
                }
                
                for trans in experiment["transformations"]:
                    output_data["transformations"].append({
                        "text": trans["transformed_text"],
                        "ai_score": trans["analysis"]["ai_score"]
                    })
                
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Transformations saved to {args.output}")
        else:
            print(f"Original text: {experiment['original']['text']}")
            print(f"Original AI score: {experiment['original']['analysis']['ai_score']}")
            print("\nTransformations:")
            for i, trans in enumerate(experiment["transformations"]):
                print(f"\n{i+1}. AI Score: {trans['analysis']['ai_score']:.3f}")
                print(f"   Text: {trans['transformed_text'][:200]}...")
    
    else:
        print("Invalid arguments. Use --help for usage information.")