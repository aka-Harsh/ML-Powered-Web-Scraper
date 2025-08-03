"""
NLP Processing module for advanced text analysis
"""

import asyncio
import logging
from typing import Dict, List, Any, Tuple, Optional
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter, defaultdict
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class NLPProcessor:
    """Advanced NLP processing for web content analysis"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.nlp_spacy = None
        self.emotion_pipeline = None
        self.summarization_pipeline = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models asynchronously"""
        try:
            # Load spaCy model
            try:
                import spacy
                self.nlp_spacy = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Some features will be limited.")
                self.nlp_spacy = None
            
            # Load emotion analysis pipeline
            try:
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=-1  # CPU
                )
            except Exception as e:
                logger.warning(f"Failed to load emotion model: {e}")
                self.emotion_pipeline = None
            
            # Load summarization pipeline
            try:
                self.summarization_pipeline = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # CPU
                )
            except Exception as e:
                logger.warning(f"Failed to load summarization model: {e}")
                self.summarization_pipeline = None
                
        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {e}")
    
    async def analyze_content(self, text: str, title: str = "") -> Dict[str, Any]:
        """
        Comprehensive NLP analysis of content
        
        Args:
            text: Text content to analyze
            title: Optional title for context
            
        Returns:
            Dictionary containing analysis results
        """
        
        if not text or len(text.strip()) < 10:
            return {"error": "Text too short for analysis"}
        
        # Clean and prepare text
        cleaned_text = self._clean_text(text)
        
        # Run analysis in parallel where possible
        results = {}
        
        try:
            # Basic text statistics
            results['statistics'] = await self._get_text_statistics(cleaned_text, title)
            
            # Sentiment analysis
            results['sentiment'] = await self._analyze_sentiment(cleaned_text)
            
            # Emotion analysis
            if self.emotion_pipeline:
                results['emotions'] = await self._analyze_emotions(cleaned_text)
            
            # Named Entity Recognition
            results['entities'] = await self._extract_entities(cleaned_text)
            
            # Topic modeling
            results['topics'] = await self._extract_topics(cleaned_text)
            
            # Keywords extraction
            results['keywords'] = await self._extract_keywords(cleaned_text)
            
            # Language detection
            results['language'] = await self._detect_language(cleaned_text)
            
            # Readability analysis
            results['readability'] = await self._analyze_readability(cleaned_text)
            
            # Content categorization
            results['category'] = await self._categorize_content(cleaned_text, title)
            
            # Text summarization
            if self.summarization_pipeline and len(cleaned_text) > 100:
                results['summary'] = await self._generate_summary(cleaned_text)
            
        except Exception as e:
            logger.error(f"NLP analysis failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        return text.strip()
    
    async def _get_text_statistics(self, text: str, title: str = "") -> Dict[str, Any]:
        """Get basic text statistics"""
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Filter out stopwords and punctuation
        content_words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Calculate statistics
        stats = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(text.split('\n\n')),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'avg_chars_per_word': sum(len(word) for word in words) / len(words) if words else 0,
            'unique_words': len(set(content_words)),
            'lexical_diversity': len(set(content_words)) / len(content_words) if content_words else 0
        }
        
        if title:
            stats['title_length'] = len(title)
            stats['title_word_count'] = len(title.split())
        
        return stats
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using multiple approaches"""
        sentiment_results = {}
        
        try:
            # VADER sentiment
            vader_scores = self.sia.polarity_scores(text)
            sentiment_results['vader'] = vader_scores
            
            # TextBlob sentiment
            blob = TextBlob(text)
            sentiment_results['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
            
            # Combined sentiment
            compound_score = vader_scores['compound']
            if compound_score >= 0.05:
                overall_sentiment = 'positive'
            elif compound_score <= -0.05:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'
            
            sentiment_results['overall'] = {
                'sentiment': overall_sentiment,
                'confidence': abs(compound_score),
                'score': compound_score
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            sentiment_results['error'] = str(e)
        
        return sentiment_results
    
    async def _analyze_emotions(self, text: str) -> Dict[str, Any]:
        """Analyze emotions using transformer model"""
        if not self.emotion_pipeline:
            return {"error": "Emotion model not available"}
        
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            # Run emotion analysis
            emotions = self.emotion_pipeline(text)
            
            # Process results
            emotion_scores = {}
            for emotion in emotions:
                emotion_scores[emotion['label'].lower()] = emotion['score']
            
            # Get dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            
            return {
                'emotions': emotion_scores,
                'dominant_emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1]
            }
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {"error": str(e)}
    
    async def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities using spaCy and NLTK"""
        entities = {
            'spacy_entities': [],
            'nltk_entities': [],
            'entity_summary': defaultdict(list)
        }
        
        try:
            # spaCy NER
            if self.nlp_spacy:
                doc = self.nlp_spacy(text)
                spacy_entities = []
                for ent in doc.ents:
                    entity_info = {
                        'text': ent.text,
                        'label': ent.label_,
                        'description': spacy.explain(ent.label_),
                        'start': ent.start_char,
                        'end': ent.end_char
                    }
                    spacy_entities.append(entity_info)
                    entities['entity_summary'][ent.label_].append(ent.text)
                
                entities['spacy_entities'] = spacy_entities
            
            # NLTK NER (backup/additional)
            try:
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                nltk_entities = []
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        entity_text = ' '.join([token for token, pos in chunk.leaves()])
                        entity_info = {
                            'text': entity_text,
                            'label': chunk.label()
                        }
                        nltk_entities.append(entity_info)
                        entities['entity_summary'][chunk.label()].append(entity_text)
                
                entities['nltk_entities'] = nltk_entities
                
            except Exception as e:
                logger.warning(f"NLTK NER failed: {e}")
            
            # Clean up entity summary
            for label, entity_list in entities['entity_summary'].items():
                entities['entity_summary'][label] = list(set(entity_list))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            entities['error'] = str(e)
        
        return dict(entities)
    
    async def _extract_topics(self, text: str) -> Dict[str, Any]:
        """Extract topics using LDA"""
        try:
            # Prepare documents (split into sentences for better topic modeling)
            sentences = sent_tokenize(text)
            if len(sentences) < 3:
                return {"error": "Not enough content for topic modeling"}
            
            # Vectorize text
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            if doc_term_matrix.shape[0] < 2:
                return {"error": "Insufficient data for topic modeling"}
            
            # LDA topic modeling
            n_topics = min(3, len(sentences))  # Adjust based on content length
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            
            lda.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_info = {
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': [float(topic[i]) for i in top_words_idx]
                }
                topics.append(topic_info)
            
            return {
                'topics': topics,
                'n_topics': n_topics,
                'perplexity': float(lda.perplexity(doc_term_matrix))
            }
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return {"error": str(e)}
    
    async def _extract_keywords(self, text: str) -> Dict[str, Any]:
        """Extract keywords using TF-IDF and frequency analysis"""
        try:
            # Word frequency analysis
            words = word_tokenize(text.lower())
            content_words = [word for word in words if word.isalpha() and word not in self.stop_words and len(word) > 2]
            
            word_freq = Counter(content_words)
            top_frequent = word_freq.most_common(10)
            
            # TF-IDF analysis
            sentences = sent_tokenize(text)
            if len(sentences) > 1:
                vectorizer = TfidfVectorizer(
                    stop_words='english',
                    max_features=20,
                    ngram_range=(1, 2)
                )
                
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get mean TF-IDF scores
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                
                # Create keyword-score pairs
                keyword_scores = list(zip(feature_names, mean_scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                
                tfidf_keywords = keyword_scores[:10]
            else:
                tfidf_keywords = []
            
            return {
                'frequency_keywords': [{'word': word, 'count': count} for word, count in top_frequent],
                'tfidf_keywords': [{'word': word, 'score': float(score)} for word, score in tfidf_keywords],
                'total_unique_words': len(set(content_words))
            }
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return {"error": str(e)}
    
    async def _detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of the text"""
        try:
            blob = TextBlob(text)
            detected_lang = blob.detect_language()
            
            # Get confidence (simplified)
            confidence = 0.8 if len(text) > 100 else 0.6
            
            return {
                'language': detected_lang,
                'confidence': confidence,
                'is_english': detected_lang == 'en'
            }
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return {
                'language': 'en',  # Default to English
                'confidence': 0.5,
                'is_english': True,
                'error': str(e)
            }
    
    async def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability"""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            syllables = sum([self._count_syllables(word) for word in words if word.isalpha()])
            
            if len(sentences) == 0 or len(words) == 0:
                return {"error": "Insufficient text for readability analysis"}
            
            # Flesch Reading Ease
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllables / len(words)
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Flesch-Kincaid Grade Level
            fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
            
            # Interpret scores
            if flesch_score >= 90:
                reading_level = "Very Easy"
            elif flesch_score >= 80:
                reading_level = "Easy"
            elif flesch_score >= 70:
                reading_level = "Fairly Easy"
            elif flesch_score >= 60:
                reading_level = "Standard"
            elif flesch_score >= 50:
                reading_level = "Fairly Difficult"
            elif flesch_score >= 30:
                reading_level = "Difficult"
            else:
                reading_level = "Very Difficult"
            
            return {
                'flesch_reading_ease': float(flesch_score),
                'flesch_kincaid_grade': float(fk_grade),
                'reading_level': reading_level,
                'avg_sentence_length': float(avg_sentence_length),
                'avg_syllables_per_word': float(avg_syllables_per_word)
            }
            
        except Exception as e:
            logger.error(f"Readability analysis failed: {e}")
            return {"error": str(e)}
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    async def _categorize_content(self, text: str, title: str = "") -> Dict[str, Any]:
        """Categorize content into predefined categories"""
        try:
            # Define category keywords
            categories = {
                'technology': ['software', 'hardware', 'computer', 'digital', 'tech', 'ai', 'machine learning', 'programming'],
                'business': ['company', 'market', 'revenue', 'profit', 'strategy', 'business', 'corporate', 'finance'],
                'science': ['research', 'study', 'experiment', 'scientific', 'analysis', 'data', 'discovery'],
                'health': ['health', 'medical', 'doctor', 'patient', 'treatment', 'medicine', 'wellness'],
                'education': ['school', 'student', 'teacher', 'learning', 'education', 'university', 'academic'],
                'entertainment': ['movie', 'music', 'game', 'entertainment', 'celebrity', 'show', 'film'],
                'sports': ['sport', 'team', 'player', 'game', 'match', 'tournament', 'athletic'],
                'politics': ['government', 'political', 'election', 'policy', 'candidate', 'vote', 'democracy'],
                'travel': ['travel', 'vacation', 'destination', 'trip', 'tourism', 'hotel', 'flight'],
                'food': ['food', 'recipe', 'cooking', 'restaurant', 'cuisine', 'meal', 'ingredient']
            }
            
            # Combine title and text for categorization
            full_text = f"{title} {text}".lower()
            words = word_tokenize(full_text)
            
            # Calculate category scores
            category_scores = {}
            for category, keywords in categories.items():
                score = sum([1 for word in words if word in keywords])
                category_scores[category] = score
            
            # Normalize scores
            total_score = sum(category_scores.values())
            if total_score > 0:
                for category in category_scores:
                    category_scores[category] = category_scores[category] / total_score
            
            # Get top category
            top_category = max(category_scores.items(), key=lambda x: x[1])
            
            return {
                'predicted_category': top_category[0] if top_category[1] > 0 else 'general',
                'confidence': float(top_category[1]),
                'all_scores': {k: float(v) for k, v in category_scores.items()}
            }
            
        except Exception as e:
            logger.error(f"Content categorization failed: {e}")
            return {"error": str(e)}
    
    async def _generate_summary(self, text: str) -> Dict[str, Any]:
        """Generate text summary"""
        if not self.summarization_pipeline:
            return {"error": "Summarization model not available"}
        
        try:
            # Truncate text if too long for the model
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            # Generate summary
            summary_result = self.summarization_pipeline(
                text,
                max_length=150,
                min_length=30,
                do_sample=False
            )
            
            summary_text = summary_result[0]['summary_text']
            
            # Calculate compression ratio
            compression_ratio = len(summary_text) / len(text)
            
            return {
                'summary': summary_text,
                'original_length': len(text),
                'summary_length': len(summary_text),
                'compression_ratio': float(compression_ratio)
            }
            
        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            return {"error": str(e)}
    
    async def batch_analyze(self, texts: List[str], titles: List[str] = None) -> List[Dict[str, Any]]:
        """Analyze multiple texts in batch"""
        if not titles:
            titles = [""] * len(texts)
        
        # Process in batches to avoid memory issues
        results = []
        batch_size = 10
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_titles = titles[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = [
                self.analyze_content(text, title)
                for text, title in zip(batch_texts, batch_titles)
            ]
            
            # Run batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({"error": str(result)})
                else:
                    results.append(result)
        
        return results