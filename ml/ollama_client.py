"""
Ollama client for local LLM integration
"""

import aiohttp
import asyncio
import logging
from typing import Dict, List, Any, Optional
import json
from config.settings import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_URL
        self.default_model = settings.DEFAULT_MODEL
        self.backup_model = settings.BACKUP_MODEL
        self.session = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=120)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [model['name'] for model in data.get('models', [])]
                else:
                    logger.error(f"Failed to list models: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def generate(
        self,
        prompt: str,
        model: str = None,
        system_prompt: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text using Ollama"""
        
        if not model:
            model = self.default_model
        
        try:
            session = await self._get_session()
            
            # Prepare request data
            request_data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            if system_prompt:
                request_data["system"] = system_prompt
            
            async with session.post(
                f"{self.base_url}/api/generate",
                json=request_data
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'response': result.get('response', ''),
                        'model': model,
                        'tokens_used': result.get('eval_count', 0),
                        'generation_time': result.get('total_duration', 0) / 1e9  # Convert to seconds
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama generation failed: {response.status} - {error_text}")
                    
                    # Try backup model if default fails
                    if model == self.default_model and model != self.backup_model:
                        logger.info(f"Retrying with backup model: {self.backup_model}")
                        return await self.generate(prompt, self.backup_model, system_prompt, max_tokens, temperature)
                    
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}",
                        'model': model
                    }
                    
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            
            # Try backup model if default fails
            if model == self.default_model and model != self.backup_model:
                logger.info(f"Retrying with backup model: {self.backup_model}")
                return await self.generate(prompt, self.backup_model, system_prompt, max_tokens, temperature)
            
            return {
                'success': False,
                'error': str(e),
                'model': model
            }
    
    async def analyze_content(
        self,
        content: str,
        model: str = None,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze content using Ollama with specialized prompts"""
        
        if len(content) > settings.MAX_TEXT_LENGTH:
            content = content[:settings.MAX_TEXT_LENGTH]
        
        # Different analysis prompts
        prompts = {
            "comprehensive": self._get_comprehensive_prompt(content),
            "summary": self._get_summary_prompt(content),
            "sentiment": self._get_sentiment_prompt(content),
            "topics": self._get_topics_prompt(content),
            "entities": self._get_entities_prompt(content),
            "quality": self._get_quality_prompt(content)
        }
        
        if analysis_type == "all":
            # Perform all types of analysis
            results = {}
            for prompt_type, prompt_data in prompts.items():
                try:
                    result = await self.generate(
                        prompt_data["prompt"],
                        model,
                        prompt_data.get("system"),
                        max_tokens=prompt_data.get("max_tokens", 500)
                    )
                    if result['success']:
                        results[prompt_type] = self._parse_analysis_response(
                            result['response'], prompt_type
                        )
                    else:
                        results[prompt_type] = {"error": result.get('error', 'Unknown error')}
                except Exception as e:
                    results[prompt_type] = {"error": str(e)}
            
            return results
        
        else:
            # Single analysis type
            prompt_data = prompts.get(analysis_type, prompts["comprehensive"])
            result = await self.generate(
                prompt_data["prompt"],
                model,
                prompt_data.get("system"),
                max_tokens=prompt_data.get("max_tokens", 1000)
            )
            
            if result['success']:
                return {
                    analysis_type: self._parse_analysis_response(result['response'], analysis_type),
                    'model_info': {
                        'model': result['model'],
                        'tokens_used': result['tokens_used'],
                        'generation_time': result['generation_time']
                    }
                }
            else:
                return {
                    'error': result.get('error', 'Analysis failed'),
                    'model': result.get('model', model)
                }
    
    def _get_comprehensive_prompt(self, content: str) -> Dict[str, str]:
        """Get comprehensive analysis prompt"""
        return {
            "system": "You are an expert content analyst. Provide thorough, structured analysis.",
            "prompt": f"""Analyze the following content comprehensively:

Content: {content}

Provide analysis in the following areas:
1. Main topic and themes
2. Key insights and takeaways
3. Content quality and readability
4. Sentiment and tone
5. Target audience
6. Strengths and weaknesses
7. Actionable recommendations

Format your response as structured text with clear sections.""",
            "max_tokens": 1000
        }
    
    def _get_summary_prompt(self, content: str) -> Dict[str, str]:
        """Get summarization prompt"""
        return {
            "system": "You are an expert summarizer. Create concise, informative summaries.",
            "prompt": f"""Summarize the following content in 2-3 paragraphs, capturing the main points and key insights:

Content: {content}

Summary:""",
            "max_tokens": 300
        }
    
    def _get_sentiment_prompt(self, content: str) -> Dict[str, str]:
        """Get sentiment analysis prompt"""
        return {
            "system": "You are a sentiment analysis expert. Analyze emotional tone and sentiment.",
            "prompt": f"""Analyze the sentiment and emotional tone of this content:

Content: {content}

Provide:
1. Overall sentiment (positive/negative/neutral)
2. Confidence score (0-1)
3. Key emotional indicators
4. Tone description

Format as structured response.""",
            "max_tokens": 200
        }
    
    def _get_topics_prompt(self, content: str) -> Dict[str, str]:
        """Get topic modeling prompt"""
        return {
            "system": "You are a topic analysis expert. Identify main topics and themes.",
            "prompt": f"""Identify the main topics and themes in this content:

Content: {content}

Provide:
1. Primary topic
2. Secondary topics (2-3)
3. Key themes
4. Topic categories
5. Relevance scores

Format as structured list.""",
            "max_tokens": 300
        }
    
    def _get_entities_prompt(self, content: str) -> Dict[str, str]:
        """Get entity extraction prompt"""
        return {
            "system": "You are an entity extraction expert. Identify people, places, organizations, and concepts.",
            "prompt": f"""Extract named entities from this content:

Content: {content}

Identify:
1. People (names, titles)
2. Organizations (companies, institutions)
3. Places (locations, countries, cities)
4. Products/Services
5. Concepts/Technologies
6. Dates/Times

Format as categorized lists.""",
            "max_tokens": 400
        }
    
    def _get_quality_prompt(self, content: str) -> Dict[str, str]:
        """Get content quality assessment prompt"""
        return {
            "system": "You are a content quality expert. Assess content quality across multiple dimensions.",
            "prompt": f"""Assess the quality of this content:

Content: {content}

Evaluate:
1. Information accuracy and reliability
2. Writing clarity and structure
3. Depth and comprehensiveness
4. Originality and uniqueness
5. Engagement and readability
6. Overall quality score (1-10)

Provide detailed assessment with scores.""",
            "max_tokens": 400
        }
    
    def _parse_analysis_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse Ollama response into structured data"""
        
        # Basic parsing - in production, you'd want more sophisticated parsing
        parsed = {
            'raw_response': response,
            'analysis_type': analysis_type,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        # Try to extract structured information based on analysis type
        try:
            lines = response.split('\n')
            
            if analysis_type == "sentiment":
                for line in lines:
                    if "sentiment" in line.lower() or "overall" in line.lower():
                        if "positive" in line.lower():
                            parsed['sentiment'] = 'positive'
                        elif "negative" in line.lower():
                            parsed['sentiment'] = 'negative'
                        else:
                            parsed['sentiment'] = 'neutral'
                    elif "confidence" in line.lower():
                        # Try to extract confidence score
                        import re
                        scores = re.findall(r'(\d*\.?\d+)', line)
                        if scores:
                            parsed['confidence'] = float(scores[0])
            
            elif analysis_type == "topics":
                topics = []
                for line in lines:
                    if line.strip() and any(keyword in line.lower() for keyword in ['topic', 'theme', '-', '•']):
                        topic = line.strip().lstrip('-•').strip()
                        if topic:
                            topics.append(topic)
                parsed['topics'] = topics[:5]  # Top 5 topics
            
            elif analysis_type == "quality":
                for line in lines:
                    if "score" in line.lower():
                        import re
                        scores = re.findall(r'(\d+(?:\.\d+)?)', line)
                        if scores:
                            score = float(scores[0])
                            if score <= 10:  # Assuming 1-10 scale
                                parsed['quality_score'] = score / 10  # Normalize to 0-1
                            else:
                                parsed['quality_score'] = min(score / 100, 1.0)  # Assuming 0-100 scale
            
        except Exception as e:
            logger.warning(f"Failed to parse {analysis_type} response: {e}")
        
        return parsed
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None
    ) -> Dict[str, Any]:
        """Chat-style interaction with Ollama"""
        
        if not model:
            model = self.default_model
        
        try:
            session = await self._get_session()
            
            request_data = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            
            async with session.post(
                f"{self.base_url}/api/chat",
                json=request_data
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'message': result.get('message', {}),
                        'model': model
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}",
                        'model': model
                    }
                    
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': model
            }
    
    async def health_check(self) -> bool:
        """Check if Ollama is available"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception:
            return False