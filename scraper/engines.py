"""
Multi-engine web scraping capabilities
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
from urllib.parse import urljoin, urlparse
from config.settings import settings

logger = logging.getLogger(__name__)


class ScrapingEngine:
    """Base scraping engine class"""
    
    def __init__(self):
        self.session = None
        self.driver = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if self.driver:
            self.driver.quit()


class StaticHTMLEngine(ScrapingEngine):
    """Engine for scraping static HTML content"""
    
    async def scrape(self, url: str) -> Dict[str, Any]:
        """Scrape static HTML content"""
        try:
            if not self.session:
                timeout = aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT)
                self.session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers={'User-Agent': settings.USER_AGENT}
                )
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    return await self._parse_html(html_content, url)
                else:
                    raise Exception(f"HTTP {response.status}: {response.reason}")
                    
        except Exception as e:
            logger.error(f"Static HTML scraping failed for {url}: {e}")
            raise


class JavaScriptEngine(ScrapingEngine):
    """Engine for scraping JavaScript-heavy sites"""
    
    def __init__(self):
        super().__init__()
        self._setup_driver()
    
    def _setup_driver(self):
        """Setup Selenium WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"--user-agent={settings.USER_AGENT}")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(settings.REQUEST_TIMEOUT)
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            raise
    
    async def scrape(self, url: str) -> Dict[str, Any]:
        """Scrape JavaScript-heavy content"""
        try:
            # Run in thread pool since Selenium is sync
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._scrape_sync, url)
            return result
            
        except Exception as e:
            logger.error(f"JavaScript scraping failed for {url}: {e}")
            raise
    
    def _scrape_sync(self, url: str) -> Dict[str, Any]:
        """Synchronous scraping method"""
        self.driver.get(url)
        
        # Wait for page to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Additional wait for dynamic content
        time.sleep(2)
        
        html_content = self.driver.page_source
        return asyncio.run(self._parse_html(html_content, url))


class DynamicEngine(ScrapingEngine):
    """Engine for scraping dynamic content with API calls"""
    
    async def scrape(self, url: str) -> Dict[str, Any]:
        """Scrape dynamic content by analyzing network requests"""
        try:
            # First, try static scraping
            static_result = await StaticHTMLEngine().scrape(url)
            
            # Then, try to find API endpoints
            api_data = await self._find_api_endpoints(url)
            
            # Combine results
            return {
                **static_result,
                'api_data': api_data,
                'scraping_type': 'dynamic'
            }
            
        except Exception as e:
            logger.error(f"Dynamic scraping failed for {url}: {e}")
            raise
    
    async def _find_api_endpoints(self, url: str) -> List[Dict]:
        """Find potential API endpoints"""
        # This is a simplified implementation
        # In production, you'd use browser automation to intercept network requests
        api_endpoints = []
        
        try:
            # Common API patterns
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            common_endpoints = [
                "/api/data",
                "/api/content",
                "/api/posts",
                "/api/articles",
                "/data.json",
                "/content.json"
            ]
            
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for endpoint in common_endpoints:
                    try:
                        full_url = urljoin(base_url, endpoint)
                        async with session.get(full_url) as response:
                            if response.status == 200:
                                data = await response.json()
                                api_endpoints.append({
                                    'url': full_url,
                                    'data': data
                                })
                    except:
                        continue
                        
        except Exception as e:
            logger.warning(f"API endpoint discovery failed: {e}")
        
        return api_endpoints


class MultiEngine:
    """Multi-engine scraper that tries different approaches"""
    
    def __init__(self):
        self.engines = {
            'static': StaticHTMLEngine,
            'javascript': JavaScriptEngine,
            'dynamic': DynamicEngine
        }
    
    async def scrape(self, url: str, engine_type: str = 'auto') -> Dict[str, Any]:
        """Scrape using specified or auto-detected engine"""
        
        if engine_type == 'auto':
            engine_type = await self._detect_engine_type(url)
        
        engine_class = self.engines.get(engine_type, StaticHTMLEngine)
        
        async with engine_class() as engine:
            result = await engine.scrape(url)
            result['engine_used'] = engine_type
            result['url'] = url
            result['scraped_at'] = time.time()
            return result
    
    async def _detect_engine_type(self, url: str) -> str:
        """Auto-detect the best engine for the URL"""
        try:
            # Quick check for common patterns
            if any(pattern in url.lower() for pattern in ['spa', 'react', 'angular', 'vue']):
                return 'javascript'
            
            # Try static first (fastest)
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            # Check for JavaScript-heavy indicators
                            if any(indicator in html.lower() for indicator in [
                                'window.react', 'window.angular', 'window.vue',
                                'data-reactroot', 'ng-app', 'v-app'
                            ]):
                                return 'javascript'
                            return 'static'
            except:
                pass
            
            # Default to dynamic for complex sites
            return 'dynamic'
            
        except Exception as e:
            logger.warning(f"Engine detection failed for {url}: {e}")
            return 'static'
    
    async def batch_scrape(self, urls: List[str], engine_type: str = 'auto') -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently"""
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_SCRAPES)
        
        async def scrape_with_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    result = await self.scrape(url, engine_type)
                    await asyncio.sleep(settings.REQUEST_DELAY)
                    return result
                except Exception as e:
                    logger.error(f"Batch scraping failed for {url}: {e}")
                    return {
                        'url': url,
                        'error': str(e),
                        'scraped_at': time.time(),
                        'engine_used': engine_type
                    }
        
        tasks = [scrape_with_semaphore(url) for url in urls[:settings.MAX_URLS_PER_BATCH]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch scraping exception: {result}")
            else:
                valid_results.append(result)
        
        return valid_results


# Common HTML parsing method
async def _parse_html(html_content: str, url: str) -> Dict[str, Any]:
    """Parse HTML content and extract structured data"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Extract basic information
    title = soup.find('title')
    title_text = title.get_text().strip() if title else "No title"
    
    # Extract meta description
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    description = meta_desc.get('content', '') if meta_desc else ''
    
    # Extract all text content
    text_content = soup.get_text()
    
    # Clean up text
    lines = (line.strip() for line in text_content.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = ' '.join(chunk for chunk in chunks if chunk)
    
    # Extract links
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith(('http://', 'https://')):
            links.append({
                'url': href,
                'text': link.get_text().strip()
            })
        elif href.startswith('/'):
            links.append({
                'url': urljoin(url, href),
                'text': link.get_text().strip()
            })
    
    # Extract images
    images = []
    for img in soup.find_all('img', src=True):
        src = img['src']
        if src.startswith(('http://', 'https://')):
            images.append({
                'url': src,
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            })
        elif src.startswith('/'):
            images.append({
                'url': urljoin(url, src),
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            })
    
    # Extract headers
    headers = []
    for i in range(1, 7):
        for header in soup.find_all(f'h{i}'):
            headers.append({
                'level': i,
                'text': header.get_text().strip()
            })
    
    return {
        'title': title_text,
        'description': description,
        'text_content': clean_text,
        'word_count': len(clean_text.split()),
        'links': links,
        'images': images,
        'headers': headers,
        'html_content': html_content,
        'scraping_type': 'static'
    }

# Bind method to engine classes
StaticHTMLEngine._parse_html = _parse_html
JavaScriptEngine._parse_html = _parse_html
DynamicEngine._parse_html = _parse_html