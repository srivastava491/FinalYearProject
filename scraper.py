import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
import logging
from typing import Set, List
import json
import fitz  # <-- Added for PDF processing

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NITKKRScraper:
    def __init__(self, base_url="https://nitkkr.ac.in/", max_depth=2, delay=1):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_depth = max_depth
        self.delay = delay  # Delay between requests to be respectful
        self.visited = set()
        self.scraped_pages = []
        
        # Create directories
        os.makedirs("nitkkr_pages", exist_ok=True)
        os.makedirs("extracted_text", exist_ok=True)
        
    def save_page(self, url: str, html: str, metadata: dict | None = None) -> str:
        """Save page HTML content and metadata."""
        # Convert URL to a safe filename
        parsed_url = urlparse(url)
        filename = parsed_url.path.replace("/", "_")
        if not filename or filename == "_":
            filename = "index"
        
        # Clean filename
        filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
        filepath = f"nitkkr_pages/{filename}.html"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
            
        # Save metadata
        if metadata is not None:
            metadata_file = f"nitkkr_pages/{filename}_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        
        return filepath
    
    def extract_text_from_html(self, html: str, url: str) -> dict:
        """Extract clean text from HTML content."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Extract title
        title = soup.find("title")
        title_text = title.get_text().strip() if title and hasattr(title, 'get_text') else "No Title"
        
        # Extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = ""
        if meta_desc and hasattr(meta_desc, 'attrs'):
            description = meta_desc.attrs.get("content", "")  # type: ignore
        
        return {
            "url": url,
            "title": title_text,
            "description": description,
            "text": clean_text,
            "word_count": len(clean_text.split())
        }
    
    def extract_text_from_pdf(self, pdf_bytes: bytes, url: str) -> dict:
        """Extract clean text from PDF content."""
        text = ""
        title = "No Title"
        description = ""
        
        try:
            # Open the PDF from in-memory bytes
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Extract title and description from metadata
            if pdf_doc.metadata:
                title = pdf_doc.metadata.get('title', 'No Title')
                if not title.strip() or title == "No Title":
                    # Fallback to filename if title is generic
                    title = os.path.basename(urlparse(url).path)
                description = pdf_doc.metadata.get('subject', '')
            
            # Extract text from all pages
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                text += page.get_text() + "\n"
                
            pdf_doc.close()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)
            
            return {
                "url": url,
                "title": title,
                "description": description,
                "text": clean_text,
                "word_count": len(clean_text.split())
            }
        except Exception as e:
            logger.error(f"❌ Error extracting text from PDF {url}: {e}")
            return {
                "url": url,
                "title": "PDF Extraction Failed",
                "description": "",
                "text": "",
                "word_count": 0
            }

    def save_extracted_text(self, text_data: dict) -> str:
        """Save extracted text to a file."""
        url = text_data["url"]
        parsed_url = urlparse(url)
        filename = parsed_url.path.replace("/", "_")
        if not filename or filename == "_":
            filename = "index"
        
        # Make filename safer for PDFs (which can have long names)
        filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
        if len(filename) > 100:  # Truncate long filenames
            filename = filename[:100] + "_truncated"
             
        filepath = f"extracted_text/{filename}.txt"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"URL: {text_data['url']}\n")
            f.write(f"Title: {text_data['title']}\n")
            f.write(f"Description: {text_data['description']}\n")
            f.write(f"Word Count: {text_data['word_count']}\n")
            f.write("-" * 50 + "\n")
            f.write(text_data['text'])
        
        return filepath
    
    def crawl(self, url: str, depth: int = 0) -> None:
        """Recursively crawl NIT KKR pages."""
        if url in self.visited or depth > self.max_depth:
            return
        
        # Clean URL fragment
        url = url.split('#')[0]
        self.visited.add(url)
        
        try:
            logger.info(f"Crawling (depth {depth}): {url}")
            
            # Add delay to be respectful to the server
            if self.delay > 0:
                time.sleep(self.delay)
            
            response = requests.get(url, verify=False, timeout=10)
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "")
            
            text_data = None
            html_file = None

            if "text/html" in content_type:
                # 1. Handle HTML (as before)
                html = response.text
                metadata = {
                    "url": url,
                    "depth": depth,
                    "content_type": content_type,
                    "status_code": response.status_code,
                    "timestamp": time.time()
                }
                html_file = self.save_page(url, html, metadata)
                text_data = self.extract_text_from_html(html, url)
                
            elif "application/pdf" in content_type:
                # 2. Handle PDF
                logger.info(f"📄 Found PDF: {url}")
                pdf_bytes = response.content
                text_data = self.extract_text_from_pdf(pdf_bytes, url)
                # We don't save the PDF HTML, so html_file remains None
            
            else:
                # 3. Skip other content types
                logger.info(f"Skipping non-HTML/PDF content: {url} ({content_type})")
                return

            # Save extracted text if any was found
            if text_data and text_data["word_count"] > 10:  # Only save if text was extracted
                text_file = self.save_extracted_text(text_data)
                
                # Store page info
                self.scraped_pages.append({
                    "url": url,
                    "depth": depth,
                    "html_file": html_file,  # Will be None for PDFs
                    "text_file": text_file,
                    "title": text_data["title"],
                    "word_count": text_data["word_count"]
                })
                logger.info(f"✅ Successfully processed: {url} (Title: {text_data['title']}, Words: {text_data['word_count']})")
            
            else:
                logger.info(f"Skipping page with no content: {url}")

            # Find and follow links (only from HTML pages)
            if "text/html" in content_type:
                soup = BeautifulSoup(response.text, "html.parser")
                
                for link in soup.find_all("a", href=True):
                    next_url = urljoin(url, link["href"])
                    parsed_next = urlparse(next_url)

                    # Skip mailto, JS, and non-http links
                    if next_url.startswith("mailto:") or next_url.startswith("javascript:") or not next_url.startswith("http"):
                        continue
                    
                    # Only follow links from the same domain
                    if self.domain in parsed_next.netloc:
                        self.crawl(next_url, depth + 1)
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request error for {url}: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error for {url}: {e}")
    
    def get_crawl_summary(self) -> dict:
        """Get summary of crawling results."""
        total_pages = len(self.scraped_pages)
        total_words = sum(page["word_count"] for page in self.scraped_pages)
        
        return {
            "total_pages": total_pages,
            "total_words": total_words,
            "average_words_per_page": total_words / total_pages if total_pages > 0 else 0,
            "pages_by_depth": {
                depth: len([p for p in self.scraped_pages if p["depth"] == depth])
                for depth in range(self.max_depth + 1)
            }
        }
    
    def save_crawl_summary(self) -> None:
        """Save crawling summary to file."""
        summary = self.get_crawl_summary()
        summary_file = "crawl_summary.json"
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Crawl summary saved to {summary_file}")


def main():
    """Main function to run the scraper."""
    scraper = NITKKRScraper(max_depth=2, delay=1)
    
    logger.info("Starting NIT KKR website scraping...")
    scraper.crawl(scraper.base_url)
    
    # Save summary
    scraper.save_crawl_summary()
    
    # Print summary
    summary = scraper.get_crawl_summary()
    logger.info(f"Scraping completed!")
    logger.info(f"Total pages crawled: {summary['total_pages']}")
    logger.info(f"Total words extracted: {summary['total_words']}")
    logger.info(f"Average words per page: {summary['average_words_per_page']:.1f}")


if __name__ == "__main__":
    main()
