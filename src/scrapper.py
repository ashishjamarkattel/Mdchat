import requests
from bs4 import BeautifulSoup
import pandas as pd
from sentence_transformers import SentenceTransformer
import re
import pdb
from src.add_image_markdown import get_content

class ScrapeWebPage:
    """Scrapes the Web page and processes it as required.
    """
    def __init__(self, url) -> None:
        self.url =url
    
    @staticmethod
    def extract_base_url(url:str)->str:
        """Extracts the base url from a long url."""
        pattern = r'^.+?[^\/:](?=[?\/]|$)'
        match = re.match(pattern, url)
        if match:
            return match.group(0)
        else: 
            raise Exception("Invalid URL.")
        
    def get_url(self):
        base_url = ScrapeWebPage.extract_base_url(self.url)
        print(f"BASE URL:{base_url}")
        reqs = requests.get(self.url)
        soup = BeautifulSoup(reqs.text, "html.parser")
        urls = []
        for link in soup.find_all("a"):
            urls.append(link.get("href"))
        if self.url not in urls:
            urls.append(self.url)
        elif base_url not in urls:
            urls.append(base_url)
        urls = list(set(urls))
        urls = list(filter(None, urls))
        print(urls)
        return urls, base_url
    
    def process_urls(self, url_list:list, base_url:str)->list:
        """Processes unnecessary urls in the list and adds the base url if required. 

        Args:
            url_list (list): List of urls to process.

        Returns:
            list: List of processed urls.
        """
        new_url_list = [url for url in url_list if "#" not in url]
        # processed_list = [url for url in new_url_list if url.startswith(self.url)]
        for index, item in enumerate(new_url_list):
            if item.startswith("/"):
                new_url_list[index] = f"{self.url.rstrip('/')}{item}"
        new_url_list = [url for url in new_url_list if base_url in url]
        return new_url_list

        
    def get_page_contents_markdown(self, url_list:list):
        pages=[]
        for link in url_list:
            try:
                print(f"Processing link: {link}")
                filtered_text = get_content(link)
                cleaned_text = ScrapeWebPage.remove_whitespace(filtered_text)
                pages.append({
                    "text": cleaned_text,
                    "source": link
                }) 
            except Exception as e:
                print("Invalid URL: ", link)
        return pages
    
    def get_page_contents(self, url_list:list):
        pages=[]
        for link in url_list:
            try:
                print(f"Processing link: {link}")
                request = requests.get(link)
                scraped_data = BeautifulSoup(request.text, "html.parser")
                filtered_text = scraped_data.text
                cleaned_text = ScrapeWebPage.remove_whitespace(filtered_text)
                pages.append({
                    "text": cleaned_text,
                    "source": link
                }) 
            except Exception as e:
                print("Invalid URL: ", link)
        return pages
    
    @staticmethod
    def remove_whitespace(text:str):
        pattern = r"\s+"
        s = re.sub(pattern, " ", text)
        return s
    

# tai_scraper = ScrapeWebPage("https://www.wiseadmit.io/")
# url_list, base_url = tai_scraper.get_url()
# processed_url = tai_scraper.process_urls(url_list=url_list, base_url=base_url)
# # content = tai_scraper.get_page_contents(url_list = set(processed_url))
# print(processed_url)
