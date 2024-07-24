from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import requests
import json
import os
import html2text

def scrape_website(url):
    html_content = requests.get(url)
    return html_content.text

def convert_html_to_markdown(html):

    # Create an html2text converter
    converter = html2text.HTML2Text()

    # Configure the converter
    converter.ignore_links = False

    # Convert the HTML to Markdown
    markdown = converter.handle(html)

    return markdown



def convert_to_absolute_url(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')

    for img_tag in soup.find_all('img'):
        if img_tag.get('src'):
            src = img_tag.get('src')
            if src.startswith(('http://', 'https://')):
                continue
            absolute_url = urljoin(base_url, src)
            img_tag['src'] = absolute_url
        elif img_tag.get('data-src'):
            src = img_tag.get('data-src')
            if src.startswith(('http://', 'https://')):
                continue
            absolute_url = urljoin(base_url, src)
            img_tag['data-src'] = absolute_url

    for link_tag in soup.find_all('a'):
        href = link_tag.get('href')
        if href.startswith(('http://', 'https://')):
            continue
        absolute_url = urljoin(base_url, href)
        link_tag['href'] = absolute_url

    updated_html = str(soup)

    return updated_html


def get_content(url):
    html = scrape_website(url)
    updated_html = convert_to_absolute_url(html, url)
    markdown = convert_html_to_markdown(updated_html)

    return markdown

# if __name__ == "__main__":
#     url = "https://thecapitaldubai.com/"
#     html = scrape_website(url=url)
#     updated_html = convert_to_absolute_url(html, url)
#     print(convert_html_to_markdown(updated_html))

