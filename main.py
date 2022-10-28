import urllib, urllib.request
from bs4 import BeautifulSoup

url = 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1'
data = urllib.request.urlopen(url).read().decode('utf-8')

soup = BeautifulSoup(data, "xml")
print(soup.prettify())