import urllib.request
from bs4 import BeautifulSoup
import csv

subjects = ['Astrophysics', 'Mathematics', 'q-bio', 'Economics', 'Statistics']


with open('data.csv', 'a', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["Subject", "Title", "Summary"])
    for subject in subjects:
        url = f'http://export.arxiv.org/api/query?search_query=all:%s&start=0&max_results=20' % subject
        data = urllib.request.urlopen(url).read().decode('utf-8')
        soup = BeautifulSoup(data, "xml")

        entries = soup.findAll('entry')
        for entry in entries:
            summary = entry.find('summary').text.replace('\n', ' ')
            writer.writerow([subject, entry.find('title').text, summary])