import time
import urllib.request
from bs4 import BeautifulSoup
import csv

def download_data():
    subjects = ['Astrophysics', 'Mathematics', 'q-bio', 'Economics', 'Statistics']
    num_results = 100

    with open('data_100_with_link.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["Subject", "Title", "Summary", "Link"])
        for subject in subjects:
            url = f'http://export.arxiv.org/api/query?search_query=all:%s&start=0&max_results=%s' % (subject, num_results)
            data = urllib.request.urlopen(url).read().decode('utf-8')
            soup = BeautifulSoup(data, "xml")

            entries = soup.findAll('entry')
            for entry in entries:
                link = soup.find('link', attrs= {'type': 'text/html'})['href']
                summary = entry.find('summary').text.replace('\n', ' ')
                writer.writerow([subject, entry.find('title').text, summary.strip(), link])
            time.sleep(1)

if __name__ == '__main__':
    download_data()
