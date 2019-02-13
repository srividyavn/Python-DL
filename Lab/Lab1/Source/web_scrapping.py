#program to parse a website using a beautiful soup library

import requests
from bs4 import BeautifulSoup

#extracting the contents of the web page into html document
html_doc = requests.get('https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States')

#parsing a website into html doc
soup = BeautifulSoup(html_doc.text, 'html.parser')
data = soup.find('table')
data = data.find('tbody')

flag = 0
#opening a file web scrapping to store the state and capital.
with open("web_scrapping.txt", "w", encoding="utf-8") as f:
    for row in data.find_all('tr')[2:]:
        items = row.find_all('td')
        f.write("State: " + row.th.a.string + "-" + row.td.string)
        for i in items[2].text:
            if i.isdigit():
                flag = 1
        if flag == 1:
            f.write("Capitals: " + items[1].text)
        else:
            f.write("Capitals: " + items[1].text + "," + items[2].text)







