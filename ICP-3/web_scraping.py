import requests
from bs4 import BeautifulSoup

url = input("Please enter the Wiki URL: ")

html = requests.get(url) 
bsObj = BeautifulSoup(html.content, "html.parser") 

print("Title: " + bsObj.title.string)

links_list = bsObj.find_all('a', {'class':'Header-link'})
print("Number of links: " + str(len(links_list)))

print("\n\n----------list of links------------\n\n")
for link in links_list:
    print(link.get('href'))
    