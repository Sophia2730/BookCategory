import requests
import re
from html2text import html2text
import csv, numpy as np
from bs4 import BeautifulSoup

hello = [0,0,0,0,0,0,0,0,0,0,0,0,0 ]
count = 0
err = 0
for i in range(12258356, 12258374):
    url = "http://book.naver.com/bookdb/book_detail.nhn?bid=" + str(i)
    html_result = requests.get(url)
    soup=BeautifulSoup(html_result.text,"html.parser")
    book_info = [0,0,0,0,0]
    try:
        temp1 = soup.find(id='category_location1_depth')
        temp1 = temp1.get_text()
        temp2 = soup.find(id='category_location2_depth')
        temp2 = temp2.get_text()
        book_info[0] = temp1, temp2

        li_up=soup.find_all('div', attrs={'class':'thumb_type'})
        book_info[1] = li_up[0].find('img')['alt']

        text_author = soup.find(id="authorIntroContent")
        book_info[2] = text_author.get_text()

        text_intro = soup.find(id="bookIntroContent")
        book_info[3] = text_intro.get_text()

        text_content = soup.find(id="pubReviewContent")
        book_info[4] = text_content.get_text()
    except:
        print(".", end= "")
        err += 1
    else:
        print(".".rstrip())
        hello[count] = [book_info[0],book_info[1],book_info[2],book_info[3],book_info[4]]
        count += 1
list_out = []
print("count : ", count, " err : ", err)
for i in range(count):
    with open('./book_data.csv', "w+", encoding='utf-8') as fout:
        csvout = csv.DictWriter(fout, ['Category', 'Title', 'Author', 'Intro', 'Content'])
        csvout.writeheader()
        list_out.append(
            {'Category': ''.join(hello[i][0]), 'Title': ''.join(hello[i][1]), 'Author': ''.join(hello[i][2]),
             'Intro': ''.join(hello[i][3]), 'Content': ''.join(hello[i][4])})
        csvout.writerows(list_out)

