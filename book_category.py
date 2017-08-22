import requests
import csv
from bs4 import BeautifulSoup

data = [0 for i in range(20)]       # 크롤링한 데이터 값을 저장할 배열
count = 0
err = 0
for i in range(12258356, 12258370):
    url = "http://book.naver.com/bookdb/book_detail.nhn?bid=" + str(i)
    html_result = requests.get(url)
    soup=BeautifulSoup(html_result.text,"html.parser")
    book_info = [0,0,0,0,0]
    try:
        temp = soup.find(id='category_location1_depth')
        temp = temp.get_text()
        book_info[0] = temp

        li_up=soup.find_all('div', attrs={'class':'thumb_type'})
        book_info[1] = li_up[0].find('img')['alt']

        text_author = soup.find(id="authorIntroContent")
        book_info[2] = text_author.get_text()

        text_intro = soup.find(id="bookIntroContent")
        book_info[3] = text_intro.get_text()

        text_content = soup.find(id="pubReviewContent")
        book_info[4] = text_content.get_text()
    except:     # 에러 발생시 에러카운트 증가
        print(".", end= "")
        err += 1
    else:       # 에러 미발생시 data에 값을 저장
        print(".".rstrip())
        data[count] = [book_info[0],book_info[1],book_info[2],book_info[3],book_info[4]]
        count += 1


list_out = []
print("count : ", count, " err : ", err)
for i in range(count):
    with open('./book_data.csv', "w+", encoding='utf-8') as fout:
        csvout = csv.DictWriter(fout, ['Category', 'Title', 'Author', 'Intro', 'Content'])
        csvout.writeheader()
        list_out.append(
            {'Category': ''.join(data[i][0]), 'Title': ''.join(data[i][1]), 'Author': ''.join(data[i][2]),
             'Intro': ''.join(data[i][3]), 'Content': ''.join(data[i][4])})
        csvout.writerows(list_out)