#-*- coding: utf-8 -*-
import csv
import codecs
import numpy as np
from konlpy.tag import Twitter
import math, operator


def tf(word, blob):
    return blob.count(word) / len(blob)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

pos_tagger = Twitter()
def tokenize(doc):
    return [t for t in pos_tagger.pos(doc, norm=True, stem=True)]
f = codecs.open('book_data.csv', 'r', encoding='utf-8', errors='ignore')

data = csv.reader(f,delimiter=",")

book_stop_words = ['class', '영역', '시작', '분야상세보기' ,'배너' ,'gif', 'src','img','하위','http' , 'href', 'sysimage', '상품', '카테고리','alt','bl_arr','blank','height','width','정보', 'image', 'alt', 'morecate', 'renew','더보기','교환','빠른분야', 'com','bl_arrr','dvd', '비주얼','반품','안내','관련','행사','body','분야','보기','도서','메뉴','상세','중고','할인','공연','영화','도서','배송','판매','가격','포인트','구입','예매','상세','소개','이미지','레이어','문의','광고','예스','패션','이벤트','국내','연재','어린이','음반','서비스','리뷰','소비자','키즈','해외','코너','쿠폰','플래티넘','로얄','골드','제휴','페이지','외국','해외','추가','경우','만원','바로가기','페이지','비주','플레티넘','오버레이','링크','블루레이','평일','공휴일','기간','스크립트','회원','그림','텍스트','버튼','헤더','품절','마이','수입','센터','메일','수량','이상','한정','취소','적립','다운로드','카트','검색','주문','채널','센터','구매','고객','마케팅','바로','해더','제목','사이트','세부','환불','휴무','배송비','vol','code','footer','script','tracking','거래처']
stopword = len(book_stop_words)
count = -1
noun = [[] for row in range(20)]
category = []
title = []
sent = []

for i in data:
    count += 1
    category.append(i[0])
    title.append(i[1])
    temp = []
    temp.append(i[2])
    temp.append(i[3])
    temp.append(i[4])
    sent.append(temp)
    for j in range(3):
        doc = [tokenize(temp[j])]
        for k in range (0, len(doc[0])):
            if(doc[0][k][1] == "Noun"):
                check = 0
                for l in range (stopword):
                    if(doc[0][k][0] == book_stop_words[l]):
                        check = 1
                if( check == 0):
                    noun[count].append(doc[0][k][0])
f.close()
print(category[2], title[2])

frequency = {}
for word in noun[2]:
    count = frequency.get(word, 0)
    frequency[word] = count + 1

# 책 index, 1-10 단어 , tf-idf
# 책 index 횟수만큼 반복해서 학습

# 빈도수를 확인해서 내림차순으로 정렬
freq = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
print(freq[1][0], tfidf(freq[1][0], noun[2], noun))