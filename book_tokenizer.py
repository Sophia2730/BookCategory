#-*- coding: utf-8 -*-
import csv
import codecs
import numpy as np
import tensorflow as tf
from konlpy.tag import Twitter
import math, operator
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec


BOOK_INDEX = 71
VECTOR_SIZE = 30

def tf_(word, blob):
    return blob.count(word) / len(blob)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf_(word, blob) * idf(word, bloblist)


pos_tagger = Twitter()
def tokenize(doc):
    return [t for t in pos_tagger.pos(doc, norm=True, stem=True)]
f = codecs.open('book_data.csv', 'r', encoding='utf-8', errors='ignore')

data = csv.reader(f,delimiter=",")

book_stop_words = ['class', '영역', '시작', '분야상세보기' ,'배너' ,'gif', 'src','img','하위',
                   'http' , 'href', 'sysimage', '상품', '카테고리','alt','bl_arr','blank','height','width',
                   '정보', 'image', 'alt', 'morecate', 'renew','더보기','교환','빠른분야', 'com','bl_arrr','dvd',
                   '비주얼','반품','안내','관련','행사','body','분야','보기','도서','메뉴','상세','중고','할인',
                   '공연','영화','도서','배송','판매','가격','포인트','구입','예매','상세','소개','이미지','레이어',
                   '문의','광고','예스','패션','이벤트','국내','연재','어린이','음반','서비스','리뷰','소비자','키즈',
                   '해외','코너','쿠폰','플래티넘','로얄','골드','제휴','페이지','외국','해외','추가','경우','만원','바로가기',
                   '페이지','비주','플레티넘','오버레이','링크','블루레이','평일','공휴일','기간','스크립트','회원','그림','텍스트',
                   '버튼','헤더','품절','마이','수입','센터','메일','수량','이상','한정','취소','적립','다운로드','카트','검색',
                   '주문','채널','센터','구매','고객','마케팅','바로','해더','제목','사이트','세부','환불','휴무','배송비',
                   'vol','code','footer','script','tracking','거래처', '년', '수', '책', '등', '간', '내', '안', '의', '회', '타', '제', '권', '를', '그', '띠', '이', '소',
                   '대학교', '백', '윤', '저자', '대해', '대학교', '루', '동아', '학교', '로', '노', '속', '와', '반', '작가', '것', '레', '석', '주', '중', '교수', '탈', '홍승표', '진짜',
                   '뒤', '지은', '대학', '강정연', '웬', '두', '주인공', '미니', '윤가영', '배병옥', '하라', '최재천', '오세정', '짓', '에서', '성공회대', '한양대', '진짜', '이승연', '백진호',
                   '최재승']
# if (len(data)==1):
#     stopword = data
stopword = len(book_stop_words)
count = -1
noun = [[] for row in range(BOOK_INDEX)]    # 전체 데이터
category = []
title = []
sent = []
list_out = []

for i in data:
    count += 1
    category.append(i[0])
    title.append(i[1])
    temp = []
    temp.append(i[1])
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

input = [[[] for i in range(5)] for j in range(BOOK_INDEX)]  # 상위 10개의 값만 가져옴, 책 index / 순위 / 제목or벡터값
for index in range(1,BOOK_INDEX):       # 책 index 횟수만큼 반복해서 학습
    frequency = {}
    for word in noun[index]:
        count = frequency.get(word, 0)
        frequency[word] = count + 1

    try:
        # 책 index, 1-10 단어 , tf-idf
        # 빈도수를 확인해서 내림차순으로 정렬
        freq = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
        # print(freq)
        for i in range(5):
            input[index-1][i] = [freq[i][0], tfidf(freq[i][0], noun[index], noun)]
    except:
        ''''''

category.pop(0)
# print(input[0])
# print(category[0])

feature_labels = np.array(category)
encoder = LabelEncoder()
encoder.fit(feature_labels)
feature_labels = encoder.transform(feature_labels)
# print("feature_labels : ", feature_labels)
# print("class : ", encoder.classes_)     ## output class 확인

y_data = [[0 for i in range(25)] for j in range(BOOK_INDEX*5-5)]

for i in range(len(feature_labels)):
    for j in range(5):
        feature_index = feature_labels[i]
        y_data[i*5+j][feature_index] += 1        # one-hot 방식으로 변경

# print(arr)      ## output으로 줄 onehot 값
# print(arr[1].index(1))     ## 출력후 값이 1인 index 확인하는 방법

top5 = [[0 for j in range(6)] for i in range(BOOK_INDEX-1)]
vector = [[0 for j in range(5)] for i in range(BOOK_INDEX-1)]
for i in range(BOOK_INDEX-1):
    for j in range(10):
        try:
            top5[i][j] = input[i][j][0]
        except:
            ''''''
    try:
        top5[i][5] = category[i]
    except:
        ''''''

model = Word2Vec(top5, min_count=1, size=VECTOR_SIZE)
# print(model.wv["인공"])     # 인공 단어의 벡터화

for i in range(BOOK_INDEX-1):
    for j in range(5):
        vector[i][j] = model.wv[top5[i][j]]

print("top5: ", top5)
x_data = [[0 for i in range(VECTOR_SIZE)] for j in range((BOOK_INDEX-1) * 5)]

# 벡터화 된 값을 2차원 배열로 변경
for i in range(70):
    for j in range(5):          # 1-5위
        for k in range(VECTOR_SIZE):     # vector size
            x_data[i*5+j][k] = vector[i][j][k]

# print("x_data : ",x_data)
# print("y_data : ",y_data)

'''
input   : 빈도수 1-10위 단어 tf-idf 값
output  : 카테고리 ( one-hot 사용, 약 25개의 카테고리를 가짐 )
['가정/생활/요리' '건강' '경제/경영' '과학/공학' '국어/외국어' '만화' '사회' '소설' '시/에세이' '어린이'
 '여행/지도' '역사/문화' '예술/대중문화' '유아' '인문' '자기계발' '종교' '취업/수험서' '컴퓨터/IT' '학습/참고서']
'''
##########
## 학습  ##
##########
X = tf.placeholder("float", [None, 30])
Y = tf.placeholder("float", [None, 25])

W1 = tf.Variable(tf.random_normal([30, 15]), name='weight1')
b1 = tf.Variable(tf.random_normal([15]), name='bias1')

W2 = tf.Variable(tf.random_normal([15, 15]), name='weight2')
b2 = tf.Variable(tf.random_normal([15]), name='bias2')

W3 = tf.Variable(tf.random_normal([15, 25]), name='weight3')
b3 = tf.Variable(tf.random_normal([25]), name='bias3')

# h1은 첫번째 히든계층의 값들 계산
h1 = tf.matmul(X, W1) + b1
h2 = tf.matmul(h1, W2) + b2
hypothesis = tf.nn.softmax(tf.matmul(h2,W3) + b3)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
prediction = tf.arg_max(hypothesis, 1)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
x_test = [[-0.0105449,-0.00733523,0.00259471,0.000239543,-0.00332217,0.0129727,-0.00750322,0.000371167,-0.0136839,0.012156,-0.0074239,0.0151548,-0.0121081,0.013814,-0.00645747,0.00233305,-0.00772442,-0.00130056,0.016141,0.00198796,0.000809719,0.000336766,0.00805131,0.015156,-0.00407055,-0.000148496,0.00198379,-0.0131182,-0.0029102,-0.0096999], [-0.0115755,-0.00523591,0.0146293,0.0126481,-0.00554028,0.01307,0.00958671,0.00773702,-0.00217445,-0.00948835,0.0119835,-0.00318063,0.0166088,-0.0110419,-0.009804,0.0109573,-0.00697838,-0.00495963,-0.00716577,0.00796535,-0.0115131,-0.0149793,0.0163886,0.0102798,-0.00477338,0.0118827,-0.0137196,-0.0135935,0.0125603,0.00364718], [-0.00855128,0.00845256,-0.00670937,0.00260967,0.011948,0.0102638,-0.0159786,-0.015011,0.00148017,-0.00123284,0.00276484,-0.012006,0.0145319,-0.0151017,0.00837444,0.00833535,0.0064657,-0.00823254,-0.00462886,-0.00584191,-0.0103965,0.00966934,0.0166442,-0.0132585,0.0080022,-0.00280734,0.00270832,0.00989533,-0.00436177,-0.00292181], [0.00160347,-0.0109371,-0.0158884,-0.00955908,-0.000769243,-0.00984238,0.00784462,-0.00483038,0.00750916,-0.00992298,-0.00617114,0.0138138,0.00904788,0.0113368,0.00887151,-0.0110659,-0.00427228,-0.0023889,-0.0152698,-0.0166286,0.00368374,0.00878231,0.0096055,-0.0139757,-0.00532839,-0.00927704,0.00797618,-0.0101676,0.00219013,0.00894748], [-0.0133681,-0.0104627,-0.000730098,-0.00982444,-0.0102341,-0.0147977,0.00220372,-0.011808,0.00810674,0.000801531,-0.0132926,0.00537094,-0.0142046,0.00855081,-0.00679808,-0.012949,0.0108963,0.0136892,-0.0137903,-0.00976883,-0.0082311,-0.00402375,-0.00933076,-0.00705463,-0.00461182,0.0134919,-0.0124138,0.00150455,0.00890511,0.0131741]]
y_test = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print("cost : ", cost_val)


    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))


'''
결과에 따른 index 해당 값
['가정/생활/요리' '건강' '경제/경영' '과학/공학'
 '국어/외국어' '만화' '사회' '소설' 
 '시/에세이' '어린이' '여행/지도' '역사/문화' 
 '예술/대중문화' '유아' '인문' '자기계발'
 '종교' '취업/수험서' '컴퓨터/IT' '학습/참고서']
'''