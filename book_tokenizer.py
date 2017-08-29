#-*- coding: utf-8 -*-
import csv
import codecs
import numpy as np
import tensorflow as tf
import operator
from konlpy.tag import Twitter
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec


BOOK_INDEX = 957    # 입력 데이터의 갯수 ( csv 파일 - 1 )
VOCA_RANK = 7       # 1 ~ x위 까지 뽑는 단어
VECTOR_SIZE = 10    # 단어 벡터화시 차원수
TEST = 5            # 테스트할 데이터, 밑에서 x 번째

pos_tagger = Twitter()
def tokenize(doc):
    return [t for t in pos_tagger.pos(doc, norm=True, stem=True)]
f = codecs.open('book_data_957.csv', 'r', encoding='utf-8', errors='ignore')

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
                   'vol','code','footer','script','tracking','거래처', '대학교', '저자', '대해', '대학교', '루', '동아', '학교',
                   '지은', '대학', '강정연', '웬', '두', '주인공', '미니', '윤가영', '배병옥', '하라', '최재천', '오세정', '에서', '성공회대', '한양대', '진짜', '이승연', '백진호',
                   '최재승', '조영식', '이케다', '이사쿠', '연놈', '로서', '산이', '어디', '통해', '어디', '집필', '상재영', '재영']

stopword = len(book_stop_words)
count = -1
noun = [[] for row in range(10000)]    # 전체 데이터 + title
category = []
list_out = []

## 토큰화 진행
for i in data:
    count += 1
    category.append(i[0])
    temp = []
    temp.append(i[1] * 3)
    temp.append(i[2])
    temp.append(i[3])
    for j in range(3):
        doc = [tokenize(temp[j])]
        # print(j, doc)
        for k in range (0, len(doc[0])):
            if(doc[0][k][1] == "Noun" and len(doc[0][k][0]) != 1):
                check = 0
                for l in range (stopword):
                    if(doc[0][k][0] == book_stop_words[l]):
                        check = 1
                if(check == 0):
                    try:
                        noun[count].append(doc[0][k][0])
                    except:
                        print("====================//")
                        print(j,k,l,count)
                        exit(0)
f.close()

category.pop(0)

# 카테고리별 데이터 정보 확인
category_freq = {}  # dictionary로 빈도수 확인
for word in category:
    count = category_freq.get(word, 0)
    category_freq[word] = count + 1
category_freq = sorted(category_freq.items(), key=operator.itemgetter(1), reverse=True)  # 빈도수를 확인해서 내림차순으로 정렬
print(category_freq)        # 카테고리별 데이터 갯


top7 = [[[] for i in range(VOCA_RANK)] for j in range(BOOK_INDEX)]  # 빈도수 상위 7개의 값만 가져옴, 책 index / 순위 / 제목or벡터값
for index in range(1,BOOK_INDEX+1):         # 책 index 횟수만큼 반복
    frequency = {}                          # dictionary로 빈도수 확인
    for word in noun[index]:
        count = frequency.get(word, 0)
        frequency[word] = count + 1

    try:
        freq = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)  # 빈도수를 확인해서 내림차순으로 정렬
        for i in range(VOCA_RANK):
            top7[index-1][i] = freq[i][0]          # 빈도수 높은 7개 단어를 input 배열에 넣는다
    except:
        ''''''
# print("============================")
# print(top7)


### y_data 생성 과정
## Category 단어를 one-hot으로 바꿈
feature_labels = np.array(category)
encoder = LabelEncoder()
encoder.fit(feature_labels)
feature_labels = encoder.transform(feature_labels)
# print("index : ", feature_labels)     ## category의 분류되는 클레스 index 확인
print("class : ", encoder.classes_)     ## category의 분류되는 클레스 값 확인

output_len = len(encoder.classes_)

## y_data에 index * rank 갯수 곱한만큼의 배열 생성
## 단어 하나하나 마다에 맞는 one-hot 값 생성
y_data = [[0 for i in range(output_len)] for j in range(BOOK_INDEX*VOCA_RANK)]
for i in range(len(feature_labels)):
    for j in range(VOCA_RANK):
        try:
            feature_index = feature_labels[i]
            y_data[i*VOCA_RANK+j][feature_index] += 1        # one-hot 방식으로 변경
        except:
            print(i, j, len(feature_labels))

vector = [[[] for j in range(VOCA_RANK)] for i in range(BOOK_INDEX)]

model = Word2Vec(top7, size=VECTOR_SIZE, min_count=1, iter=100, sg=1)
for i in range(BOOK_INDEX):
    for j in range(VOCA_RANK):
            vector[i][j] = model.wv[top7[i][j]]
# print(len(vector), vector)
x_data = [[0 for i in range(VECTOR_SIZE)] for j in range(BOOK_INDEX * VOCA_RANK)]

# 벡터화 된 값을 2차원 배열로 변경
for i in range(BOOK_INDEX):
    for j in range(VOCA_RANK):          # 1-5위
        for k in range(VECTOR_SIZE):     # vector size
            try:
                x_data[i*VOCA_RANK+j][k] = vector[i][j][k]
            except:
                print(i,j,k)

# print(len(x_data), len(y_data))
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
training_epoches = 50
batch_size = 14

x_data = np.array(x_data)
y_data = np.array(y_data)

x_train = x_data[0:(BOOK_INDEX*VOCA_RANK-VOCA_RANK*TEST), :]
y_train = y_data[0:(BOOK_INDEX*VOCA_RANK-VOCA_RANK*TEST), :]

x_test = x_data[(BOOK_INDEX*VOCA_RANK-VOCA_RANK*TEST):(BOOK_INDEX*VOCA_RANK-VOCA_RANK*TEST) + VOCA_RANK, :]
y_test = y_data[(BOOK_INDEX*VOCA_RANK-VOCA_RANK*TEST):(BOOK_INDEX*VOCA_RANK-VOCA_RANK*TEST) + VOCA_RANK, :]

train_x_batch, train_y_batch = tf.train.batch([x_data[0:(BOOK_INDEX*VOCA_RANK-VOCA_RANK*TEST), :], y_data[0:(BOOK_INDEX*VOCA_RANK-VOCA_RANK*TEST), :]], batch_size=70)
# print(x_test)
# print(y_test)

X = tf.placeholder("float", [None, VECTOR_SIZE])
Y = tf.placeholder("float", [None, output_len])

W1 = tf.get_variable("W1", shape=[VECTOR_SIZE, 15], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([15]), name='bias1')

W2 = tf.get_variable("W2", shape=[15, 20], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([20]), name='bias2')

W3 = tf.get_variable("W3", shape=[20, output_len], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([output_len]), name='bias3')

# h1은 첫번째 히든계층의 값들 계산
h1 = tf.matmul(X, W1) + b1
h2 = tf.matmul(h1, W2) + b2
hypothesis = tf.nn.softmax(tf.matmul(h2,W3) + b3)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=1.0e-2).minimize(cost)
prediction = tf.arg_max(hypothesis, 1)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
        if step % 200 == 0:
            print("cost : ", cost_val)

    print("테스트 데이터 : ", top7[BOOK_INDEX-TEST])
    print("단어벌 예측값 :  " ,end="")
    for i in range(VOCA_RANK):
        print(encoder.classes_[sess.run(prediction, feed_dict={X: x_test})[i]], end=" ")
    print("\n실제 카테고리 : ", category[BOOK_INDEX-TEST])


'''
['가정/생활/요리' '건강' '경제/경영' '과학/공학' 
'국어/외국어' '만화' '사전' '사회' 
'소설' '시/에세이' '어린이' '여행/지도' 
'역사/문화' '예술/대중문화' '유아' '인문' 
'자기계발' '잡지' '종교' '취미/레저' 
'취업/수험서' '컴퓨터/IT' '학습/참고서']
'''