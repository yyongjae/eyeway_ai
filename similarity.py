# -*- coding: utf-8 -*-


# 필요한 것들 import
#!pip install spacy
import spacy
#!python -m spacy download ko_core_news_sm
nlp = spacy.load("ko_core_news_sm")
import numpy as np
import pandas as pd


# 5호선, 7호선 지하철 역 이름 정리 리스트
station_name_list = ['청구', '신금호', '행당', '왕십리', '마장', '답십리', '장한평', '군자', '아차산', '미사', '거여', '광나루', '천호', '강동', '길동', '굽은다리', '명일', '고덕', '상일동', '둔촌동', '올림픽공원', '방이', '오금', '개롱', '하남검단산', '하남시청', '하남풍산', '강일', '마천', '방화', '개화산', '김포공항', '송정', '마곡', '발산', '우장산', '화곡', '까치산', '신정', '목동', '오목교', '양평', '영등포구청', '영등포시장', '신길', '여의도', '여의나루', '마포', '공덕', '애오개', '충정로', '서대문', '광화문', '종로3가', '을지로4가', '동대문역사문화공원', '남성', '이수', '내방', '고속터미널', '반포', '논현', '학동', '강남구청', '청담', '뚝섬유원지', '건대입구', '어린이대공원', '군자', '중곡', '용마산', '산곡', '부평구청', '굴포천', '중화', '사가정', '면목', '상봉', '상동', '부천시청', '신중동', '춘의', '부천종합운동장', '까치울', '마들', '노원', '중계', '먹골', '하계', '공릉', '태릉입구', '석남', '삼산체육관', '수락산', '도봉산', '장암', '온수', '천왕', '광명사거리', '철산', '가산디지털단지', '남구로', '대림', '신풍', '보라매', '신대방삼거리', '장승배기', '상도', '숭실대입구']

# 문자열 유사도 측정
# - Levenshtein 거리 => 거리는 초성/중성/종성을 나눠서 진행
# - Jaccard 유사성 지수

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def korean_to(korean_word):
    r_lst = []
    for w in list(korean_word.strip()):
        if '가'<=w<='힣':
            ch1 = (ord(w) - ord('가'))//588
            ch2 = ((ord(w) - ord('가')) - (588*ch1)) // 28
            ch3 = (ord(w) - ord('가')) - (588*ch1) - 28*ch2
            if(ch3 == 0):
              r_lst.append([CHOSUNG_LIST[ch1], JUNGSUNG_LIST[ch2]])
            else:
              r_lst.append([CHOSUNG_LIST[ch1], JUNGSUNG_LIST[ch2], JONGSUNG_LIST[ch3]])
        else:
            r_lst.append([w])
    return r_lst

def levenshtein_distance(str1, str2):
    # 문자열 길이 계산
    row_len = len(str1) + 1
    col_len = len(str2) + 1

    # 빈 행렬 초기화, List comprehension 으로 2차 배열 선언쓰
    matrix = [[0 for _ in range(col_len)] for _ in range(row_len)]

    # 행렬 초기 설정
    for row in range(row_len):
        matrix[row][0] = row

    for col in range(col_len):
        matrix[0][col] = col

    # Levenshtein 거리 계산
    for i in range(1, row_len):
        for j in range(1, col_len):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,      # 삭제
                matrix[i][j - 1] + 1,      # 삽입
                matrix[i - 1][j - 1] + cost  # 대체
            )
    return matrix[-1][-1]


# 2. Levenshtein

def Levenshtein_similarity(_ocr_test):
    distances = {}
    for station_name in station_name_list:
      distance = levenshtein_distance(korean_to(_ocr_test), korean_to(station_name))
      distances[station_name] = distance

    # 유사성이 가장 높은 역 찾기
    most_similar_station = min(distances, key=distances.get)
    similarity_score = distances[most_similar_station]
    return min(distances, key=distances.get)


# Jaccard 유사도(Similarity)
# 두 집합 간의 유사성을 측정하는 메트릭, 두 문자열을 각각 문자 단위로 나눈 후, 공통 문자의 비율 계산

def jaccard(str1, str2):
    # 두 문자열을 문자(또는 단어) 집합으로 변환
    set1 = set(str1)
    set2 = set(str2)

    # 자카드 유사도 계산
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    similarity = intersection / union

    return similarity


def jaccard_similarity(_ocr_test):
    ja_similarities = {}
    for station_name in station_name_list:
        ja_similarity = jaccard(_ocr_test, station_name)
        ja_similarities[station_name] = ja_similarity

    # 유사성이 가장 높은 역 찾기
    most_similar_station = max(ja_similarities, key=ja_similarities.get)
    similarity_score = ja_similarities[most_similar_station]
    return similarity_score, most_similar_station

#################################################################

# 아래의 변수에 확인하고자 하는 변수를 넣어주세요.
ocr_test = "타는곳"

def find_same_string(i, str1, str2):
  if(i == 0):
    return None
  if str1 == str2:
      return str1
  else:
      return str1

# a, s = jaccard_similarity(ocr_test)
# print(find_same_string(a, s, Levenshtein_similarity(ocr_test)))