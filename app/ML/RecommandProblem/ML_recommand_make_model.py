import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from joblib import dump
import ast

# CSV 파일 로드
df = pd.read_csv('problem_information.csv')

# categories 열을 실제 리스트 형태로 변환
df['categories'] = df['categories'].apply(lambda x: ast.literal_eval(x))

# 원-핫 인코딩 처리
mlb = MultiLabelBinarizer()
categories_encoded = mlb.fit_transform(df['categories'])

# NearestNeighbors 모델 초기화 및 학습
nn = NearestNeighbors(n_neighbors=30, metric='cosine')  # 30개의 가장 유사한 문제를 찾기 위해 n_neighbors=30으로 설정
nn.fit(categories_encoded)


# 모델을 파일로 저장
dump(nn, 'nn_model.joblib')

# MultiLabelBinarizer 인스턴스 저장
dump(mlb, 'mlb.joblib')

# 원-핫 인코딩된 데이터 저장
dump(categories_encoded, 'categories_encoded.joblib')


