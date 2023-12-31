# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import json

# import joblib
#
# with open("tier_solution.json", "r", encoding='utf-8') as file:
#     data = json.load(file)
#
# # 피처와 타겟 분리
# X = [item["solution_code"] for item in data]  # 입력 코드
# y = [item["problem_tier"] for item in data]   # 문제 난이도
#
# # 데이터 벡터화
# vectorizer = CountVectorizer()
# X_vectorized = vectorizer.fit_transform(X)
#
# # 훈련 및 테스트 세트 분리
# X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
#
# # 모델 훈련
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# joblib.dump(model, "./sklearn_model.pkl")
# joblib.dump(vectorizer, './vectorizer.pkl')
# # 예측 및 성능 평가
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# v2

import json
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm

# CodeBERT 모델과 토크나이저 불러오기
model_name = "microsoft/codebert-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)


def encode_code(code):
    inputs = tokenizer(code, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()

# 데이터 불러오기
with open("tier_solution.json", "r", encoding='utf-8') as file:
    data = json.load(file)

# 피처와 타겟 분리
X = [encode_code(item["solution_code"]) for item in tqdm(data, desc="Processing items")]

y = [item["problem_tier"] for item in data]                # 문제 난이도
print(len(X), len(y))
joblib.dump(X, 'encoded_data.joblib')
# 훈련 및 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest 모델 훈련
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


