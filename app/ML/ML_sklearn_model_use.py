import json
from app.util.DatabaseConnection import DatabaseConnection
import re
import joblib
from sklearn.feature_extraction.text import CountVectorizer
model = joblib.load("sklearn_model.pkl")


DatabaseConnection.startTransaction()

cursor = DatabaseConnection().cursor()

select_sql = """
    SELECT
        problem.id AS problem_id, 
        problem.platform_difficulty_id AS problem_tier,
        solution.code AS solution_code
    FROM problem
    LEFT JOIN solution ON problem.id = solution.problem_id
    WHERE problem.platform_id != 1 AND solution.problem_id IS NOT NULL AND solution.code <> 'error'
    ORDER BY problem.solved_count
    """


cursor.execute(select_sql)

# 결과 가져오기 및 출력
results = cursor.fetchall()

X_data = []
# 저장된 벡터라이저 불러오기
vectorizer = joblib.load('vectorizer.pkl')

# 학습 단계에서 사용된 동일한 어휘를 사용하여 데이터 벡터화


for row in results:
    code = row[2]
    match = re.search(r'#include.*?int main.*?return 0;.*?}', code, re.DOTALL)
    if match:
        code = match.group(0)
    else:
        match2 = re.search(r'#include.*?solution.*?return ans.*?}', code, re.DOTALL)
        if match2:
            code = match2.group(0)
    code = re.sub(r'//.*?\n|/\*.*?\*/', '', code, flags=re.DOTALL)
    # 주석 제거 및 줄바꿈 통일
    code = re.sub(r'\s*\n\s*', '\n', code)

    X_sample = vectorizer.transform([code])

    # 예측 수행
    tier_pred = model.predict(X_sample)
    cursor = DatabaseConnection().cursor()
    # INSERT 쿼리 실행

    insert_sql = f"UPDATE problem SET difficulty_id = {tier_pred[0]} WHERE id = {row[0]}"
    cursor.execute(insert_sql)
    DatabaseConnection().commit()  # 중요: 데이터베이스에 변경 사항을 저장합니다.
    cursor.close()


# JSON 포맷으로 파일에 저장
# with open("tier_solution.json", "w") as file:
#     json.dump(data, file, indent=4, ensure_ascii=False)


DatabaseConnection().close()