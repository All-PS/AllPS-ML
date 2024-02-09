from app.util.DatabaseConnection import DatabaseConnection
import pandas as pd

DatabaseConnection.startTransaction()

cursor = DatabaseConnection().cursor()


select_sql = """
    SELECT
        problem.id AS problem_id,
        problem.difficulty_id AS problem_tier,
        GROUP_CONCAT(problem_category.category_id) AS categories
    FROM problem
    JOIN problem_category ON problem.id = problem_category.problem_id
    GROUP BY problem.id
    """


cursor.execute(select_sql)

# 결과 가져오기
results = cursor.fetchall()
problems_info = []
for row in results:
    problem_id = row[0]
    problem_tier = row[1]  # 난이도
    categories_str = row[2]  # 카테고리 ID들이 콤마로 구분된 문자열
    categories_list = categories_str.split(',')  # 문자열을 콤마 기준으로 나누어 리스트 생성

    # 문제 정보를 딕셔너리로 생성
    problem_info = {
        "problem_id": problem_id,
        "problem_tier": problem_tier,
        "categories": categories_list
    }
    problems_info.append(problem_info)


df_problems_info = pd.DataFrame(problems_info)

# CSV 파일로 저장
df_problems_info.to_csv('problem_information.csv', index=False)


DatabaseConnection().close()
