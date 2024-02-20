from app.util.DatabaseConnection import DatabaseConnection
import pandas as pd
import ast
from joblib import load
from flask import Flask, request, jsonify


def print_problem_info(problem_id, cursor):
    select_sql = f"""
        SELECT
            problem.name AS problem_name,
            problem.id AS problem_id,
            problem.difficulty_id AS problem_tier,
            GROUP_CONCAT(problem_category.category_id) AS categories
        FROM problem
        LEFT JOIN problem_category ON problem.id = problem_category.problem_id
        WHERE problem.id = {problem_id}
        GROUP BY problem.id
        """

    cursor.execute(select_sql)
    result = cursor.fetchone()

    if result:
        print(problem_id, sep=" ")
        return problem_id

    return None


def recommend_similar_problems(problem_id, max_neighbors=30, max_recommendations=5, df=None, categories_encoded=None,
                               nn=None, input_select=None):
    # 주어진 문제의 인덱스 찾기
    try:
        # 주어진 문제의 인덱스 찾기
        index = df.index[df['problem_id'] == problem_id].tolist()[0]
    except IndexError:
        print(f"Error: (id={problem_id})문제는 카테고리가 존재하지 않아 유사한 문제를 추천할 수 없습니다.")
        return []

    # 주어진 문제의 난이도
    problem_difficulty = df['problem_tier'].iloc[index]

    # 주어진 문제의 카테고리 벡터
    problem_vector = categories_encoded[index].reshape(1, -1)

    # 유사한 문제 찾기
    distances, indices = nn.kneighbors(problem_vector, n_neighbors=max_neighbors)

    # 유사한 문제들의 난이도와 ID 추출
    similar_difficulties = df['problem_tier'].iloc[indices[0]].values
    similar_problem_ids = df['problem_id'].iloc[indices[0]].values

    # 난이도 조건을 만족하는 문제 필터링
    if input_select == 1:
        filtered_indices = [j for j, diff in enumerate(similar_difficulties) if abs(diff - problem_difficulty) <= 2]
    else:
        filtered_indices = [j for j, diff in enumerate(similar_difficulties) if
                            diff - problem_difficulty > 0 and diff - problem_difficulty <= 10]
    filtered_ids = similar_problem_ids[filtered_indices]

    # 거리에 따라 정렬된 상태이므로, 최대 추천 개수만큼 잘라서 반환
    recommended_ids = filtered_ids[:max_recommendations]

    return recommended_ids


app = Flask(__name__)


@app.route('/api')
def home():
    df = pd.read_csv('problem_information.csv')

    # categories 열을 실제 리스트 형태로 변환
    df['categories'] = df['categories'].apply(lambda x: ast.literal_eval(x))

    # 저장된 모델을 파일로부터 불러오기
    nn = load('nn_model.joblib')

    # 저장된 MultiLabelBinarizer 인스턴스 불러오기
    mlb_loaded = load('mlb.joblib')

    # 저장된 원-핫 인코딩된 데이터 불러오기
    categories_encoded = load('categories_encoded.joblib')

    cursor = DatabaseConnection().cursor()

    input_id = int(request.args.get('id'))
    print(input_id, sep=" ")
    # 1: 유사한 문제들 중 난이도가 유사한 문제 추천, 2: 유사한 문제들 중 난이도가 더 어려운 문제 추천)
    input_select = int(request.args.get('type'))
    print(input_select, sep=" ")

    print("\n입력된 문제의 정보")
    print_problem_info(input_id, cursor)

    recommended_problem_ids = recommend_similar_problems(input_id, 30, 5, df, categories_encoded, nn, input_select).tolist()

    print(recommended_problem_ids)
    # return 'OK'
    return jsonify(recommended_problem_ids)


app.run()
