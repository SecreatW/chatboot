import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein_Distance import calc_distance

class SimpleChatBot:
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)  # 질문을 TF-IDF로 변환


    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
        answers = data['A'].tolist()   # 답변열만 뽑아 파이썬 리스트로 저장
        return questions, answers

    def find_best_answer(self, input_sentence):
        # input_sentence
        a = sorted(self.questions, key=lambda n: calc_distance(input_sentence, n)) # 질문 데이터들과 실제 질문을 레벤 슈타인으로 매치
        distence= calc_distance(input_sentence,a[0]) # 1. 가장 낮은 질문의 유사도(a[0])를 구하여 레벤슈타인 거리를 이용해 구하기
        index = self.questions.index(a[0]) # 2. chat의 다량의 질문 데이터들 중 레벤슈타인 유사도가 가장 낮은 거리의 질문의 인덱스를 구함.
        # print('Distence =', distence)
        # print('Index =', index)
        input_vector = self.vectorizer.transform([input_sentence]) # 질문 문장 벡터화.
        similarities = cosine_similarity(input_vector, self.question_vectors) # 코사인 유사도 값들을 저장
        best_match_index = similarities.argmax()   # 유사도 값이 가장 큰 값의 인덱스를 반환
        return self.answers[best_match_index], self.answers[index]

# CSV 파일 경로를 지정하세요.
filepath = 'ChatbotData.csv'

# 간단한 챗봇 인스턴스를 생성합니다.
chatbot = SimpleChatBot(filepath)

# '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복합니다.
while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    cos_response, lev_response = chatbot.find_best_answer(input_sentence)
    print('Cosine chatbot:', cos_response) # 코사인 유사도의 가장 큰 값의 인덱스를 정답 데이터의 인덱스로 적용 후 출력.
    print('Levenshtein chatbot:', lev_response) # chat의 다량의 질문 데이터들 중 레벤슈타인 유사도가 가장 낮은 거리의 질문의 인덱스로 정답 데이터의 인덱스로 적용 후 출력.


