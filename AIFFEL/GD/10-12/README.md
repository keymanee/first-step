# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 김현희
- 리뷰어 : 소용현


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [o] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  ![image](https://github.com/keymanee/first-step/assets/100551891/262dc0e4-0f9d-445b-af9f-034f6cd4a342)

- [o] 2.주석을 보고 작성자의 코드가 이해되었나요?
```
data_dir = os.getenv('HOME')+'/aiffel/transformer/data'
kor_path = data_dir+"/korean-english-park.train.ko"
eng_path = data_dir+"/korean-english-park.train.en"

# 데이터 정제 및 토큰화
def clean_corpus(kor_path, eng_path):
    with open(kor_path, "r") as f: kor = f.read().splitlines()
    with open(eng_path, "r") as f: eng = f.read().splitlines()
    assert len(kor) == len(eng) # kor eng 쌍

    cleaned_corpus = []
    for sen_pair in zip(kor, eng):
        cleaned_corpus.append(sen_pair)
    cleaned_corpus = list(set(cleaned_corpus))
    return cleaned_corpus

cleaned_corpus = clean_corpus(kor_path, eng_path)
```
네 이해되었습니다.
- [x] 3.코드가 에러를 유발할 가능성이 있나요?
```
def positional_encoding(pos, d_model):
    def cal_angle(position, i):
        return position / np.power(10000, int(i) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, i) for i in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return sinusoid_table
```
형변환등을 통해 에러 유발 가능성을 줄였다.
- [o] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?  
번역기에 적합한 트랜스포머 모델을 활용하고, spm토크나이저를 활용하여 작성하였다.
- [o] 5.코드가 간결한가요?
```
examples = [
    "오바마는 대통령이다.",
    "시민들은 도시 속에 산다.",
    "커피는 필요 없다.",
    "일곱 명의 사망자가 발생했다."
]
```
테스트 문장을 배열로 나열하여 코드를 간결하게 표현하였다.

