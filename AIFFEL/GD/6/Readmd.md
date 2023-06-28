# 230628

아이펠캠퍼스 온라인4기 피어코드리뷰
.
- 코더 : 김현희
- 리뷰어 : 임지혜
-------------------------------------------------- -----------

PRT(PeerReviewTemplate)

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  >num_words를 None, 5000등 변경해가며 8가지 모델에 대한 결과를 도출하고, f1-score계산까지 완료 

- [O] 주석을 보고 작성자의 코드가 이해되었나요?
```python
# 데이터도 벡터화과정을 단계별로 주석 처리 
x_test_dtm = dtmvector.transform(x_test) #테스트 데이터를 DTM으로 변환
tfidfv_test = tfidf_transformer.transform(x_test_dtm) #DTM을 TF-IDF 행렬로 변환
```

- ['X] 코드가 에러를 유발할 가능성이 있나요?
  > (모델과 데이터의 반복으로 에러날 가능성 약간 있음)

- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
  >딥러닝 모델 적용에서 46개 클래스 구분문제에 맞게 loss, metric, activation fn 선언
  >모데를의 변수(eg. C, max_iter등)들을 변경해가며 최적값 도출 

- [O] 코드가 간결한가요?
```python
# 소프트보팅; 관련 함수들 약어 써서 간결하게 정리 
voting_classifier = VotingClassifier(estimators=[
         ('lr', LogisticRegression(C=26506, penalty='l2')),
        ('cnb', ComplementNB()),
        ('grbt', GradientBoostingClassifier(random_state=0)),
], voting='soft', n_jobs=-1)
voting_classifier.fit(tfidfv, y_train)
```

----------------------------------------------
