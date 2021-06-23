# The Machine Learning pipeline on AWS(http://bit.ly/ml-pipeline)
> 이 곳은 The Machine Learning pipeline on AWS 수업을 위한 자료 모음집 입니다.

> AWS 학습 링크집 시리즈
- AWS 트레이닝 및 자격증 소개 http://bit.ly/aws-tnc-intro
- AWS 학습 자료집 http://bit.ly/aws-study-resource
- AWS 공인 솔루션스 아키텍트 - 어소시에이트 수험 가이드  http://bit.ly/sacertguide
- AWS 공인 개발자 - 어소시에이트 수험 가이드  http://bit.ly/devcertguide
- AWS 보안 관련 컨텐츠 모음집  http://bit.ly/secontents
- AWS 기반 빅데이터 학습자료집 http://bit.ly/bdonaws
- AWS 딥러닝 학습 자료 모음집 http://bit.ly/dlonaws
- 2019년 re:Invent 에서 공개된 AI/ML관련 서비스 소개 http://bit.ly/2019-ml-recap
- AWS The Machine Learning pipeline on AWS 교육 학습 자료집 http://bit.ly/ml-pipeline

## 실습 & 워크숍
### 실습1 - Project: Bank Marketing Prediction a.k.a Capstone Project(정기예금 전화 마케팅 예측)
- [Project: Bank Marketing Prediction](https://github.com/shashankvarshney/MLND-Capstone-project-Bank-Marketing-Prediction) - 분석 및 예측 전반에 걸친 세세한 내용이 잘 정리된 github 리포지토리입니다.
- [Getting started with TensorFlow 2](https://www.coursera.org/learn/getting-started-with-tensor-flow2/home/welcome) - Coursera에 개설되어 있는 강의로 청강은 무료입니다. 5주 구성으로 되어 있으며 마지막 5주차에 Capstone프로젝트를 다룹니다. 이론과 실습의 균형이 잘 잡혀 있는 강의로 기계번역티가 많이 나긴 하지만 한글 자막도 제공됩니다.

### 워크숍 - 아마존 리뷰
- [Amazon Customer Reviews Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html)
### 워크숍 - 신용카드 사기 탐지
- [신용카드 사기 거래 감지하기](https://laboputer.github.io/machine-learning/2020/05/29/creditcardfraud/) - EDA와 데이터 클랜징 뿐만 아니라 알고리즘 선택과 모델 학습까지 정리된 블로그 글
- [Credit Fraud || Dealing with Imbalanced Datasets(Kaggle)](https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)
### 워크숍 - 비행기 지연 예측
- [Predicting flight delays(Kaggle)](https://www.kaggle.com/fabiendaniel/predicting-flight-delays-tutorial)

## Python 기초
- [파이썬 코딩도장](https://dojang.io/course/view.php?id=7)
- [Learn Python On AWS Workshop](https://learn-to-code.workshop.aws/)

## 수학 기초
- [울프럼 알파(Wolfram Alpha) 사용법](https://www.youtube.com/watch?v=3vl7QUGMRMA)
- [3Blue1Brown 한국어](https://www.youtube.com/channel/UCJK07Uk2KY9r78ksPoXg-3g) - 알기쉽게 다양한 수학 개념을 설명하는 유튜브인 3Blue1Brown이 한국어 자막과 함께 제공됩니다.

## Jupyter notebook
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [How to Use Jupyter Notebook in 2020: A Beginner’s Tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)
- [Jupyter Notebook Tutorial: Introduction, Setup, and Walkthrough(영상 튜토리얼)](https://www.youtube.com/watch?v=HW29067qVWk)

## Amazon SageMaker
- [SageMaker 설명서](https://docs.aws.amazon.com/ko_kr/sagemaker/index.html)
- [Amazon SageMaker 데모 - 김대근, AWS 데이터 사이언티스트 :: AIML 특집 웨비나](https://www.youtube.com/watch?v=miIVGlq6OUk) - 한국어로 친철하게 설명한 SageMaker 데모
- [SageMaker 셀프 스터디 가이드](https://github.com/serithemage/AWS_AI_Study/blob/master/DLonAWS/SageMaker_Self-Study_Guide.md)
- [SageMaker Notebook Instance Lifecycle Config Samples](https://github.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples)
  - [auto-stop-idle](https://github.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/tree/master/scripts/auto-stop-idle) - 1시간 이상 노트북 사용이 없을 경우 자동으로 중지시키는 스크립트
- [SageMaker 내장 알고리즘](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/algos.html)
- [SageMaker Script Mode 예제](https://github.com/aws-samples/amazon-sagemaker-script-mode)
  - [Script Mode 데모 영상](https://www.youtube.com/watch?v=x94hpOmKtXM)
    - [데모 영상에서 사용한 소스코드](https://gitlab.com/juliensimon/dlnotebooks/tree/master/keras/05-keras-blog-post)
- [SageMaker AutoPilot 예제](https://aws.amazon.com/ko/getting-started/hands-on/create-machine-learning-model-automatically-sagemaker-autopilot/)
- [SageMaker BYOC(Bring Your Own Container)](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/docker-containers.html)
  - [예제 노트북: 자체 알고리즘 또는 모델 사용](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/docker-containers-notebooks.html)
  - [한글 핸즈온 - BRING-YOUR-OWN-CONTAINER 기능 실습하기](https://www.sagemaker-workshop-kr.com/kr/sagemaker/_module_8.html)
- [AWS를 사용하여 전체 기계 학습 수명주기 설계 및 구축 : End-to-End Amazon SageMaker Demo](https://aws.amazon.com/ko/blogs/machine-learning/architect-and-build-the-full-machine-learning-lifecycle-with-amazon-sagemaker/)
  -  [End-to-End Amazon SageMaker Demo 한글화된 노트북](https://github.com/daekeun-ml/sagemaker-studio-end-to-end)

## 문제 공식화
- [핸즈온 머신러닝에서 제안한 머신러닝 프로젝트 체크리스트](https://github.com/ageron/handson-ml/blob/master/ml-project-checklist.md)
- [보이지 않는 총알자국 - 생존자 편향](https://m.blog.naver.com/PostView.nhn?blogId=shc427118&logNo=220944502924)

## 분석, 전처리, 피처 엔지니어링
- [Amazon SageMaker Ground Truth 처음 시작하기](https://aws.amazon.com/ko/getting-started/hands-on/build-training-datasets-amazon-sagemaker-ground-truth/)
- [Open source data labeling tools](https://github.com/heartexlabs/awesome-data-labeling)
- [One-Hot 인코딩 이란?](https://www.kakaobrain.com/blog/6)
- [어떤 스케일러를 쓸 것인가? StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler](https://mkjjo.github.io/python/2019/01/10/scaler.html)
- [이상치(outlier)에 대하여](https://nittaku.tistory.com/450)
  - [아웃라이어 제거](https://mkjjo.github.io/python/2019/01/10/outlier.html)
- [평균에 대한 정리(mean, median, mode)](https://blog.acronym.co.kr/401)
- [차원축소 시각화 t-SNE](https://www.youtube.com/watch?v=a__3LhLkBMw)
  - [t-SNE 개념과 사용법](https://gaussian37.github.io/ml-concept-t-SNE/)
- [boxplot 해석방법](https://codedragon.tistory.com/7012)
- [왜도와 첨도](https://m.blog.naver.com/PostView.nhn?blogId=s2ak74&logNo=220616766539&proxyReferer=https:%2F%2Fwww.google.com%2F)
- [결측치 처리](https://wooono.tistory.com/103)
- [분위수(quantile)](https://bioinformaticsandme.tistory.com/246)
- [대푯값 - 평균, 중앙값, 최빈값, 사분위수, 백분위수, 절사 평균, 이상점, 기댓값](https://namu.wiki/w/%EB%8C%80%ED%91%AF%EA%B0%92)
- [로그를 이용하여 데이터의 분포를 정규분포로 만들기](https://hong-yp-ml-records.tistory.com/28)
  - [위 문서에 나온 numpy.log1p(=log(1+x))의 모양](https://www.wolframalpha.com/input/?i=log%281+%2B+x%29)
- [정규 분포로 분포 변환 (PowerTransformer, 로그 변환)](https://wikidocs.net/83559)
- [선형 회귀 모델에서 '선형'이 의미하는 것은 무엇인가?](https://brunch.co.kr/@gimmesilver/18)
- [주성분 분석(PCA) 설명](https://angeloyeo.github.io/2019/07/27/PCA.html)
  - [차원 축소 - PCA, 주성분분석](https://excelsior-cjh.tistory.com/167) - 조금 더 자세한 설명
  - [PCA 차원 축소 알고리즘 및 파이썬 구현 (주성분 분석)](https://www.youtube.com/watch?v=DUJ2vwjRQag)
- [One-hot 인코딩 쉽게 하기](https://minjejeon.github.io/learningstock/2017/06/05/easy-one-hot-encoding.html)
- [피어슨 상관관계](https://ko.wikipedia.org/wiki/%ED%94%BC%EC%96%B4%EC%8A%A8_%EC%83%81%EA%B4%80_%EA%B3%84%EC%88%98)
- [상관계수를 이용한 특징 선택에 대하여](https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf)

### EDA in Python
- [파이썬을 이용한 탐색적 자료 분석(Exploratory Data Analysis:EDA)](https://3months.tistory.com/325)
- [Exploratory Data Analysis (EDA) in Python](https://medium.com/@atanudan/exploratory-data-analysis-eda-in-python-893f963cc0c0)

### Pandas
- [Data preprocessing with Python Pandas](https://towardsdatascience.com/data-preprocessing-with-python-pandas-part-1-missing-data-45e76b781993)
- [Pandas와 scikit-learn으로 정말 간단한 pre-processing 몇 가지 팁](https://teddylee777.github.io/scikit-learn/sklearn%EC%99%80-pandas%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EA%B0%84%EB%8B%A8-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D)
- [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling#documentation)
  - [Speed Up Your Exploratory Data Analysis With Pandas-Profiling](https://towardsdatascience.com/speed-up-your-exploratory-data-analysis-with-pandas-profiling-88b33dc53625)
  - [pandas의 극강의 라이브러리 Pandas Profiling](https://john-analyst.medium.com/pandas-%EC%9D%98-%EA%B7%B9%EA%B0%95%EC%9D%98-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC-pandas-profiling-b5187dbcbd26)
- [pandas를 SQL처럼 쓰는법](https://medium.com/jbennetcodes/how-to-rewrite-your-sql-queries-in-pandas-and-more-149d341fc53e)
- [pandas를 이용한 결측치 처리](https://rfriend.tistory.com/263)

## 모델/알고리즘 선택
- [머신 러닝의 모델 평가와 모델 선택, 알고리즘 선택](https://tensorflow.blog/%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D%EC%9D%98-%EB%AA%A8%EB%8D%B8-%ED%8F%89%EA%B0%80%EC%99%80-%EB%AA%A8%EB%8D%B8-%EC%84%A0%ED%83%9D-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%84%A0%ED%83%9D-1/)
- [Browse State-of-the-Art](https://paperswithcode.com/sota)
- [Overview: State-of-the-Art Machine Learning Algorithms per Discipline & per Task](https://towardsdatascience.com/overview-state-of-the-art-machine-learning-algorithms-per-discipline-per-task-c1a16a66b8bb)
- [XGBoost 심화학습](https://housekdk.gitbook.io/ml/ml/tabular/xgboost)

## 트레이닝
- [SageMaker 처음 사용자용 훈련 예제](https://aws.amazon.com/ko/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/)
- [SageMaker 워크숍(한글)](https://www.sagemaker-workshop-kr.com/)
  - [MODULE 5: TENSORFLOW MNIST로 자동 모델 튜닝하기](https://www.sagemaker-workshop-kr.com/kr/sagemaker/_module_5.html)
- [SageMaker 하이퍼파라미터 튜닝작업 예제](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/automatic-model-tuning-ex.html)
- [SageMaker Autopilot 사용 예제](https://aws.amazon.com/ko/getting-started/hands-on/create-machine-learning-model-automatically-sagemaker-autopilot/)
- [SageMaker Autopilot Workshop](https://www.getstartedonsagemaker.com/workshop/)


## 추론
- [Amazon SageMaker 모델 레지스트리에 모델 등록 및 배포](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/model-registry.html)
- [Amazon SageMaker 다중 모델 엔드포인트을 사용하여 여러 모델 호스팅](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/multi-model-endpoints.html)
- [Amazon SageMaker Model Monitor](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/model-monitor.html)
  - [Amazon SageMaker Model Monitor – Fully Managed Automatic Monitoring For Your Machine Learning Models](https://aws.amazon.com/ko/blogs/aws/amazon-sagemaker-model-monitor-fully-managed-automatic-monitoring-for-your-machine-learning-models/)
- [추론 파이프라인 배포](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/inference-pipelines.html)
- [SageMaker를 사용한 배치 추론](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html)
- [SageMaker를 사용한 실시간 추론](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipeline-real-time.html)
- [Amazon SageMaker Endpoint Auto Scaling](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/endpoint-auto-scaling.html)

## 검증
- [교차검증](https://m.blog.naver.com/ckdgus1433/221599517834)
- [혼동행렬](https://itwiki.kr/w/%ED%98%BC%EB%8F%99_%ED%96%89%EB%A0%AC)
- [ROC curve](https://angeloyeo.github.io/2020/08/05/ROC.html)
- [회귀 모델의 적합도](https://m.blog.naver.com/samsjang/221003939973)
- [결정계수(R Square)](https://medium.com/@Aaron__Kim/%EC%84%A0%ED%98%95-%ED%9A%8C%EA%B7%80%EC%9D%98-%EA%B2%B0%EC%A0%95%EA%B3%84%EC%88%98-linear-regression-r-squared-determination-coefficient-a66e4a32a9d6)
  - [조정 결정 계수(Adjusted R Square)는 어디에 사용하는 것인가?](https://chukycheese.github.io/statistics/adjusted-r2/)
- [크로스 엔트로피](https://www.youtube.com/watch?v=Jt5BS71uVfI)
- [사이킷런 (scikit learn) 에서의 교차검증 (cross validation), Kfold 정리](https://sgmath.tistory.com/61)

## 완전 자동화된 ML
- [Gluon AutoML](https://auto.gluon.ai/stable/index.html) - 강추!
- [SageMaker AutoPilot](https://aws.amazon.com/ko/sagemaker/autopilot/)
  - [SageMaker AutoPilot 예제](https://aws.amazon.com/ko/getting-started/hands-on/create-machine-learning-model-automatically-sagemaker-autopilot/)

## MLOps - 프로덕션 ML 워크로드를 위한 자료들
- [기계 학습 렌즈 AWS Well-Architected 프레임워크](https://docs.aws.amazon.com/ko_kr/wellarchitected/latest/machine-learning-lens/wellarchitected-machine-learning-lens.pdf) - ML을 위한 Well-Architected 프레임워크
- [AWS를 사용하여 전체 기계 학습 수명주기를 설계 및 구축하는 법 : End to End Amazon SageMaker 데모](https://aws.amazon.com/ko/blogs/machine-learning/architect-and-build-the-full-machine-learning-lifecycle-with-amazon-sagemaker/)
- [Amazon SageMaker MLOps 실습](https://github.com/aws-samples/mlops-amazon-sagemaker-devops-with-ml)
- [From DevOps to MLOPS: Integrate Machine Learning Models using Jenkins and Docker](https://towardsdatascience.com/from-devops-to-mlops-integrate-machine-learning-models-using-jenkins-and-docker-79034dbedf1)

## 참고자료

### 온라인 코스
- [Amazon Machine Learning University](https://www.youtube.com/channel/UC12LqyqTQYbXatYS9AA7Nuw) 
- [모두를 위한 딥러닝](https://hunkim.github.io/ml/)
- [생활코딩 머신러닝](https://opentutorials.org/module/4916)
- [개발자를 위한 실전 딥러닝](https://course.fast.ai/)
- [Deep Learning - The Straight Dope](https://gluon.mxnet.io/)


### 서적
- [비즈니스 머신러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B6474110466) - 아마존 세이지메이커와 주피터를 활용한 빠르고 효과적인 머신러닝 활용법
- [세상에서 가장 쉬운 통계학 입문](http://www.yes24.com/Product/Goods/3625262)
- [세상에서 가장 쉬운 베이즈통계학 입문](http://www.yes24.com/Product/Goods/36928073)
- [혼자 공부하는 머신러닝 + 딥러닝](https://books.google.co.kr/books?id=9Q0REAAAQBAJ&printsec=frontcover&dq=%ED%98%BC%EC%9E%90+%EA%B3%B5%EB%B6%80%ED%95%98%EB%8A%94+%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D&hl=ko&sa=X&ved=2ahUKEwjEkLSimcjuAhV0LH0KHTV4Dg8Q6AEwAHoECAQQAg#v=onepage&q=%ED%98%BC%EC%9E%90%20%EA%B3%B5%EB%B6%80%ED%95%98%EB%8A%94%20%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D&f=false)
- [밑바닥부터 시작하는 딥러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198) - 가장 쉽게 딥러닝의 작동원리와 구현을 설명한 책
  - [파이썬 노트북으로 정리한 밑바닥부터 시작하는 딥러닝 예제들](https://github.com/SDRLurker/deep-learning) 
- [핸즈온 머신러닝(2판)](https://www.hanbit.co.kr/store/books/look.php?p_code=B7033438574) - 난이도가 있지만 실무자의 시선에서 머신러닝 전반을 자세하게 다룬 책
- [틀리지 않는 법: 수학적 사고의 힘](https://books.google.co.kr/books/about/%ED%8B%80%EB%A6%AC%EC%A7%80_%EC%95%8A%EB%8A%94_%EB%B2%95.html?id=r6o9DAAAQBAJ&printsec=frontcover&source=kp_read_button&redir_esc=y#v=onepage&q&f=false) - 수학적 사고에 대해서 통찰한 교양서적

### 블로그 & 읽을거리
- [AWS Machine Learning Blog](https://aws.amazon.com/ko/blogs/machine-learning/)
- [KDNuggets](https://www.kdnuggets.com/)
- [Towards Data Science](https://towardsdatascience.com/)
  - [Don’t learn machine learning(머신러닝을 공부하지 마시오)](https://towardsdatascience.com/dont-learn-machine-learning-8af3cf946214) - 당신은 로그인 기능을 만들기 위해 커스텀 해시 함수를 제작하는가? 아니면 만들어진 해시 함수를 사용하는가?
- [설명 가능한 인공지능](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence)
