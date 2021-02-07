# Self-Study on SageMaker(https://bit.ly/sagemaker-self-study)

> 혼자서 SageMaker를 배울 수 있도록 공개된 자료들을 모아놓았습니다.

## SageMaker 선수 지식 및 101

* AWS Cloud 일반 지식 (S3, EC2, IAM 등) (초급/중급 정도 수준)
* Python 코딩 (Pandas, Numpy 패키지 초급/중급 정도 수준)
* 쥬피터 노트북 1회 실행
    * https://www.ciokorea.com/tags/11396/R/118326
    * https://scikitlearn.tistory.com/73
* 머신 러닝 기본 (기타 자료를 보셔도 됩니다.)
    * 생활 코딩: 머신러닝 
        * https://opentutorials.org/course/4548
* 다커 컨테이너 동작 방식 - **이것은  옵션 입니다. 지나 치셔도 됩니다.**
    * Hello Docker
        * https://github.com/mullue/hello-docker


## SageMaker 입문 

**목표: 세이지 메이커의 기본 구조와 사용법 알기**

* SageMaker 소개 유튜브 비디오 (한글)
    *  SageMaker Overview (소개 동영상) - https://youtu.be/jF2BN98KBlg
    * SageMaker demo - https://youtu.be/miIVGlq6OUk (1시간 데모에 많은 내용을 압축해서 다루고 있습니다. 반복해서 보시거나 돌려보기로 차근차근 보셔도 괜찮습니다.)
* Introduction to Amazon SageMaker (12분)
    * 코세라 공식 세이지 메이커 소개 비디오
      * https://www.coursera.org/lecture/aws-machine-learning/introduction-to-amazon-sagemaker-QugTh
* Amazon SageMaker Deep Dive Series** (아래 두개만 일단 보셔도 됩니다. 각 비디오가 약 10-20분 사이 입니다.)**
    * https://www.youtube.com/playlist?list=PLhr1KZpdzukcOr_6j_zmSrvYnLUtgqsZz
        * Fully-Managed Notebook Instanaces with Amazon SageMaker - a Deep Dive
        * Built-in Machine Learning Algorithms with Amazon SageMaker - a Deep Dive
* SageMaker 최초 실습
    *  한글 워크샵 사이트(https://www.sagemaker-workshop-kr.com/kr)에서 다음 두 모듈을 진행합니다.(빌트인 알고리즘 활용)
        * `o` 모듈1 SageMaker > S3 bucket과 노트북 생성하기 - https://www.sagemaker-workshop-kr.com/kr/sagemaker/_module_1.html
        * `o` 모듈2 Linear Learner MNIST - https://www.sagemaker-workshop-kr.com/kr/sagemaker/_module_2.html
    * 기본 Tabular 데이터를 사용
        * 다이렉트 마케팅 (SageMaker 내장 알고리즘 XGBoost 사용)
            * https://github.com/mullue/xgboost (1번 노트북 진행)



## SageMaker 기본

**목표: 세이지 메이커의 유스 케이스별 사용 알기**

* Tensorflow 활용실습 (Tensorflow 2.0 script mode와 stepfunctions사용하기) 
    *  https://github.com/mullue/sm-tf2
* Use Case 별 예제
    * 공식 예제 사이트
        *  SageMaker example code(https://github.com/awslabs/amazon-sagemaker-examples
    * 공식 ML 블로그: 적용 사례, 다른 aws서비스와의 통합관련 예제나 기술 팁등 다양한 주제들이 다루어집니다.
        * aws ML blog(https://aws.amazon.com/ko/blogs/machine-learning/
    * 한글 예제 사이트
        * 자동차 번호판 인식 문제 해결에 SageMaker 적용해보기 - https://github.com/mullue/lab-custom-model-anpr
        * 빌트인 알고리즘을 이용한 한글처리 - https://github.com/daekeun-ml/blazingtext-workshop-korean
        * BERT 이용한 한글처리 - https://github.com/daekeun-ml/kobert-workshop
        * SageMaker와 EFS 연결하기 - https://aws.amazon.com/ko/blogs/machine-learning/speed-up-training-on-amazon-sagemaker-using-amazon-efs-or-amazon-fsx-for-lustre-file-systems/
        * Kubernetes Kubeflow와의 통합 - https://aws.amazon.com/ko/blogs/machine-learning/introducing-amazon-sagemaker-components-for-kubeflow-pipelines/
        * Elasticsearch와 연계한 이미지 검색 시스템 - https://aws.amazon.com/ko/blogs/machine-learning/building-a-visual-search-application-with-amazon-sagemaker-and-amazon-es/
        * Ground Truth를 이용한 3D 레이블링 - https://aws.amazon.com/ko/blogs/machine-learning/labeling-data-for-3d-object-tracking-and-sensor-fusion-in-amazon-sagemaker-ground-truth/
        * DLAMI를 이용한 분산학습 - https://aws.amazon.com/ko/blogs/machine-learning/multi-gpu-distributed-deep-learning-training-at-scale-on-aws-with-ubuntu18-dlami-efa-on-p3dn-instances-and-amazon-fsx-for-lustre/
        * 유스케이스별로 deploy하여 확인할 수 있는 케이스는 aws solutions ML 섹션을 참고하십시오.
            * https://aws.amazon.com/solutions/implementations/?solutions-all.sort-by=item.additionalFields.sortDate&solutions-all.sort-order=desc&awsf.AWS-Product%20Category=tech-category%23ai-ml
    * 워크샵
        *  동일 사이트의 reference 페이지 (https://www.sagemaker-workshop-kr.com/kr/references.html)
* 기타 유용한 자료 (백서 등)
    * ML 프로젝트 전반에 대한 이해가 필요하시면 다음 AWS ML백서들이 도움이 되실 수 있습니다.
        * https://d1.awsstatic.com/whitepapers/Deep_Learning_on_AWS.pdf?did=wp_card&trk=wp_card
        * https://d1.awsstatic.com/whitepapers/aws-managing-ml-projects.pdf?did=wp_card&trk=wp_card
    * 체계적인 교육
        * Udacity nanodegree 코스 - https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t
        * Coursera Sagemaker - https://www.coursera.org/lecture/aws-machine-learning/introduction-to-amazon-sagemaker-QugTh
        * Coursera computer vision - https://www.coursera.org/learn/aws-computer-vision-gluoncv
    * 외부 사이트
        * [medium.com](http://medium.com/) 에서 검색해서 사용
        * 비젼 딥 러닝
            * 라온 피블 블로그
              * https://m.blog.naver.com/PostList.nhn?blogId=laonple
        * NLP 딥러닝
            * 딥 러닝을 이용한 자연어 처리 입문
              * https://wikidocs.net/book/2155



## SageMaker를 이용한 데이터 과학 학습 자료(준비중)

- 데이터 분석 - **_선형회귀분석 이론_** 및 Jupyter Notebook 코드리뷰 이론 + 실습

* Regression with Amazon SageMaker Linear Learner algorithm
    * https://github.com/aws/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/linear_learner_abalone/Linear_Learner_Regression_csv_format.ipynb

- 데이터 분석 – **_트리 이론_** 및 Jupyter Notebook 코드리뷰 이론 + 실습

  - ***Gradient Boosted Trees를 이용하는 지도학습: 편향된 클래스의 이진 분류 예측문제 해결***
    - https://github.com/mullue/xgboost/blob/master/1.xgboost_direct_marketing_sagemaker.ipynb

- 데이터 분석 – **_인공신경망 이론_** 및 Jupyter Notebook 코드리뷰 이론 + 실습
  - https://github.com/SDRLurker/deep-learning - 노트북으로 정리한 밑바닥부터 시작하는 딥러닝 예제들
- 데이터 분석 – **_SVM 이론_** 및 Jupyter Notebook 코드리뷰 이론 + 실습
  - https://github.com/Jean-njoroge/Breast-cancer-risk-prediction - 유방암 위험 분석
- 데이터 분석 – **_Naive Bayses 이론_** 및 Jupyter Notebook 코드리뷰 이론 + 실습
  - https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html

- 데이터 분석 – **_k-means 이론_** 및 Jupyter Notebook 코드리뷰 이론 + 실습
  - Analyze US census data for population segmentation using Amazon SageMaker
    * https://github.com/aws/amazon-sagemaker-examples/blob/master/introduction_to_applying_machine_learning/US-census_population_segmentation_PCA_Kmeans/sagemaker-countycensusclustering.ipynb

- 데이터 분석 – **_텍스트 데이터 분석 이론_** 및 Jupyter Notebook 코드리뷰 이론 + 실습
  - 토픽 모델링을 사용한 온라인 상품 부정 리뷰 분석
    * https://github.com/gonsoomoon-ml/topic-modeling

- **_rcf 와 deepar_**를 이용한 시계열 데이터 이상탐지 
  - https://github.com/aws-samples/amazon-sagemaker-anomaly-detection-with-rcf-and-deepar

- **_배포 Amazon SageMaker 를 활용한 모델 배포 실습_**
  - TensorFlow 2 프로젝트 워크플로우를 SageMaker에서 실행하기
    * https://github.com/mullue/sm-tf2/blob/master/tf-2-workflow.ipynb

