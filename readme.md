# Project Overview
핵심 가설
- 제번스의 역설(Jevons Paradox)을 AI 분야에 대입하여, 단위 파라미터당 효율성이 높아질수록 더 어려운 벤치마크와 고성능에 대한 수요가 발생하고, 결과적으로 전체 모델 규모(Parameter count)가 기하급수적으로 증가함을 분석합니다.

분석 대상
- 2018년 ~ 2025년 사이의 멀티모달 및 LLM 관련 학술 논문 메타데이터 및 추이.

주요 목표
- 시대별 모델 파라미터 수와 성능(Benchmarking)의 상관관계 시각화.
- 효율성 증대 기술(LoRA, 효율적인 Backbone 등)의 등장이 모델 크기 증가에 미친 영향 분석.
- 키워드 네트워크 분석을 통한 연구 트렌드의 수렴 과정(Fragmentation → Convergence) 확인.


# Tools (python : 3.11.14)
## TextMining
- langchain
- ollama
- pydantic
- faiss
- docling

## Data Analysis
- pandas
- numpy
- scipy
- scikit-learn
- BERTopic

## Visualization
- matplotlib
- plotly
- seaborn

# Report
https://www.notion.so/2f51a949c14c8085a3b8c1dd685bbf31#2f61a949c14c80728a10ce59de0aa90b
