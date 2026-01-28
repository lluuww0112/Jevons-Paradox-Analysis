# 논문 정보 추출 시스템 (Paper Information Extraction System)

## 프로젝트 개요

이 프로젝트는 과학 논문(Markdown 형식)에서 구조화된 정보를 자동으로 추출하는 시스템입니다. LLM(Large Language Model)과 RAG(Retrieval-Augmented Generation) 기술을 활용하여 논문에서 다음 세 가지 주요 정보를 추출합니다:

1. **논문 레벨 메타데이터** (papers_df)
2. **모델 정보** (models_df)
3. **성능 지표** (metrics_df)

## 주요 특징

### 1. Chain of Chains 아키텍처

**서로 다른 모델이 협력하는 3단계 체인 구조**를 채택했습니다:

각 단계마다 특화된 모델을 사용하여 최적의 성능을 달성합니다:

- **Chain 1 (Paper Extraction)**: 대형 모델 사용 - 논문 전체를 종합적으로 이해하고 메타데이터 추출
- **Chain 2 (Models Extraction)**: 중형 모델 사용 - 구조화된 모델 정보 추출에 특화
- **Chain 3 (Metrics Extraction)**: 소형 모델 사용 - 숫자/테이블 파싱에 최적화

각 체인은 이전 체인의 결과를 활용하여 더 정확한 추출을 수행합니다.

### 2. 3단계 분할 추출 전략

기존의 단일 추출 방식에서 개선하여, 각 데이터 타입을 별도로 추출하는 3단계 전략을 채택했습니다:

- **Step 1: 논문 정보 추출** - 논문 유형, 효율성, 환경 영향, James Burn AI 역설 관련성 등
- **Step 2: 모델 정보 추출** - 모델 이름, 파라미터 수, 태스크 등
- **Step 3: 성능 지표 추출** - 정확도, mAP, BLEU 등 다양한 성능 메트릭

이 분할 전략의 장점:
- LLM이 한 번에 처리해야 할 데이터 양 감소로 안정성 향상
- 각 단계별로 최적화된 프롬프트 사용 가능
- 한 단계 실패 시에도 다른 단계는 계속 진행 가능

### 2. 향상된 성능 지표 추출

성능 지표 추출 성능을 개선하기 위해 다음 기능을 추가했습니다:

- **모델 정보 컨텍스트 활용**: Step 2에서 추출한 모델 정보를 Step 3에 전달하여 모델-메트릭 매칭 정확도 향상
- **다중 검색 쿼리 전략**: 여러 검색 쿼리를 사용하여 관련 문서를 더 많이 검색
- **강화된 테이블 파싱**: 테이블 구조를 더 정확하게 파싱하는 지시사항 추가
- **다양한 메트릭 타입 인식**: Accuracy, mAP, BLEU, F1 등 다양한 메트릭 타입 자동 인식

## 프로젝트 구조

```
00_test/
├── extract.py          # 메인 추출 스크립트 (Chain of Chains)
├── schemas.py          # Pydantic 스키마 정의
├── preprocessing.py   # 전처리 유틸리티
├── ollama_server.py    # Ollama 서버 설정
├── config.example.json # 예제 모델 설정 파일
├── README.md          # 이 문서
└── papers/
    └── result/
        ├── markdown/   # 입력: Markdown 형식 논문
        └── extracted/  # 출력: 추출된 CSV 파일
            ├── papers.csv
            ├── models.csv
            └── metrics.csv
```

## 데이터 스키마

### papers_df (논문 정보)

| 필드 | 설명 | 타입 |
|------|------|------|
| paper_id | 논문 고유 ID (모든 테이블의 foreign key) | str |
| title | 논문 제목 | str (optional) |
| paper_type | 논문 유형 (deep_learning_model, method_without_model_size, theoretical, survey, non_ml, unclear) | str (optional) |
| has_explicit_models | 구체적인 모델이 명시적으로 등장하는지 여부 | bool (optional) |
| efficiency_mentioned | 효율성 관련 언급 여부 | bool (optional) |
| environment_mentioned | 환경 영향 언급 여부 | bool (optional) |
| compute_cost_reported | 계산 비용 보고 여부 | bool (optional) |
| james_burn_relevance | James Burn AI 역설 관련성 (explicit, implicit, not_relevant) | str (optional) |
| paradox_reason | 역설 관련성 판단 근거 | str (optional) |
| extraction_confidence | 추출 신뢰도 (high, medium, low) | str (optional) |

### models_df (모델 정보)

| 필드 | 설명 | 타입 |
|------|------|------|
| paper_id | 논문 ID (papers_df 참조) | str |
| model_id | 논문 내 모델 고유 ID | int |
| model_name | 모델 또는 아키텍처 이름 | str (optional) |
| task | 모델이 수행한 주요 태스크 | str (optional) |
| parameter_count | 파라미터 수 또는 모델 사이즈 | float (optional) |
| parameter_unit | 단위 (B: 십억, M: 백만, MB: 메가바이트) | str (optional) |
| notes | 추가 설명 | str (optional) |

### metrics_df (성능 지표)

| 필드 | 설명 | 타입 |
|------|------|------|
| paper_id | 논문 ID | str |
| model_id | 모델 ID (models_df 참조) | int (optional) |
| metric_name | 성능 지표 이름 (accuracy, mAP, BLEU 등) | str (optional) |
| metric_value | 성능 지표 수치 값 | float (optional) |
| metric_unit | 단위 (%, score 등) | str (optional) |
| dataset | 평가 데이터셋 이름 | str (optional) |
| split | 데이터 분할 (train, val, test) | str (optional) |

## 사용 방법

### 1. 환경 설정

필요한 패키지 설치:
```bash
pip install langchain-ollama langchain-community langchain-core pydantic pandas faiss-cpu tqdm
```

### 2. Ollama 모델 다운로드

A100 GPU에 최적화된 모델 구성 (권장):
```bash
# Chain 1: Paper extraction (대형 모델)
ollama pull llama3.1:70b

# Chain 2: Models extraction (중형 모델)
ollama pull qwen2.5:32b

# Chain 3: Metrics extraction (소형 모델)
ollama pull deepseek-r1:7b
```

### 3. 실행

**기본 설정 사용 (권장 모델 구성)**:
```bash
python extract.py
```

**커스텀 모델 구성 (JSON 파일)**:
```bash
# 예제 설정 파일 사용
cp config.example.json config.json
# 필요에 따라 config.json 수정

python extract.py config.json
```

또는 직접 생성:
```bash
cat > config.json << EOF
{
  "paper": {
    "model": "llama3.1:70b",
    "context_size": 8192
  },
  "models": {
    "model": "qwen2.5:32b",
    "context_size": 4096
  },
  "metrics": {
    "model": "deepseek-r1:7b",
    "context_size": 4096
  }
}
EOF

python extract.py config.json
```

**레거시 모드 (모든 체인에 동일 모델 사용)**:
```bash
python extract.py <model_name> <context_size>
# 예: python extract.py llama3 8192
```

### 3. 입력 파일 준비

`./papers/result/markdown/` 디렉토리에 Markdown 형식의 논문 파일을 배치합니다.

### 4. 실행

```bash
python extract.py
```

### 5. 결과 확인

추출된 결과는 `./papers/result/extracted/` 디렉토리에 CSV 파일로 저장됩니다:
- `papers.csv`: 논문 정보
- `models.csv`: 모델 정보
- `metrics.csv`: 성능 지표

## 주요 함수 설명

### `build_retriever(md_path: str)`

Markdown 파일을 로드하고 섹션 단위로 분할하여 FAISS 벡터 스토어를 구축합니다.

**특징:**
- 섹션 인식 기반 분할
- 긴 섹션은 서브 청크로 분할
- 참고문헌 섹션 자동 제거

**파라미터:**
- `MAX_SECTION_CHARS = 4000`: 섹션 최대 길이
- `SUB_CHUNK_SIZE = 1200`: 서브 청크 크기
- `SUB_CHUNK_OVERLAP = 300`: 서브 청크 오버랩

### `initialize_all_models(custom_config)`

모든 체인에 사용할 LLM 모델을 초기화합니다.

**파라미터:**
- `custom_config`: 선택적 모델 구성 딕셔너리

**기본 모델:**
- Chain 1: `llama3.1:70b` (논문 이해)
- Chain 2: `qwen2.5:32b` (모델 추출)
- Chain 3: `deepseek-r1:7b` (메트릭 추출)

### `extract_paper_info(retriever, paper_id, paper_name, llm_instance)`

**Chain 1**: 논문 레벨 메타데이터를 추출합니다.

**사용 모델**: 대형 모델 (기본: llama3.1:70b)

**검색 키워드:**
- 논문 유형, 기여도
- 효율성, 스케일링
- 계산 비용 (FLOPs, GPU 시간)
- 환경 영향
- James Burn AI 역설 관련성

### `extract_models_info(retriever, paper_id, paper_name, paper_record, llm_instance)`

**Chain 2**: 모델 정보를 추출합니다.

**사용 모델**: 중형 모델 (기본: qwen2.5:32b)

**입력:**
- `paper_record`: Chain 1에서 추출한 논문 정보 (선택적, 검색 쿼리 개선에 활용)

**검색 키워드:**
- 제안된 방법 또는 모델 아키텍처
- 모델 파라미터 수
- 모델 이름, 모델 사이즈

### `extract_metrics_info(retriever, paper_id, paper_name, models_list, paper_record, llm_instance)`

**Chain 3**: 성능 지표를 추출합니다.

**사용 모델**: 소형 모델 (기본: deepseek-r1:7b)

**입력:**
- `models_list`: Chain 2에서 추출한 모델 정보 (모델-메트릭 매칭에 필수)
- `paper_record`: Chain 1에서 추출한 논문 정보 (선택적)

**개선된 버전**으로 다음 기능을 포함합니다:

**주요 개선사항:**
1. **모델 정보 컨텍스트 활용**: 추출된 모델 정보를 프롬프트에 포함하여 모델-메트릭 매칭 정확도 향상
2. **다중 검색 쿼리**: 5개의 서로 다른 검색 쿼리를 사용하여 관련 문서를 더 많이 검색
   - "experimental results performance metrics evaluation"
   - "accuracy precision recall F1 score mAP"
   - "results table performance comparison"
   - "dataset evaluation test validation results"
   - "experiments section quantitative results"
3. **중복 제거**: 동일한 문서의 중복 검색 방지
4. **강화된 프롬프트**: 테이블 파싱, 다양한 메트릭 타입 인식, 숫자 값 추출 규칙 등 상세한 지시사항 포함

**검색 전략:**
- 첫 번째 쿼리 결과를 우선적으로 사용 (가장 관련성 높음)
- 다른 쿼리 결과에서 추가 문서 보완
- 최대 12개 문서 사용 (첫 쿼리 8개 + 다른 쿼리 4개)

## 성능 개선 전략

### 문제점
기존 시스템에서는 성능 수치를 잘 추출하지 못하는 문제가 있었습니다.

### 개선 방안

1. **분할 추출 전략**
   - 한 번에 모든 정보를 추출하는 대신, 논문 정보 → 모델 정보 → 성능 지표 순으로 단계별 추출
   - 각 단계별로 최적화된 프롬프트 사용

2. **모델 정보 활용**
   - Step 2에서 추출한 모델 정보를 Step 3에 전달
   - 모델 이름과 태스크 정보를 활용하여 메트릭과 모델 매칭 정확도 향상

3. **향상된 검색 전략**
   - 단일 검색 쿼리 대신 다중 검색 쿼리 사용
   - "results", "experiments", "evaluation" 등 다양한 키워드로 검색
   - 더 많은 관련 문서 검색 (기존 8개 → 최대 12개)

4. **강화된 프롬프트**
   - 테이블 구조 파싱에 대한 상세한 지시사항
   - 다양한 메트릭 타입 인식 (Accuracy, mAP, BLEU, F1, Top-1/5, Perplexity 등)
   - 숫자 값 추출 규칙 명확화 (%, 소수점, 범위 등)
   - 데이터셋 및 분할 정보 추출 강화

5. **테이블 파싱 개선**
   - 행/열 헤더 식별 규칙
   - 여러 데이터셋/메트릭을 별도 레코드로 추출
   - 테이블 구조 추론 규칙

## A100 GPU 모델 구성 추천

### 권장 모델 구성 (A100 80GB 기준)

A100 GPU의 80GB 메모리를 활용하여 최적의 성능을 위한 모델 구성을 추천합니다:

#### 구성 1: 고성능 구성 (권장) ⭐
```json
{
  "paper": {
    "model": "llama3.1:70b",
    "context_size": 8192
  },
  "models": {
    "model": "qwen2.5:32b",
    "context_size": 4096
  },
  "metrics": {
    "model": "deepseek-r1:7b",
    "context_size": 4096
  }
}
```

**특징:**
- **Chain 1**: llama3.1:70b (약 40GB) - 논문 전체 이해에 최적
- **Chain 2**: qwen2.5:32b (약 20GB) - 구조화된 정보 추출에 강함
- **Chain 3**: deepseek-r1:7b (약 5GB) - 숫자/테이블 파싱에 특화
- **총 메모리**: 약 65GB (여유 공간 확보)

#### 구성 2: 균형 구성
```json
{
  "paper": {
    "model": "qwen2.5:72b",
    "context_size": 8192
  },
  "models": {
    "model": "llama3.1:8b",
    "context_size": 4096
  },
  "metrics": {
    "model": "qwen2.5:7b",
    "context_size": 4096
  }
}
```

**특징:**
- 모든 체인에서 고품질 모델 사용
- 메모리 사용량: 약 70GB

#### 구성 3: 효율 구성 (빠른 처리)
```json
{
  "paper": {
    "model": "qwen2.5:32b",
    "context_size": 6144
  },
  "models": {
    "model": "llama3.1:8b",
    "context_size": 4096
  },
  "metrics": {
    "model": "deepseek-r1:7b",
    "context_size": 4096
  }
}
```

**특징:**
- 빠른 처리 속도
- 메모리 사용량: 약 40GB
- 대량 처리에 적합

### 모델별 역할 및 특성

| 체인 | 모델 크기 | 역할 | 추천 모델 | 메모리 |
|------|----------|------|----------|--------|
| **Chain 1** | 대형 (70B+) | 논문 전체 이해, 메타데이터 추출 | llama3.1:70b, qwen2.5:72b | 35-45GB |
| **Chain 2** | 중형 (8B-32B) | 구조화된 모델 정보 추출 | qwen2.5:32b, llama3.1:8b | 5-20GB |
| **Chain 3** | 소형 (7B-8B) | 숫자/테이블 파싱, 빠른 처리 | deepseek-r1:7b, qwen2.5:7b | 4-5GB |

### 메모리 관리 팁

1. **모델 동시 로딩**: A100 80GB는 3개 모델을 동시에 메모리에 유지 가능
2. **Quantization 고려**: 메모리 부족 시 Q4_K_M 양자화 모델 사용
   ```bash
   ollama pull llama3.1:70b-q4_K_M  # 약 20GB
   ```
3. **Context Size 조정**: 필요에 따라 context_size를 줄여 메모리 절약

### 성능 최적화

- **병렬 처리**: 각 체인은 순차 실행되지만, 모델은 미리 로드되어 대기
- **캐싱**: 동일 논문 재처리 시 RAG 결과 캐싱 활용 가능
- **배치 처리**: 여러 논문을 순차 처리하여 모델 로딩 오버헤드 최소화

## 기술 스택

- **LLM**: Ollama (로컬 LLM 실행, Chain of Chains 아키텍처)
- **RAG**: LangChain + FAISS (벡터 스토어)
- **임베딩**: nomic-embed-text
- **스키마 검증**: Pydantic
- **데이터 처리**: Pandas

## 주의사항

1. **비추론 원칙**: 명시적으로 서술되지 않은 정보는 추론하지 않습니다. 모든 값은 논문에 명시적으로 언급된 경우에만 추출됩니다.

2. **제안 기여도 식별**: 논문의 제안 기여도만 추출하며, 베이스라인이나 비교 모델의 정보는 제외합니다.

3. **모델-메트릭 매칭**: 모델 정보가 없는 경우 메트릭의 model_id는 null로 설정됩니다.

4. **에러 처리**: 각 추출 단계에서 에러가 발생하면 해당 단계만 실패하고 다른 단계는 계속 진행됩니다.

## 향후 개선 방향

1. **메트릭 추출 정확도 향상**
   - 테이블 구조 자동 인식 개선
   - 다양한 논문 형식 지원

2. **성능 최적화**
   - 병렬 처리 지원
   - 캐싱 메커니즘 추가

3. **검증 기능**
   - 추출 결과 자동 검증
   - 신뢰도 점수 개선

4. **시각화**
   - 추출 결과 시각화 도구
   - 통계 대시보드

## 라이선스

이 프로젝트는 데이터톤 프로젝트의 일부입니다.

## 작성일

2025년

