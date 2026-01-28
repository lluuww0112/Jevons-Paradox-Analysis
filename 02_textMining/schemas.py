"""
    papers_df
        paper_id : 논문을 고유하게 식별하기 위한 ID. 모든 테이블에서 foreign key로 사용됨
        title : 논문 제목 (명시되지 않은 경우 null)
        paper_type : 논문 유형
            - deep_learning_model : 딥러닝 모델 아키텍처 및 실험 결과를 명시적으로 제시한 논문
            - method_without_model_size : 방법론은 있으나 모델 파라미터 수를 보고하지 않은 논문
            - theoretical : 이론적 분석 중심 논문
            - survey : 리뷰 또는 서베이 논문
            - non_ml : 머신러닝과 직접적인 관련이 없는 논문
            - unclear : 유형 판별이 불명확한 경우
        has_explicit_models : 논문 내에 구체적인 모델(아키텍처 또는 인스턴스)이 명시적으로 등장하는지 여부
        efficiency_mentioned : 모델 효율성, 경량화, 스케일링 전략 등이 언급되었는지 여부
        environment_mentioned : 에너지 소비, 탄소 배출 등 환경 영향이 언급되었는지 여부
        compute_cost_reported : FLOPs, GPU 시간 등 정량적인 계산 비용이 보고되었는지 여부
        james_burn_relevance : James Burn AI 역설과의 관련성
            - explicit : 효율화로 인해 더 큰 모델 또는 환경 비용 증가를 명시적으로 서술
            - implicit : 효율화가 스케일 확장을 가능하게 함을 암묵적으로 시사
            - not_relevant : 해당 역설과 무관
        paradox_reason : james_nurm_relevence 판단의 근거가 되는 논문 내 서술 요약
        extraction_confidence : 추출 결과의 신뢰도 (high / medium / low)

    models_df
        paper_id : 해당 모델이 등장한 논문의 ID (papers_df.paper_id 참조)
        model_id : 논문 내에서 모델을 구분하기 위한 고유 ID
        model_name : 모델 또는 아키텍처 이름 (예: ResNet-50, ViT-B/16)
        task : 모델이 수행한 주요 태스크
        parameter_count : 논문에 명시적으로 보고된 모델 파라미터 수 (없을 경우 null), 또는 모델 사이즈
        parameter_unit : 파라미터 수의 단위
            - B : 십억 단위
            - M : 만 단위
        notes : 파라미터 일부만 보고된 경우나 추가적인 보충 설명

    metrics_df
        paper_id : 성능 지표가 보고된 논문의 ID
        model_id : 해당 성능을 낸 모델의 ID (models_df.model_id 참조)
        metric_name : 성능 지표 이름 (예: accuracy, mAP, BLEU, F1 등)
        metric_value : 성능 지표의 수치 값
        metric_unit : 성능 지표의 단위 (%, score 등)
        dataset : 성능 평가에 사용된 데이터셋 이름
        split : 평가에 사용된 데이터 분할 (train / val / test)
"""

# schemas.py
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PaperRecord(BaseModel):
    paper_id: Optional[str] = Field(
        description="Unique identifier for the paper. This ID is reused as a foreign key across all extracted tables." ,
        default=None
    )

    title: Optional[str] = Field(
        description="Title of the paper if explicitly stated; null if not available.",
        default=None
    )

    paper_type: Optional[Literal[
        "deep_learning_model",
        "method_without_model_size",
        "theoretical",
        "survey",
        "non_ml",
        "unclear"
    ]] = Field(
        description=(
            "Categorization of the paper type. "
            "'deep_learning_model' indicates papers that explicitly present deep learning model architectures "
            "and experimental results. "
            "'method_without_model_size' refers to papers that propose methods but do not report model parameter counts. "
            "'theoretical' denotes theory-focused papers without empirical models. "
            "'survey' refers to review or survey papers. "
            "'non_ml' indicates papers not directly related to machine learning. "
            "'unclear' is used when the paper type cannot be confidently determined."
        ),
        default=None
    )

    has_explicit_models: Optional[bool] = Field(
        description="Whether the paper explicitly describes concrete models or architectures.",
        default=None
    )

    efficiency_mentioned: Optional[bool] = Field(
        description=(
            "Whether the paper mentions efficiency-related techniques such as model compression, "
            "parameter efficiency, scaling strategies, or computational optimization."
        ),
        default=None
    )

    environment_mentioned: Optional[bool] = Field(
        description=(
            "Whether the paper mentions environmental impact, including energy consumption, "
            "carbon emissions, or sustainability-related considerations."
        ),
        default=None
    )

    compute_cost_reported: Optional[bool] = Field(
        description=(
            "Whether the paper explicitly reports quantitative compute cost metrics such as FLOPs, "
            "GPU hours, training time, or energy usage."
        ),
        default=None
    )

    james_burn_relevance: Optional[Literal[
        "explicit",
        "implicit",
        "not_relevant"
    ]] = Field(
        description=(
            "Assessment of the paper's relevance to the James Burn AI paradox. "
            "'explicit' indicates the paper explicitly states that efficiency improvements enable larger models "
            "or increased environmental cost. "
            "'implicit' indicates the paper implicitly suggests that efficiency improvements allow model scaling. "
            "'not_relevant' indicates no meaningful connection to the paradox."
        ),
        default=None
    )

    paradox_reason: Optional[str] = Field(
        description=(
            "A brief summary of the reasoning or textual evidence from the paper that justifies "
            "the assigned James Burn paradox relevance."
        ),
        default=None
    )

    extraction_confidence: Optional[Literal["high", "medium", "low"]] = Field(
        description="Confidence level of the extraction result based on clarity and completeness of the paper.",
        default=None
    )


class ModelRecord(BaseModel):
    paper_id: Optional[str] = Field(
        description="Identifier of the paper in which this model is described; references papers_df.paper_id.",
        default=None
    )

    model_id: Optional[int] = Field(
        description="Unique identifier for the model within the context of a single paper. System will fill this id",
        default=None
    )

    model_name: Optional[str] = Field(
        description="The name of the model or architecture proposed in the paper",
        default=None
    )

    task: Optional[str] = Field(
        description="Primary task performed by the model, such as image classification or machine translation.",
        default=None
    )

    parameter_count: Optional[float] = Field(
        description=(
            "Number of model parameters explicitly reported in the paper. "
            "Also, it can be size of model"
            "Null if the parameter count is not stated."
        ),
        default=None
    )

    parameter_unit: Optional[
        Literal["B", "M", "MB"]
    ] = Field(
        description=(
            "Unit used for the reported parameter count. "
            "'B' (billions) and 'M' (millions) for Parameter Count"
            "MB for model size"
        ),
        default=None
    )

    notes: Optional[str] = Field(
        description=(
            "Additional notes or caveats about the model, such as partial parameter reporting, "
            "shared backbones, or missing architectural details."
        ),
        default=None
    )


class MetricRecord(BaseModel):
    paper_id: Optional[str] = Field(
        description="Identifier of the paper in which this metric is reported.",
        default=None
    )

    model_id: Optional[int] = Field(
        description="Identifier of the model that produced this metric; references models_df.model_id. System will fill this filed",
        default=None
    )

    metric_name: Optional[str] = Field(
        description="Name of the performance metric, such as accuracy, mAP, BLEU, or F1 that stated in paper",
        default=None
    )

    metric_value: Optional[float] = Field(
        description=(
            "Numerical value of the performance metric if explicitly reported."
            "If the paper discusses performance improvements, extract the quantitative metrics that show the improvement."
        ),
        default=None
    )

    metric_unit: Optional[str] = Field(
        description="Unit of the metric value, such as percentage (%) or raw score.",
        default=None
    )

    dataset: Optional[str] = Field(
        description="Name of the dataset used for evaluation, if explicitly stated.",
        default=None
    )

    split: Optional[str] = Field(
        description="Dataset split used for evaluation, such as train, validation, or test.",
        default=None
    )


class ModelsListOutput(BaseModel):
    """Wrapper for list of ModelRecord to work with PydanticOutputParser"""
    models: List[ModelRecord] = Field(
        description="List of models explicitly described in the paper; empty if no models are present.",
        default_factory=list
    )


class MetricsListOutput(BaseModel):
    """Wrapper for list of MetricRecord to work with PydanticOutputParser"""
    metrics: List[MetricRecord] = Field(
        description="List of quantitative performance metrics reported in the paper; empty if none are present.",
        default_factory=list
    )


class ExtractionOutput(BaseModel):
    paper: PaperRecord = Field(
        description="Paper-level metadata and analysis extracted from the document."
    )

    models: List[ModelRecord] = Field(
        description="List of models explicitly described in the paper; empty if no models are present.",
        default_factory=list
    )

    metrics: List[MetricRecord] = Field(
        description="List of quantitative performance metrics reported in the paper; empty if none are present.",
        default_factory=list
    )


if __name__ == "__main__":
    try: # pydantic로 데이터 프레임을 생성하는 코드 예시
        import pandas as pd
        from pydantic import TypeAdapter

        adapter = TypeAdapter(ExtractionOutput)

        llm_output = None
        result = adapter.validate_python(llm_output)

        papers_df = pd.DataFrame([result.paper.model_dump()])
        models_df = pd.DataFrame([m.model_dump() for m in result.models])
        metrics_df = pd.DataFrame([m.model_dump() for m in result.metrics])
    except:
        pass
