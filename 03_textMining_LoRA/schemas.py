"""
    papers_df
        paper_id : 논문을 고유하게 식별하기 위한 ID. 모든 테이블에서 foreign key로 사용됨
        title : 논문 제목 (명시되지 않은 경우 null)
        paper_type : 논문 유형 (효율적인 모델 학습 여부로 구분)
            - efficient_learning : 효율적인 모델 학습 방법을 제안하거나 다루는 논문
                * parameter-efficient fine-tuning (LoRA, Adapters, Prefix Tuning 등)
                * model compression (pruning, quantization, distillation)
                * efficient training methods (gradient checkpointing, mixed precision 등)
                * 경량화, 효율성 개선에 초점을 맞춘 논문
            - standard_learning : 일반적인 모델 학습 방법을 다루는 논문
                * full fine-tuning
                * standard training methods
                * 일반적인 모델 아키텍처 제안
            - unclear : 효율적인 모델 학습 여부를 판별하기 어려운 경우
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
        model_name : 효율적 파인튜닝 방법론을 적용한 모델의 이름 또는 방법론 이름 (예: LoRA-GPT-3, AdaLoRA-BERT)
        backbone_model_name : 백본 모델의 이름 (예: GPT-3, BERT-base, T5-large, ViT-B/16)
        task : 모델이 수행한 주요 vision 태스크
            - Image Classification : 이미지 분류
            - Object Detection : 객체 탐지
            - Semantic Segmentation : 의미 분할
            - Instance Segmentation : 인스턴스 분할
            - Image Generation : 이미지 생성
            - Image Captioning : 이미지 캡셔닝
            - Visual Question Answering : 시각적 질의응답
            - Other : 위에 해당하지 않는 기타 태스크
        backbone_parameter_count : 백본 모델의 전체 파라미터 수 (없을 경우 null)
            - CRITICAL: 효율적 파인튜닝 방법의 trainable 파라미터가 아닌 백본 모델 전체의 파라미터 수
            - 예: GPT-3의 경우 175B, BERT-base의 경우 110M
        backbone_parameter_unit : 백본 파라미터 수의 단위
            - B : 십억 단위 (billions)
            - M : 백만 단위 (millions)
        learnable_parameter_count : 효율적 파인튜닝 방법에 의한 학습 가능한 파라미터 수 (없을 경우 null)
            - LoRA, Adapter, Prefix Tuning 등에 의해 추가된 trainable 파라미터
            - 예: LoRA rank 파라미터, adapter 파라미터 등
        learnable_parameter_unit : 학습 가능한 파라미터 수의 단위
            - B : 십억 단위
            - M : 백만 단위
        learnable_parameter_ratio : 백본 파라미터 수 대비 학습 가능한 파라미터 비율 (백본 파라미터가 없고 비율만 보고된 경우)
            - 예: 0.01% (0.0001), 0.1% (0.001)
            - 백본 파라미터 수가 있는 경우, 이 값으로 백본 파라미터 수를 역산 가능
        notes : 추가적인 보충 설명 (예: LoRA rank, adapter 크기, 특이사항)

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
        "efficient_learning",
        "standard_learning",
        "unclear"
    ]] = Field(
        description=(
            "Categorization of the paper type based on whether it focuses on efficient model learning. "
            "'efficient_learning' indicates papers that propose or discuss efficient model learning methods, including: "
            "- Parameter-efficient fine-tuning (LoRA, Adapters, Prefix Tuning, P-tuning, etc.) "
            "- Model compression techniques (pruning, quantization, knowledge distillation) "
            "- Efficient training methods (gradient checkpointing, mixed precision training, etc.) "
            "- Papers that focus on reducing computational cost, memory usage, or parameter count while maintaining performance. "
            "'standard_learning' indicates papers that deal with standard or full model learning approaches, including: "
            "- Full fine-tuning of models "
            "- Standard training methods without efficiency focus "
            "- General model architecture proposals without efficiency improvements "
            "'unclear' is used when it cannot be confidently determined whether the paper focuses on efficient learning or standard learning."
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
        description="Name of the parameter-efficient fine-tuning enhanced model or the method name (e.g., LoRA-GPT-3, AdaLoRA-BERT, Prefix-T5)",
        default=None
    )

    backbone_model_name: Optional[str] = Field(
        description="Name of the base/backbone model that parameter-efficient fine-tuning methods are applied to (e.g., GPT-3, BERT-base, T5-large, ViT-B/16)",
        default=None
    )

    task: Optional[Literal[
        "Image Classification",
        "Object Detection",
        "Semantic Segmentation",
        "Instance Segmentation",
        "Image Generation",
        "Image Captioning",
        "Visual Question Answering",
        "Other"
    ]] = Field(
        description=(
            "Primary vision task performed by the model. "
            "Choose from: 'Image Classification', 'Object Detection', 'Semantic Segmentation', "
            "'Instance Segmentation', 'Image Generation', 'Image Captioning', 'Visual Question Answering', or 'Other' "
            "if the task does not fit into the above categories."
        ),
        default=None
    )

    backbone_parameter_count: Optional[float] = Field(
        description=(
            "CRITICAL: Total number of parameters in the backbone/base model, NOT the trainable parameters from efficient fine-tuning methods. "
            "This is the full model size before parameter-efficient fine-tuning adaptation (e.g., 175B for GPT-3, 110M for BERT-base). "
            "Extract this value even if the paper emphasizes parameter efficiency. "
            "Null if not explicitly stated."
        ),
        default=None
    )

    backbone_parameter_unit: Optional[Literal["B", "M"]] = Field(
        description=(
            "Unit for backbone parameter count. "
            "'B' for billions, 'M' for millions."
        ),
        default=None
    )

    learnable_parameter_count: Optional[float] = Field(
        description=(
            "Number of trainable parameters introduced by parameter-efficient fine-tuning methods (e.g., LoRA, Adapters, Prefix Tuning). "
            "This includes method-specific trainable parameters such as LoRA rank parameters, adapter parameters, prefix parameters, etc. "
            "Null if not explicitly stated."
        ),
        default=None
    )

    learnable_parameter_unit: Optional[Literal["B", "M"]] = Field(
        description=(
            "Unit for learnable parameter count. "
            "'B' for billions, 'M' for millions."
        ),
        default=None
    )

    learnable_parameter_ratio: Optional[float] = Field(
        description=(
            "Ratio of learnable parameters relative to backbone parameters (e.g., 0.0001 for 0.01%, 0.001 for 0.1%). "
            "Use this when backbone parameter count is NOT reported but only the ratio is given. "
            "If both backbone_parameter_count and this ratio are available, extract both. "
            "Null if not explicitly stated."
        ),
        default=None
    )

    notes: Optional[str] = Field(
        description=(
            "Additional notes such as method-specific details (e.g., LoRA rank, adapter size, prefix length) or any special considerations about parameter extraction."
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
