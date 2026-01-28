import re
import os
import sys
import json
import pandas as pd
from uuid import uuid4
from tqdm import tqdm
from typing import List

# LLM
from langchain_ollama import ChatOllama

# RAG
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Prompt / Parser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Schema (user-defined)
from schemas import PaperRecord, ModelRecord, MetricRecord, ModelsListOutput, MetricsListOutput


# Model Configuration
# A100 GPU optimized model setup
DEFAULT_MODELS = {
    "paper": "gemma3:27b",       # Medium-large model for comprehensive paper understanding (약 20GB)
    "models": "gemma3:27b",      # Medium model for structured model extraction (약 5GB)
    "metrics": "gemma3:27b"   # Small model optimized for numerical/table parsing (약 4GB)
}

DEFAULT_CONTEXT_SIZES = {
    "paper": 32000,    # 약간 줄여서 메모리 절약
    "models": 32000,
    "metrics": 32000
}


def initialize_llm(model_name: str, context_size: int, temperature: float = 0) -> ChatOllama:
    """Initialize a ChatOllama model with specified configuration"""
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        format="json",
        num_ctx=context_size
    )


def initialize_model_chain(chain_type: str, custom_config: dict = None) -> ChatOllama: 
    """
    Initialize LLM for a specific chain type.
    
    Args:
        chain_type: One of 'paper', 'models', 'metrics'
        custom_config: Optional dict with 'model' and 'context_size' keys
    
    Returns:
        Initialized ChatOllama instance
    """
    if custom_config:
        model_name = custom_config.get("model", DEFAULT_MODELS[chain_type])
        context_size = custom_config.get("context_size", DEFAULT_CONTEXT_SIZES[chain_type])
    else:
        model_name = DEFAULT_MODELS[chain_type]
        context_size = DEFAULT_CONTEXT_SIZES[chain_type]
    
    print(f"[Model Init] {chain_type.upper()} chain: {model_name} (context: {context_size})")
    return initialize_llm(model_name, context_size)


# Global model instances (initialized in main or via command line)
paper_llm = None
models_llm = None
metrics_llm = None

# Parsers for each extraction type
paper_parser = PydanticOutputParser(pydantic_object=PaperRecord)
models_parser = PydanticOutputParser(pydantic_object=ModelsListOutput)
metrics_parser = PydanticOutputParser(pydantic_object=MetricsListOutput)

# Prompt for paper information extraction (Chain 1: 전체 논문 요약 및 논문 유형 판단)
paper_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an information extraction system for scientific papers.

        Your task is to SUMMARIZE the entire paper and classify its type.

        CHAIN 1 ROLE: Paper Summary and Type Classification
        - Read and understand the ENTIRE paper content provided
        - Generate a comprehensive summary of the paper's main contribution
        - Classify the paper type based on its content
        - Assess efficiency, environmental impact, and James Burn AI paradox relevance

        STRICT NON-INFERENCE RULES:
        - Do NOT infer, assume, estimate, or complete missing information.
        - If a value, relationship, or attribution is not explicitly stated, return null.
        - Do NOT reinterpret baseline or comparison results as proposed results.

        PAPER SUMMARY:
        - Summarize the main contribution of the paper
        - Identify what the paper explicitly proposes (model, method, analysis, etc.)
        - Note key aspects: efficiency, computational cost, environmental impact
        - This summary will help subsequent chains understand the paper's focus

        PAPER TYPE CLASSIFICATION:
        - 'deep_learning_model': Papers that explicitly present deep learning model architectures and experimental results.
        - 'method_without_model_size': Papers that propose methods but do not report model parameter counts.
        - 'theoretical': Theory-focused papers without empirical models.
        - 'survey': Review or survey papers.
        - 'non_ml': Papers not directly related to machine learning.
        - 'unclear': When the paper type cannot be confidently determined.

        JAMES BURN PARADOX ASSESSMENT:
        - 'explicit': The paper explicitly states that efficiency improvements enable larger models or increased environmental cost.
        - 'implicit': The paper implicitly suggests that efficiency improvements allow model scaling.
        - 'not_relevant': No meaningful connection to the paradox.

        OUTPUT CONSTRAINTS:
        - Output a single JSON object strictly following the provided schema.
        - Use '{provided_paper_id}' as the paper_id.
        - Focus on paper-level metadata and classification, NOT detailed model or metric information.

        {format_instructions}
        """
    ),
    (
        "human",
        """
        Analyze and summarize the ENTIRE scientific paper to classify its type and extract paper-level metadata.

        Paper Name : {provided_paper_name}
        Paper ID : {provided_paper_id}
        
        Full paper content (markdown):
        ---
        {context}
        ---

        Your task:
        1. Summarize the paper's main contribution
        2. Classify the paper type
        3. Extract paper-level metadata (efficiency, environment, James Burn relevance, etc.)
        
        Return a JSON object that strictly follows the provided schema.
        Do NOT extract detailed model names or performance metrics - those will be handled by subsequent chains.
        """
    )
])

# Prompt for models extraction (Chain 2: 논문에서 제안하는 모델 및 모델 크기 탐색)
models_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an information extraction system for scientific papers.

        CHAIN 2 ROLE: Model Discovery and Size Extraction
        Your task is to SEARCH and EXTRACT information about models PROPOSED by the paper authors.
        Focus on: model names, model sizes (parameters), and model architectures.

        STRICT NON-INFERENCE RULES:
        - Do NOT infer, assume, estimate, or complete missing information.
        - If a value is not explicitly stated, return null.
        - Do NOT extract names of baseline, reference, or comparison models.

        MODEL DISCOVERY RULES:
        - SEARCH for models/methods that the paper PROPOSES, INTRODUCES, or DEVELOPS.
        - Extract model or method names ONLY if explicitly stated as proposed by the authors.
        - CRITICAL: Do NOT extract baseline, reference, or comparison models/methods.
        - Do NOT extract models/methods that are:
          * Described as "baseline", "baseline method", "baseline model"
          * Referenced as "previous work", "prior work", "existing methods"
          * Used for comparison purposes only (e.g., "compared with", "versus", "against")
          * Mentioned as "state-of-the-art" or "SOTA" methods that are NOT proposed by the authors
          * Listed in comparison tables but not proposed by the paper
        - Only extract models/methods explicitly described as:
          * "We propose", "We introduce", "We present", "Our method", "Our model"
          * "The proposed", "The introduced", "This paper presents"
          * Models/methods that are the main contribution of the paper
        - Use model or method name as model_name schema.
        - Each extracted model must have a unique model_id (1, 2, 3, ...).

        MODEL SIZE EXTRACTION (CRITICAL):
        - SEARCH for and extract model SIZE information:
          * Number of parameters (preferred)
          * Model size in MB (if parameters not available)
        - Parameter count units: M (million), B (billion)
        - Model size unit: MB (megabytes)
        - Extract the EXACT values as stated in the paper
        - Do NOT convert between size and parameter count
        - If size information is not explicitly stated, set parameter_count and parameter_unit to null

        TASK EXTRACTION:
        - Extract the primary task that each model performs (e.g., image classification, machine translation, etc.)

        OUTPUT CONSTRAINTS:
        - Output a JSON object with a "models" field containing an array of model objects.
        - Use '{provided_paper_id}' as the paper_id for all models.
        - If no models are proposed, return {{"models": []}}.
        - Focus on model names and sizes - performance metrics will be extracted by Chain 3.

        {format_instructions}
        """
    ),
    (
        "human",
        """
        SEARCH the following scientific paper for PROPOSED models and their sizes.

        Paper Name : {provided_paper_name}
        Paper ID : {provided_paper_id}
        
        Paper Type from Chain 1: {paper_type}
        
        IMPORTANT: 
        - Only extract models/methods that are PROPOSED by the authors of this paper.
        - Do NOT extract baseline models, comparison models, or existing methods from previous work.
        - Focus on finding: model names, parameter counts, model sizes
        
        Paper content (markdown):
        ---
        {context}
        ---

        Return a JSON object with a "models" field containing an array of model objects.
        Each model should include: model_name, parameter_count, parameter_unit, task.
        Remember: Only include models that are the main contribution of this paper.
        """
    )
])

# Prompt for metrics extraction (Chain 3: Chain 2가 발견한 모델 명들을 바탕으로 모델 성능 검색)
metrics_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an information extraction system for scientific papers.

        CHAIN 3 ROLE: Performance Metrics Search Based on Discovered Models
        Your task is to SEARCH for performance metrics SPECIFICALLY for the models discovered by Chain 2.
        Use the model names provided to find and extract their performance metrics.

        STRICT NON-INFERENCE RULES:
        - Do NOT infer, assume, estimate, or complete missing information.
        - If a value is not explicitly stated, return null.
        - Do NOT reinterpret baseline or comparison results as proposed results.

        MODEL-BASED METRIC SEARCH:
        - You have been provided with model names discovered by Chain 2: {models_context}
        - SEARCH for performance metrics that are SPECIFICALLY reported for these models
        - Match metrics to models by:
          * Exact model name match in tables or text
          * Abbreviated model name match
          * Explicit references (e.g., "Our model achieves...", "The proposed [model_name]...")
        - If a metric cannot be matched to a specific model from the provided list, set model_id to null
        - CRITICAL: Only extract metrics for the models provided in the models_context

        METRIC EXTRACTION RULES:
        - Extract ALL performance metrics that evaluate the proposed models, including:
          * Accuracy, Precision, Recall, F1-score, F1, F-score
          * mAP (mean Average Precision), AP (Average Precision)
          * BLEU, ROUGE, METEOR, CIDEr (for NLP tasks)
          * Top-1 accuracy, Top-5 accuracy
          * Perplexity, BPC (bits per character)
          * MSE, RMSE, MAE (for regression tasks)
          * Any other quantitative performance measures explicitly reported
        - CRITICAL: Extract metrics ONLY when they evaluate the proposed models from Chain 2.
        - CRITICAL: Do NOT extract metrics for baseline, reference, or comparison models/methods.
        - Ignore metrics reported solely for:
          * Baseline methods/models
          * Previous work or prior methods
          * Comparison methods that are NOT in the provided model list
          * State-of-the-art methods used only for comparison
        - If multiple metrics are reported for the same model-dataset combination, extract ALL of them as separate metric records.
        - Pay special attention to "Results", "Experiments", "Evaluation", and "Performance" sections.
        - When in doubt about whether a metric belongs to a proposed model or a baseline, do NOT extract it.

        TABLE INTERPRETATION RULES:
        - When metrics are reported in tables, carefully identify:
          * Row headers (usually model names or configurations)
          * Column headers (usually metric names, datasets, or tasks)
          * Cell values (the actual metric values)
        - CRITICAL: Before extracting any metric from a table, identify which rows correspond to the PROPOSED method/model.
        - Only extract metrics from rows that correspond to the proposed contribution.
        - Do NOT extract metrics from rows that are:
          * Clearly labeled as "baseline", "Baseline", "Baseline Method"
          * Previous work or existing methods
          * Comparison methods (e.g., "ResNet-50", "BERT-base" if they are not proposed)
          * Methods listed for comparison purposes only
        - Extract each cell value as a separate metric record with appropriate model_id, metric_name, dataset, and metric_value.
        - If a table has multiple datasets, create separate metric records for each dataset.
        - If a table has multiple metrics in columns, create separate metric records for each metric.
        - Do NOT merge values from different metric definitions into a single metric.
        - If table structure is ambiguous or you cannot clearly identify which rows are the proposed method, skip that table entirely.
        - Use table captions, section headers, and surrounding text to determine which methods are proposed vs. baselines.

        NUMERIC VALUE EXTRACTION:
        - Extract numeric values exactly as written (e.g., 95.2, 0.952, 95.2%)
        - Preserve the original format and unit.
        - If a value is given as a percentage (%), extract the numeric value and set metric_unit to "%".
        - If a value is given as a decimal (0.952), extract as is and set metric_unit appropriately.
        - Extract ranges (e.g., "95.2-96.1") as the first value, or if specified, extract both as separate records.

        MODEL ID MATCHING (CRITICAL):
        - The models discovered by Chain 2 are: {models_context}
        - Match each metric to the correct model using:
          * Exact model name match in tables or text
          * Abbreviated model name match
          * Explicit references in the text (e.g., "Our [model_name] achieves...", "The proposed [model_name]...")
        - Use the model_id from the provided model list (Model 1, Model 2, etc.)
        - If a metric cannot be matched to any model from the provided list, set model_id to null
        - Do NOT create metrics for models not in the provided list

        DATASET AND SPLIT IDENTIFICATION:
        - Extract dataset names explicitly mentioned (e.g., ImageNet, CIFAR-10, COCO, GLUE).
        - Identify the data split used: train, validation (val), test, or dev.
        - If split is not explicitly stated, infer from context (e.g., "test set", "validation accuracy", "on ImageNet").
        - If split cannot be determined, set split to null.

        OUTPUT CONSTRAINTS:
        - Output a JSON object with a "metrics" field containing an array of metric objects.
        - Use '{provided_paper_id}' as the paper_id for all metrics.
        - Extract as many metrics as possible - completeness is important.
        - If no metrics are reported, return {{"metrics": []}}.

        {format_instructions}
        """
    ),
        (
        "human",
        """
        SEARCH for performance metrics for the models discovered by Chain 2.

        Paper Name : {provided_paper_name}
        Paper ID : {provided_paper_id}
        
        Models discovered by Chain 2 (SEARCH for metrics for these models ONLY):
        {models_context}
        
        IMPORTANT: 
        - SEARCH the paper content for performance metrics SPECIFICALLY for the models listed above
        - Match metrics to models by model name
        - Only extract metrics for the PROPOSED models from Chain 2
        - Do NOT extract metrics for baseline models, comparison models, or existing methods
        - If a table contains both proposed and baseline methods, only extract rows corresponding to the proposed models
        
        Paper content (markdown):
        ---
        {context}
        ---

        Return a JSON object with a "metrics" field containing an array of metric objects.
        Each metric should be matched to a model from the provided list using model_id.
        Extract ALL performance metrics that evaluate the proposed models ONLY.
        """
    )
])



# RAG builder (Section-aware)
MAX_SECTION_CHARS = 4000
SUB_CHUNK_SIZE = 1200
SUB_CHUNK_OVERLAP = 300

def _trim_before_first_section(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("## "):
            return "\n".join(lines[i:])
    return text

def _infer_section_level(title: str) -> str:
    if re.match(r"\d+\.\d+", title):
        return "subsection"
    elif re.match(r"\d+\.", title):
        return "section"
    return "other"

def _is_reference_section(title: str) -> bool:
    """ filtering reference section """
    # 소문자 변환 및 공백 제거 후 키워드 매칭
    clean_title = title.lower().replace(" ", "")
    keywords = ["references", "bibliography", "참조문헌", "참고문헌"]
    return any(keyword in clean_title for keyword in keywords)


def build_retriever(md_path: str):
    loader = TextLoader(md_path, encoding="utf-8")
    raw_text = loader.load()[0].page_content

    
    cleaned = _trim_before_first_section(raw_text) # Trim preamble
    section_splitter = MarkdownHeaderTextSplitter( # Section-level splitting
        headers_to_split_on=[("##", "section")]
    )
    section_docs = section_splitter.split_text(cleaned) # section_docs: List[Document]
    
    if section_docs: # if last section is reference, delete it
        last_section_title = section_docs[-1].metadata.get("section", "")
        if _is_reference_section(last_section_title):
            section_docs.pop()

    sub_splitter = RecursiveCharacterTextSplitter( # Sub-chunk long sections
        chunk_size=SUB_CHUNK_SIZE,
        chunk_overlap=SUB_CHUNK_OVERLAP,
    )

    final_docs = []

    for doc in section_docs:
        section_title = doc.metadata.get("section", "").strip()
        section_level = _infer_section_level(section_title)

        base_metadata = {
            "source": md_path,
            "section": section_title,
            "section_level": section_level,
        }

        content = doc.page_content

        if len(content) <= MAX_SECTION_CHARS:
            final_docs.append(
                Document(page_content=content, metadata=base_metadata)
            )
        else:
            sub_chunks = sub_splitter.split_text(content)
            for i, sub in enumerate(sub_chunks):
                meta = dict(base_metadata)
                meta["subchunk_id"] = i
                final_docs.append(
                    Document(page_content=sub, metadata=meta)
                )

    embeddings = OllamaEmbeddings(model="nomic-embed-text") # Vector store
    vectorstore = FAISS.from_documents(final_docs, embeddings)

    # 기본 retriever (k=8)
    return vectorstore.as_retriever(search_kwargs={"k": 8})


def create_enhanced_retriever(vectorstore, search_type="similarity", k=8):
    """
    Create an enhanced retriever with configurable search parameters.
    
    Args:
        vectorstore: FAISS vector store
        search_type: "similarity" (default) or "mmr" (Maximum Marginal Relevance)
        k: Number of documents to retrieve
    """
    if search_type == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 2}  # fetch_k should be >= k
        )
    else:
        return vectorstore.as_retriever(search_kwargs={"k": k})


# Extraction (section-aware context)

def _build_section_context(docs):
    blocks = []
    for d in docs:
        header = f"[SECTION: {d.metadata.get('section','')}"

        if "subchunk_id" in d.metadata:
            header += f" | subchunk {d.metadata['subchunk_id']}"
        header += "]"

        blocks.append(f"{header}\n{d.page_content}")

    return "\n\n".join(blocks)

def extract_paper_info(retriever, paper_id, paper_name, llm_instance: ChatOllama = None):
    """
    Chain 1: 전체 논문 요약 및 논문 유형 판단
    
    Uses a large model to summarize the entire paper and classify its type.
    This provides context for subsequent chains.
    The summary and classification will be used to generate better search queries for Chain 2.
    """
    if llm_instance is None:
        llm_instance = paper_llm
    
    # Chain 1: 전체 논문을 요약하기 위해 더 넓은 범위의 문서 검색
    # 여러 검색 쿼리를 사용하여 논문 전체를 포괄적으로 검색
    search_queries = [
        "paper summary, main contribution, paper type, introduction, abstract",
        "proposed method, contribution, paper overview, main idea",
        "efficiency, scaling, computational cost, FLOPs, GPU time",
        "energy consumption, environmental impact, carbon emissions",
        "James Burn AI paradox, efficiency paradox, scaling paradox"
    ]
    
    # 여러 쿼리 결과를 통합
    all_docs = []
    seen_content = set()
    
    for query in search_queries:
        docs = retriever.invoke(query)
        for doc in docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_docs.append(doc)
    
    # 최대 12개 문서 사용 (논문 전체 이해를 위해)
    if len(all_docs) > 12:
        all_docs = all_docs[:12]

    context = _build_section_context(all_docs)

    chain = (
        paper_prompt.partial(
            format_instructions=paper_parser.get_format_instructions(),
            provided_paper_id=paper_id,
            provided_paper_name=paper_name
        )
        | llm_instance
        | paper_parser
    )

    try:
        return chain.invoke({"context": context})
    except Exception as e:
        print(f"Paper extraction failed for paper {paper_id}: {e}")
        return None


def extract_models_info(retriever, paper_id, paper_name, paper_record=None, llm_instance: ChatOllama = None):
    """
    Chain 2: 논문에서 제안하는 모델 및 모델 크기 탐색
    
    Uses a medium model to search for and extract proposed model names and sizes.
    Leverages paper_record from Chain 1 to generate optimized search queries.
    """
    if llm_instance is None:
        llm_instance = models_llm
    
    # Chain 2: Chain 1 결과를 기반으로 검색 쿼리 생성
    if paper_record:
        # Chain 1 결과를 활용한 풍부한 검색 쿼리 생성
        paper_info_parts = []
        
        if paper_record.paper_type:
            paper_info_parts.append(f"{paper_record.paper_type} model")
        
        if paper_record.has_explicit_models:
            paper_info_parts.append("explicit model architecture")
        
        if paper_record.efficiency_mentioned:
            paper_info_parts.append("efficient model architecture")
        
        # 논문 요약 정보 활용 (paradox_reason에 요약이 있을 수 있음)
        if paper_record.paradox_reason:
            # paradox_reason에서 키워드 추출하여 검색에 활용
            paper_info_parts.append("proposed contribution")
        
        # 종합 검색 쿼리 생성
        base_query = "proposed method or model architecture, model parameters or number of parameters, model name, model size, model capacity, proposed model, our model, introduced model"
        
        if paper_info_parts:
            enhanced_query = f"{base_query}, {', '.join(paper_info_parts)}"
        else:
            enhanced_query = base_query
        
        search_query = enhanced_query
    else:
        search_query = """
        proposed method or model architecture,
        model parameters or number of parameters,
        model name, model size, model capacity,
        proposed model, our model, introduced model
        """
    
    docs = retriever.invoke(search_query)
    context = _build_section_context(docs)

    chain = (
        models_prompt.partial(
            format_instructions=models_parser.get_format_instructions(),
            provided_paper_id=paper_id,
            provided_paper_name=paper_name,
            paper_type=paper_record.paper_type if paper_record else "unknown"
        )
        | llm_instance
        | models_parser
    )

    try:
        result = chain.invoke({"context": context})
        # Extract the list from the wrapper
        return result.models if hasattr(result, 'models') else []
    except Exception as e:
        print(f"Models extraction failed for paper {paper_id}: {e}")
        return []


def extract_metrics_info(retriever, paper_id, paper_name, models_list: List[ModelRecord] = None, 
                        paper_record=None, llm_instance: ChatOllama = None):
    """
    Chain 3: Chain 2가 발견한 모델 명들을 바탕으로 모델 성능 검색
    
    Uses a small, fast model to search for performance metrics specifically for the models
    discovered by Chain 2. Generates model-specific search queries.
    """
    if llm_instance is None:
        llm_instance = metrics_llm
    
    # Build model context string for prompt
    if models_list and len(models_list) > 0:
        models_context_lines = []
        model_details = []  # 검색 쿼리 생성을 위한 상세 정보
        
        for idx, model in enumerate(models_list, 1):
            model_info = f"Model {idx}: {model.model_name or 'Unnamed'}"
            if model.task:
                model_info += f" (Task: {model.task})"
            models_context_lines.append(model_info)
            
            # 모델 상세 정보 수집 (검색 쿼리 생성용)
            model_detail = {
                "name": model.model_name or "Unnamed",
                "task": model.task or "",
                "params": f"{model.parameter_count}{model.parameter_unit}" if model.parameter_count else ""
            }
            model_details.append(model_detail)
        
        models_context_str = "\n".join(models_context_lines)
        
        # Chain 3: Chain 2 결과를 기반으로 풍부한 검색 쿼리 생성
        search_queries = []
        
        # 각 모델에 대한 상세 검색 쿼리 생성
        for model_detail in model_details:
            model_name = model_detail["name"]
            task = model_detail["task"]
            params = model_detail["params"]
            
            # 모델 이름 기반 검색
            search_queries.append(f"{model_name} performance results accuracy metrics")
            search_queries.append(f"{model_name} experimental results evaluation")
            
            # 태스크 정보가 있으면 태스크 기반 검색 추가
            if task:
                search_queries.append(f"{model_name} {task} performance results")
                search_queries.append(f"{model_name} {task} accuracy metrics")
            
            # 파라미터 정보가 있으면 크기 기반 검색 추가
            if params:
                search_queries.append(f"{model_name} {params} parameters performance")
        
        # 논문 정보를 활용한 추가 검색 (Chain 1 결과 활용)
        if paper_record:
            if paper_record.paper_type:
                search_queries.append(f"{paper_record.paper_type} model performance metrics")
            
            # 모델 이름들을 조합한 검색
            model_names = [m.model_name for m in models_list if m.model_name]
            if model_names:
                combined_models = " ".join(model_names[:3])  # 최대 3개 모델 이름
                search_queries.append(f"{combined_models} performance comparison results")
        
        # 일반적인 메트릭 검색 쿼리 추가
        search_queries.extend([
            "experimental results performance metrics evaluation",
            "accuracy precision recall F1 score mAP",
            "results table performance comparison",
            "dataset evaluation test validation results"
        ])
    else:
        models_context_str = "No models extracted yet."
        # 기본 검색 쿼리
        search_queries = [
            "experimental results performance metrics evaluation",
            "accuracy precision recall F1 score mAP",
            "results table performance comparison",
            "dataset evaluation test validation results",
            "experiments section quantitative results"
        ]
    
    # Retrieve documents using multiple queries and combine
    # Chain 2 결과(모델 이름)를 기반으로 생성된 검색 쿼리들을 사용
    all_docs = []
    seen_content = set()
    
    # 모델별 검색 쿼리를 우선적으로 사용 (더 관련성 높음)
    model_specific_queries = []
    general_queries = []
    
    if models_list and len(models_list) > 0:
        model_names = [m.model_name for m in models_list if m.model_name]
        for query in search_queries:
            if any(model_name in query for model_name in model_names):
                model_specific_queries.append(query)
            else:
                general_queries.append(query)
    else:
        general_queries = search_queries
    
    # 모델별 쿼리 먼저 실행 (더 정확한 결과)
    for query in model_specific_queries:
        docs = retriever.invoke(query)
        for doc in docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_docs.append(doc)
                if len(all_docs) >= 12:
                    break
        if len(all_docs) >= 12:
            break
    
    # 일반 쿼리로 보완
    for query in general_queries:
        if len(all_docs) >= 12:
            break
        docs = retriever.invoke(query)
        for doc in docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_docs.append(doc)
                if len(all_docs) >= 12:
                    break
    
    # 최대 12개 문서 사용
    all_docs = all_docs[:12]
    
    context = _build_section_context(all_docs)

    chain = (
        metrics_prompt.partial(
            format_instructions=metrics_parser.get_format_instructions(),
            provided_paper_id=paper_id,
            provided_paper_name=paper_name,
            models_context=models_context_str
        )
        | llm_instance
        | metrics_parser
    )

    try:
        result = chain.invoke({"context": context})
        # Extract the list from the wrapper
        return result.metrics if hasattr(result, 'metrics') else []
    except Exception as e:
        print(f"Metrics extraction failed for paper {paper_id}: {e}")
        return [] 



BASE_PATH = "./papers/result"


def initialize_all_models(custom_config: dict = None):
    """
    Initialize all LLM models for the chain of chains architecture.
    
    Args:
        custom_config: Optional dict with model configurations:
            {
                "paper": {"model": "model_name", "context_size": 8192},
                "models": {"model": "model_name", "context_size": 4096},
                "metrics": {"model": "model_name", "context_size": 4096}
            }
    """
    global paper_llm, models_llm, metrics_llm
    
    print("[System] Initializing Chain of Chains Architecture")
    print("=" * 60)
    
    paper_config = custom_config.get("paper", {}) if custom_config else {}
    models_config = custom_config.get("models", {}) if custom_config else {}
    metrics_config = custom_config.get("metrics", {}) if custom_config else {}
    
    # Chain 1: Paper extraction (large model for comprehensive understanding)
    paper_llm = initialize_model_chain("paper", paper_config)
    
    # Chain 2: Models extraction (medium model for structured extraction)
    models_llm = initialize_model_chain("models", models_config)
    
    # Chain 3: Metrics extraction (small model for numerical/table parsing)
    metrics_llm = initialize_model_chain("metrics", metrics_config)
    
    print("=" * 60)
    print("[System] All models initialized successfully")
    print()


def main():
    # Parse command line arguments for custom model configuration
    custom_config = None
    if len(sys.argv) > 1:
        # Support both old format (single model) and new format (JSON config file)
        if sys.argv[1].endswith('.json'):
            import json
            with open(sys.argv[1], 'r') as f:
                custom_config = json.load(f)
        else:
            # Legacy support: single model for all chains
            ollama_model_name = sys.argv[1]
            context_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
            custom_config = {
                "paper": {"model": ollama_model_name, "context_size": context_size},
                "models": {"model": ollama_model_name, "context_size": context_size},
                "metrics": {"model": ollama_model_name, "context_size": context_size}
            }
    
    # Initialize all models
    initialize_all_models(custom_config)
    
    md_dir = os.path.join(BASE_PATH, "markdown")
    out_dir = os.path.join(BASE_PATH, "extracted")
    os.makedirs(out_dir, exist_ok=True)

    papers, models, metrics = [], [], []

    print("[System] Begin Automatic Model Text Mining with Chain of Chains")
    print()

    for fname in tqdm(os.listdir(md_dir), desc="Processing papers"):
        if not fname.endswith(".md"):
            continue

        paper_name = os.path.splitext(fname)[0] # 마지막 확장자 .md는 제외하고 파일명(논문 제목)만 선택
        paper_id = str(uuid4())

        path = os.path.join(md_dir, fname)
        retriever = build_retriever(path)
        
        # ============================================================
        # CHAIN 1: 전체 논문 요약 및 논문 유형 판단
        # Uses: paper_llm (large model for comprehensive understanding)
        # ============================================================
        print(f"[Chain 1] 전체 논문 요약 및 논문 유형 판단: {paper_name[:50]}...")
        paper_record: PaperRecord = extract_paper_info(
            retriever, paper_id, paper_name, llm_instance=paper_llm
        )
        
        if paper_record is None:
            print(f"  [SKIP] Paper extraction failed for {paper_id}")
            continue  # 파싱 실패 논문 스킵

        # Set paper metadata
        paper_record.title = paper_name
        paper_record.paper_id = paper_id
        papers.append(paper_record.model_dump())
        print(f"  [OK] Paper type: {paper_record.paper_type}")

        # ============================================================
        # CHAIN 2: 논문에서 제안하는 모델 및 모델 크기 탐색
        # Uses: models_llm (medium model for structured extraction)
        # Input: paper_record from Chain 1
        # ============================================================
        print(f"[Chain 2] 제안하는 모델 및 모델 크기 탐색...")
        models_list: List[ModelRecord] = extract_models_info(
            retriever, paper_id, paper_name, 
            paper_record=paper_record, 
            llm_instance=models_llm
        )
        
        # Model (paper-local index)
        local_model_counter = 1
        id_map = {}

        for m in models_list:
            original_id = m.model_id
            new_id = local_model_counter  # 논문 내부 1~n

            m.paper_id = paper_id
            m.model_id = new_id

            if original_id is not None:
                id_map[original_id] = new_id

            models.append(m.model_dump())
            local_model_counter += 1
        
        model_names = [m.model_name for m in models_list if m.model_name]
        print(f"  [OK] 발견된 모델: {len(models_list)}개 - {', '.join(model_names[:3])}{'...' if len(model_names) > 3 else ''}")

        # ============================================================
        # CHAIN 3: Chain 2가 발견한 모델 명들을 바탕으로 모델 성능 검색
        # Uses: metrics_llm (small model for numerical/table parsing)
        # Input: models_list from Chain 2 (모델 이름 기반 검색)
        # ============================================================
        print(f"[Chain 3] 발견된 모델들의 성능 검색...")
        metrics_list: List[MetricRecord] = extract_metrics_info(
            retriever, paper_id, paper_name, 
            models_list=models_list,
            paper_record=paper_record,
            llm_instance=metrics_llm
        )
        
        # Metric
        for met in metrics_list:
            met.paper_id = paper_id

            if met.model_id in id_map:
                met.model_id = id_map[met.model_id]

            metrics.append(met.model_dump())
        
        print(f"  [OK] Extracted {len(metrics_list)} metric(s)")
        print()

    # Save results
    pd.DataFrame(papers).to_csv(os.path.join(out_dir, "papers.csv"), index=False)
    pd.DataFrame(models).to_csv(os.path.join(out_dir, "models.csv"), index=False)
    pd.DataFrame(metrics).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    print("[DONE] Chain of Chains extraction completed.")
    print(f"  - Papers: {len(papers)}")
    print(f"  - Models: {len(models)}")
    print(f"  - Metrics: {len(metrics)}")


if __name__ == "__main__":
    main()
