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
from schemas import PaperRecord, ModelRecord, ModelsListOutput


# Model Configuration
# A100 GPU optimized model setup
DEFAULT_MODELS = {
    "paper": "gemma3:27b",       # Medium-large model for comprehensive paper understanding (약 20GB)
    "models": "gemma3:27b"       # Medium model for structured model extraction (약 5GB)
}

DEFAULT_CONTEXT_SIZES = {
    "paper": 32000,    # 약간 줄여서 메모리 절약
    "models": 32000
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
        chain_type: One of 'paper', 'models'
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

# Parsers for each extraction type
paper_parser = PydanticOutputParser(pydantic_object=PaperRecord)
models_parser = PydanticOutputParser(pydantic_object=ModelsListOutput)

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

        PAPER TYPE CLASSIFICATION (EFFICIENT LEARNING vs STANDARD LEARNING):
        Classify the paper based on whether it focuses on EFFICIENT MODEL LEARNING or STANDARD LEARNING.
        
        'efficient_learning': Papers that propose or discuss efficient model learning methods.
        - Parameter-efficient fine-tuning methods (LoRA, Adapters, Prefix Tuning, P-tuning, IA³, etc.)
        - Model compression techniques (pruning, quantization, knowledge distillation, weight sharing)
        - Efficient training methods (gradient checkpointing, mixed precision training, activation recomputation)
        - Papers emphasizing: reducing computational cost, memory usage, parameter count, training time, inference speed
        - Papers that propose methods to achieve similar or better performance with fewer resources
        - Key indicators: "parameter-efficient", "efficient fine-tuning", "low-rank adaptation", "adapter", "compression", "pruning", "quantization", "distillation"
        
        'standard_learning': Papers that deal with standard or full model learning approaches.
        - Full fine-tuning of models (training all parameters)
        - Standard training methods without efficiency focus
        - General model architecture proposals without emphasizing efficiency improvements
        - Papers that do not focus on reducing computational cost or parameters
        - Key indicators: "full fine-tuning", "standard training", general model proposals without efficiency claims
        
        'unclear': When it cannot be confidently determined whether the paper focuses on efficient learning or standard learning.
        - Survey papers that cover both efficient and standard methods
        - Theoretical papers without clear empirical focus
        - Papers where the main contribution is ambiguous

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
        You are an information extraction system for parameter-efficient fine-tuning and efficient fine-tuning papers.

        CHAIN 2 ROLE: Backbone Model and Parameter Extraction
        Your task is to SEARCH and EXTRACT information about backbone models and trainable parameters.
        This is for parameter-efficient fine-tuning papers (including but not limited to LoRA, Adapters, Prefix Tuning, etc.).

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
        - Use parameter-efficient fine-tuning method name or enhanced model name as model_name schema.
        - Each extracted model must have a unique model_id (1, 2, 3, ...).

        BACKBONE MODEL EXTRACTION (CRITICAL - HIGHEST PRIORITY):
        - SEARCH for and extract the BACKBONE/BASE model name and its TOTAL parameter count.
        - The backbone model is the pre-trained base model that efficient fine-tuning methods are applied to (e.g., GPT-3, BERT-base, T5-large).
        - CRITICAL: Extract the FULL/TOTAL parameter count of the backbone model, NOT the trainable parameters introduced by efficient fine-tuning methods.
        - Examples of backbone parameters:
          * GPT-3: 175B parameters
          * BERT-base: 110M parameters
          * T5-large: 770M parameters
          * ViT-B/16: 86M parameters
        - Look for phrases like:
          * "base model with X parameters"
          * "pre-trained model (X parameters)"
          * "backbone model (X parameters)"
          * "full model size: X"
        - Extract backbone_model_name (e.g., "GPT-3", "BERT-base", "T5-large")
        - Extract backbone_parameter_count and backbone_parameter_unit (B for billions, M for millions)

        LEARNABLE PARAMETER EXTRACTION:
        - SEARCH for and extract trainable parameter information introduced by efficient fine-tuning methods.
        - This is the number of parameters added by efficient fine-tuning methods (e.g., LoRA, Adapters, Prefix Tuning), NOT the full model.
        - Look for phrases like:
          * "trainable parameters: X"
          * "trainable parameter count: X"
          * "parameters introduced by [method]: X"
          * "adapter parameters: X"
          * "only X parameters are trainable"
          * "only X% of parameters are trainable"
        - Extract learnable_parameter_count and learnable_parameter_unit (B for billions, M for millions only)
        - CRITICAL: Convert any parameters reported in thousands (K) to millions (M) by dividing by 1000. For example, 100K becomes 0.1M.

        LEARNABLE PARAMETER RATIO EXTRACTION (FALLBACK):
        - If backbone_parameter_count is NOT explicitly stated but only the ratio is reported:
          * Extract learnable_parameter_ratio (e.g., 0.0001 for 0.01%, 0.001 for 0.1%)
          * Look for phrases like:
            - "only 0.01% of parameters"
            - "trainable parameters account for 0.1%"
            - "parameter efficiency: 0.01%"
        - CRITICAL: If both backbone_parameter_count AND ratio are available, extract BOTH.
        - The ratio allows calculation of backbone size when only ratio is given.

        TASK EXTRACTION:
        - Extract the primary vision task that each model performs.
        - Classify the task into one of the following categories:
          * "Image Classification": Classifying images into predefined categories
          * "Object Detection": Detecting and localizing objects in images
          * "Semantic Segmentation": Pixel-level classification of image regions
          * "Instance Segmentation": Detecting and segmenting individual object instances
          * "Image Generation": Generating new images (e.g., GANs, diffusion models)
          * "Image Captioning": Generating textual descriptions of images
          * "Visual Question Answering": Answering questions about images
          * "Other": Any vision task that does not fit into the above categories
        - If the task is not clearly a vision task or does not fit the categories, use "Other"

        OUTPUT CONSTRAINTS:
        - Output a JSON object with a "models" field containing an array of model objects.
        - Use '{provided_paper_id}' as the paper_id for all models.
        - If no models are proposed, return {{"models": []}}.
        - Prioritize extracting backbone_parameter_count over learnable_parameter_count.
        - If only ratio is given without backbone count, set learnable_parameter_ratio.

        {format_instructions}
        """
    ),
    (
        "human",
        """
        SEARCH the following parameter-efficient fine-tuning paper for PROPOSED models and their parameter information.

        Paper Name : {provided_paper_name}
        Paper ID : {provided_paper_id}
        
        Paper Type from Chain 1: {paper_type}
        
        CRITICAL EXTRACTION PRIORITIES:
        1. BACKBONE MODEL PARAMETERS (HIGHEST PRIORITY):
           - Extract the FULL parameter count of the backbone/base model (e.g., GPT-3: 175B, BERT-base: 110M)
           - This is NOT the trainable parameters introduced by efficient fine-tuning methods, but the entire backbone model size
           - Look for: "base model", "pre-trained model", "backbone", "full model size", "original model parameters"
        
        2. LEARNABLE PARAMETERS:
           - Extract trainable parameter count introduced by efficient fine-tuning methods if explicitly stated
           - This includes parameters from LoRA, Adapters, Prefix Tuning, or other parameter-efficient methods
        
        3. PARAMETER RATIO (FALLBACK):
           - If backbone parameter count is NOT reported but ratio is given (e.g., "0.01% of parameters"), extract the ratio
        
        IMPORTANT: 
        - Only extract models/methods that are PROPOSED by the authors of this paper.
        - Do NOT extract baseline models, comparison models, or existing methods from previous work.
        - CRITICAL: Extract backbone TOTAL parameters, not just trainable parameters from efficient fine-tuning methods.
        - If the paper only mentions trainable parameter count without backbone size, still search for backbone information.
        - The paper may or may not explicitly mention "LoRA" - focus on parameter-efficient fine-tuning methods in general.
        
        Paper content (markdown):
        ---
        {context}
        ---

        Return a JSON object with a "models" field containing an array of model objects.
        Each model should include: model_name, backbone_model_name, backbone_parameter_count, backbone_parameter_unit, 
        learnable_parameter_count, learnable_parameter_unit, learnable_parameter_ratio, task.
        Remember: 
        - Prioritize extracting backbone total parameters over trainable parameters.
        - Use only "B" (billions) or "M" (millions) for parameter units. Convert K (thousands) to M by dividing by 1000.
        - Classify tasks into one of the vision task categories: Image Classification, Object Detection, Semantic Segmentation, Instance Segmentation, Image Generation, Image Captioning, Visual Question Answering, or Other.
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
    Chain 2: Efficient fine-tuning 논문에서 백본 모델 및 파라미터 정보 탐색
    
    Uses a medium model to search for and extract backbone model names, backbone total parameters,
    and trainable parameters from parameter-efficient fine-tuning papers.
    Leverages paper_record from Chain 1 to generate optimized search queries.
    """
    if llm_instance is None:
        llm_instance = models_llm
    
    # Chain 2: Parameter-efficient fine-tuning에 특화된 검색 쿼리 생성
    # 백본 모델의 전체 파라미터 수를 우선적으로 검색
    base_query = """
    backbone model parameters, base model parameters, pre-trained model size,
    full model parameters, total model parameters, backbone model name,
    original model parameters, base model size,
    GPT-3 parameters, BERT parameters, T5 parameters, ViT parameters,
    trainable parameters, learnable parameters, adapter parameters,
    parameter efficient fine-tuning, efficient fine-tuning,
    parameter efficiency ratio, trainable parameters percentage,
    fine-tuning parameters, introduced parameters
    """
    
    # Chain 1 결과를 활용한 검색 쿼리 보완
    if paper_record:
        paper_info_parts = []
        
        if paper_record.has_explicit_models:
            paper_info_parts.append("base model architecture")
        
        if paper_record.efficiency_mentioned:
            paper_info_parts.append("parameter efficient fine-tuning")
        
        if paper_info_parts:
            search_query = f"{base_query}, {', '.join(paper_info_parts)}"
        else:
            search_query = base_query
    else:
        search_query = base_query
    
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



BASE_PATH = "./papers/result"


def initialize_all_models(custom_config: dict = None):
    """
    Initialize all LLM models for the chain of chains architecture.
    
    Args:
        custom_config: Optional dict with model configurations:
            {
                "paper": {"model": "model_name", "context_size": 8192},
                "models": {"model": "model_name", "context_size": 4096}
            }
    """
    global paper_llm, models_llm
    
    print("[System] Initializing Chain of Chains Architecture (Paper + Models)")
    print("=" * 60)
    
    paper_config = custom_config.get("paper", {}) if custom_config else {}
    models_config = custom_config.get("models", {}) if custom_config else {}
    
    # Chain 1: Paper extraction (large model for comprehensive understanding)
    paper_llm = initialize_model_chain("paper", paper_config)
    
    # Chain 2: Models extraction (medium model for structured extraction)
    models_llm = initialize_model_chain("models", models_config)
    
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
                "models": {"model": ollama_model_name, "context_size": context_size}
            }
    
    # Initialize all models
    initialize_all_models(custom_config)
    
    md_dir = os.path.join(BASE_PATH, "markdown")
    out_dir = os.path.join(BASE_PATH, "extracted")
    os.makedirs(out_dir, exist_ok=True)

    papers, models = [], []

    print("[System] Begin Automatic Model Text Mining with Chain of Chains (Paper + Models)")
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
        print()

    # Save results
    pd.DataFrame(papers).to_csv(os.path.join(out_dir, "papers.csv"), index=False)
    pd.DataFrame(models).to_csv(os.path.join(out_dir, "models.csv"), index=False)

    print("[DONE] Chain of Chains extraction completed.")
    print(f"  - Papers: {len(papers)}")
    print(f"  - Models: {len(models)}")


if __name__ == "__main__":
    main()
