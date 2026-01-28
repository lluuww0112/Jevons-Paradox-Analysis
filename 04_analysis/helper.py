import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

START = 2018
END = 2025

# RAG+LLM 추출 내용 load
def get_extracted_data(start : int = START, end : int = END):
    papers = {}
    models = {}
    metrics = {}

    for i in range(start, end+1):
        papers[f"{i}"] = pd.read_csv(f"./res/{i}/papers.csv")
        models[f"{i}"] = pd.read_csv(f"./res/{i}/models.csv")
        metrics[f"{i}"] = pd.read_csv(f"./res/{i}/metrics.csv")

    return papers, models, metrics

# 메타데이터 불러오는 헬퍼 함수
def get_metadata(start : int = START, end : int = END) -> dict[str, pd.DataFrame]:
    metadatas = {}

    for i in range(start, end+1):
        root = f"./res/{i}/metadata"
        metadatas[f"{i}"] = pd.concat(
            [
                pd.read_csv(os.path.join(root, file_name)).assign(keyword=file_name[:-4])
                for file_name in os.listdir(root)
                if file_name.endswith(".csv")
            ],
            axis=0,
            ignore_index=True
        )
        metadatas[f"{i}"] = metadatas[f"{i}"].dropna(subset=["Document Title"])
        metadatas[f"{i}"]["Abstract"] = metadatas[f"{i}"]["Abstract"].astype(str)
        
    return metadatas


metadatas = get_metadata()
papers, models, metrics = get_extracted_data()
