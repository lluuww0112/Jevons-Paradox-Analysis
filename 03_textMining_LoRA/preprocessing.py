from docling.document_converter import DocumentConverter

import os
import time
from tqdm import tqdm
import logging


# stop using converter logging
logging.getLogger("docling").setLevel(logging.ERROR)
logging.getLogger("docling_core").setLevel(logging.ERROR)

converter = DocumentConverter()

# PDF to MarkDown
def convert_pdf_to_md(paper_name : str, save_dir_path : str = "./result/markdown", root_path : str = "./papers"):
    """_summary_

        convert pdf to markdown
    Args:
        paper_name (str): title of the paper
        save_dir_path (str, optional): set save path(dir) 
        root_path (str, optional): set dir where papers are placed

    Returns:
        (int): elasped processing time
    """
    global converter
    
    # get base path
    full_paper_path = os.path.join(root_path, "res", f"{paper_name}.pdf")
    full_save_dir_path = os.path.join(root_path, save_dir_path)
    full_save_path = os.path.join(full_save_dir_path, f"{paper_name}.md")

    # check save dir exists
    os.makedirs(full_save_dir_path, exist_ok=True)

    start = time.time()
    # convert pdf to md
    result = converter.convert(full_paper_path)
    markdown_content = result.document.export_to_markdown()

    # save md
    with open(full_save_path, "w", encoding="UTF-8") as file:
        file.write(markdown_content)
    elasped_time = time.time() - start

    return elasped_time


def get_PDFs(folder_path : str = "./papers/res"):
    pdf_files = []
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith('.pdf'):
                file_name_only = os.path.splitext(entry.name)[0]
                pdf_files.append(file_name_only)
    return pdf_files


if __name__ == "__main__":
    file_names = get_PDFs()
    
    os.makedirs("./papers/result/markdown", exist_ok=True)
    
    pbar = tqdm(file_names)
    for file_name in pbar:
        pbar.set_description(f"Processing {file_name}")
        elapsed_time = convert_pdf_to_md(file_name)
        pbar.set_postfix(last_duration=f"{elapsed_time:.2f}s")