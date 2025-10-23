from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, AcceleratorDevice, AcceleratorOptions
import re
from .img_summaries import process_pdf
from collections import OrderedDict
import requests
from pathlib import Path
import fitz  # PyMuPDF
from typing import Union

def download_pdf(url: str, tmp_dir="tmp_pdfs") -> Path:
    Path(tmp_dir).mkdir(exist_ok=True)
    filename = url.split("/")[-1]
    file_path = Path(tmp_dir) / filename

    r = requests.get(url)
    r.raise_for_status()  # raise error if download fails

    with open(file_path, "wb") as f:
        f.write(r.content)
    return file_path

#Replace each base64 image with its corresponding summary
def replace_base64_images(md_text, summary_dict):
    pattern = r'!\[.*?\]\(data:image\/png;base64,[A-Za-z0-9+/=\n]+\)'

    def replacement(match):
        # Get next unused key from the summaries dict
        if summary_dict:
            key, value = summary_dict.popitem(last=False)  # pop the first item
            return f"\n\n{value}\n\n"
        else:
            return "\n\n[Image removed - no summary available]\n\n"

    return re.sub(pattern, replacement, md_text)

def generate_pdf_thumbnail(pdf_path: Union[str, Path], output_dir: Union[str, Path] = "tmp_thumbnails") -> Path:
    """
    Generate a thumbnail (PNG) from the first page of a PDF.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the thumbnail

    Returns:
        Path to the generated PNG thumbnail
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Open PDF
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # first page

    # Render page to image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # scale 2x for better resolution
    thumbnail_path = output_dir / f"{pdf_path.stem}_thumbnail.png"
    pix.save(thumbnail_path)
    doc.close()

    return thumbnail_path

# def convert_pdf_to_markdown(pdf_path: str, summaries) -> str:
#     pipeline_options = PdfPipelineOptions(
#         do_ocr=True,
#         do_table_structure=True,
#         generate_picture_images=True,
#         generate_page_images=True,
#         do_formula_enrichment=True,
#         images_scale=2,
#         table_structure_options={"do_cell_matching": True},
#         ocr_options=RapidOcrOptions(),
#         accelerator_options=AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU),
#     )

#     format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
#     converter = DocumentConverter(format_options=format_options)
    
#     result = converter.convert(pdf_path)
#     markdown_text = result.document.export_to_markdown(image_mode="embedded")
#     new_markdown = replace_base64_images(markdown_text, summaries.copy())  # copy to preserve original
#     markdown_text = new_markdown

#     return markdown_text  where to put all above