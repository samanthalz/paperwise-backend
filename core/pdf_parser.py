import time
import logging
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, AcceleratorOptions, AcceleratorDevice
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem, ImageRefMode

_log = logging.getLogger(__name__)
IMAGE_RESOLUTION_SCALE = 2.0

def parse_pdf(pdf_path: str):
    """
    Parse PDF using Docling → extract markdown (with base64 images) 
    and raw image/table elements (for later summarization).
    """
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_page_images=True,
        generate_picture_images=True,
        do_formula_enrichment=True,
        images_scale=IMAGE_RESOLUTION_SCALE,
        table_structure_options={"do_cell_matching": True},
        ocr_options=RapidOcrOptions(),
        accelerator_options=AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
    )

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    start_time = time.time()
    conv_res = doc_converter.convert(pdf_path)

    # Extract image/table elements
    image_elements = [
        (element, _level)
        for element, _level in conv_res.document.iterate_items()
        if isinstance(element, (TableItem, PictureItem))
    ]

    # Export markdown
    markdown_text = conv_res.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)

    _log.info(f"PDF parsed in {time.time()-start_time:.2f}s, found {len(image_elements)} images/tables.")

    return markdown_text, image_elements, conv_res
