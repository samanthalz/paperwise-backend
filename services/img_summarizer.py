import logging
import io
from collections import OrderedDict
from PIL import Image 
from docling_core.types.doc import TableItem

import google.generativeai as genai

_log = logging.getLogger(__name__)
IMAGE_RESOLUTION_SCALE = 2.0

def optimize_image_size(pil_image, max_size=(1024, 1024)):
    """Resize image if too large, maintaining aspect ratio"""
    if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
        pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return pil_image

def get_prompt_for_element(element):
    """Tailored prompts for different element types"""
    if isinstance(element, TableItem):
        return """Please analyze this table and provide a comprehensive summary including:
- Main subject and type of data presented
- Key columns and their meaning
- Notable patterns, trends, or comparisons
- Summary of key findings or conclusions"""
    else:
        return """Please analyze this figure and provide a detailed summary including:
- Type of visualization (chart, graph, diagram, etc.)
- Main subject and key elements shown
- Patterns, trends, or relationships visible
- Overall significance or message conveyed""" 

def summarize_with_gemini_pil(image_elements, conv_res, api_key: str, doc_filename: str) -> OrderedDict:
    """
    Summarize images using Gemini with PIL images directly from Docling.
    Returns OrderedDict of {image_filename: summary_text}.
    """
    genai.configure(api_key=api_key)    
    model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
    summaries = OrderedDict()

    table_counter = 0
    picture_counter = 0

    for element, _level in image_elements:
        try:
            pil_image = element.get_image(conv_res.document)
            
            if pil_image.mode in ('RGBA', 'P', 'LA'):
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                if pil_image.mode == 'RGBA':
                    background.paste(pil_image, mask=pil_image.split()[-1])
                else:
                    background.paste(pil_image)
                pil_image = background
            
            pil_image = optimize_image_size(pil_image, max_size=(1024, 1024))
            
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG', optimize=True)
            img_data = img_byte_arr.getvalue()
            
            if isinstance(element, TableItem):
                table_counter += 1
                filename = f"{doc_filename}-table-{table_counter}.png"
            else:
                picture_counter += 1
                filename = f"{doc_filename}-picture-{picture_counter}.png"
            
            prompt = get_prompt_for_element(element)
            
            response = model.generate_content([
                {'mime_type': 'image/png', 'data': img_data},
                prompt
            ])
            
            summary_text = response.text.strip() if response.text else "No description could be generated for this image."
            summaries[filename] = summary_text
            
        except Exception as e:
            _log.error(f"Error processing element: {str(e)}")
            if isinstance(element, TableItem):
                table_counter += 1
                filename = f"{doc_filename}-table-{table_counter}.png"
            else:
                picture_counter += 1
                filename = f"{doc_filename}-picture-{picture_counter}.png"
            summaries[filename] = f"Error generating description: {str(e)}"

    return summaries
