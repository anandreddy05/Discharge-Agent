from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_core.documents import Document


class DocumentProcessor:
    def __init__(self):
        # Configure pipeline options for PDF processing
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 1.0

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process_pdf_to_markdown(self, file_path: str) -> str:
        """Converts a PDF directly into a structured Markdown string."""
        result = self.converter.convert(file_path)
        return result.document.export_to_markdown()
