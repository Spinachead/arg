from typing import List

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader


class RapidOCRDocLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def doc2text(filepath):
            from io import BytesIO

            import numpy as np
            from docx import Document, ImagePart
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.table import Table, _Cell
            from docx.text.paragraph import Paragraph
            from PIL import Image
            
            print(f"[DEBUG] doc2text starting for {filepath}")
            
            # 使用全局 OCR 单例，避免重复初始化
            from file_rag.document_loaders.ocr import get_ocr
            print(f"[DEBUG] Initializing OCR...")
            ocr = get_ocr()
            print(f"[DEBUG] OCR initialized")
            
            print(f"[DEBUG] Opening DOCX: {filepath}")
            doc = Document(filepath)
            print(f"[DEBUG] DOCX opened, starting to parse blocks")
            resp_parts = []  # 使用列表收集，最后合并，比字符串拼接更高效

            def iter_block_items(parent):
                from docx.document import Document

                if isinstance(parent, Document):
                    parent_elm = parent.element.body
                elif isinstance(parent, _Cell):
                    parent_elm = parent._tc
                else:
                    raise ValueError("RapidOCRDocLoader parse fail")

                for child in parent_elm.iterchildren():
                    if isinstance(child, CT_P):
                        yield Paragraph(child, parent)
                    elif isinstance(child, CT_Tbl):
                        yield Table(child, parent)

            # 移除 tqdm 进度条，减少开销
            block_count = 0
            for block in iter_block_items(doc):
                block_count += 1
                if isinstance(block, Paragraph):
                    text = block.text.strip()
                    if text:
                        resp_parts.append(text)
                    
                    # 处理图片 OCR
                    images = block._element.xpath(".//pic:pic")
                    if images:
                        for image in images:
                            for img_id in image.xpath(".//a:blip/@r:embed"):
                                try:
                                    part = doc.part.related_parts[img_id]
                                    if isinstance(part, ImagePart):
                                        img = Image.open(BytesIO(part._blob))
                                        result, _ = ocr(np.array(img))
                                        if result:
                                            ocr_result = [line[1] for line in result]
                                            resp_parts.append("\n".join(ocr_result))
                                except Exception as e:
                                    # 忽略单张图片处理失败
                                    continue
                                    
                elif isinstance(block, Table):
                    for row in block.rows:
                        for cell in row.cells:
                            for paragraph in cell.paragraphs:
                                text = paragraph.text.strip()
                                if text:
                                    resp_parts.append(text)
            
            print(f"[DEBUG] Parsed {block_count} blocks, extracted {len(resp_parts)} text parts")
            return "\n".join(resp_parts)

        text = doc2text(self.file_path)
        print(f"[DEBUG] doc2text completed, text length: {len(text)}")
        
        from unstructured.partition.text import partition_text
        print(f"[DEBUG] Starting partition_text...")
        result = partition_text(text=text, **self.unstructured_kwargs)
        print(f"[DEBUG] partition_text completed, got {len(result)} elements")
        return result


if __name__ == "__main__":
    loader = RapidOCRDocLoader(file_path="../tests/samples/ocr_test.docx")
    docs = loader.load()
