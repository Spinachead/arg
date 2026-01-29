from langchain_core.documents import Document
from typing import Optional

class DocumentWithVSId(Document):
    """
    矢量化后的文档
    """

    id: Optional[str] = None
    score: float = 3.0
