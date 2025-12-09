from typing import List

from langchain.embeddings.base import Embeddings

from langchain_core.documents import Document

# 延迟导入 KBService 以避免循环导入
from knowledge_base.kb_service.base import SupportedVSType
from utils import get_default_embedding


class DefaultKBService:
    def __init__(
        self,
        knowledge_base_name: str,
        kb_info: str = None,
        embed_model: str = get_default_embedding(),
    ):
        self.kb_name = knowledge_base_name
        self.kb_info = kb_info
        self.embed_model = embed_model
        self.kb_path = "/chroma_db"
        self.doc_path = "/documents"
        self.do_init()

    def do_create_kb(self):
        pass

    def do_drop_kb(self):
        pass

    def do_add_doc(self, docs: List[Document], **kwargs):
        pass

    def do_clear_vs(self):
        pass

    def vs_type(self) -> str:
        return SupportedVSType.DEFAULT

    def do_init(self):
        pass

    def do_search(self, query, top_k, score_threshold):
        pass

    def do_insert_multi_knowledge(self):
        pass

    def do_insert_one_knowledge(self):
        pass

    def do_delete_doc(self, kb_file, **kwargs):
        pass