import os
from typing import List
from interface.base_datastore import DataItem
from interface.base_indexer import BaseIndexer
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker, DocChunk

#chunks documents
class Indexer(BaseIndexer):
    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HybridChunker()
        # Disable tokenizers parallelism to avoid OOM errors.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    #input list of documents and coverts each into chunked doc object 
    #returns list of chunked doc objects
    def index(self, document_paths: List[str]) -> List[DataItem]:
        items = []
        for document_path in document_paths:
            document = self.converter.convert(document_path).document
            chunks: List[DocChunk] = self.chunker.chunk(document)
            items.extend(self._items_from_chunks(chunks))
        return items


    #input list of chunks
    #adds dataitem with info about source document to each chunk
    def _items_from_chunks(self, chunks: List[DocChunk]) -> List[DataItem]:
        items = []
        for i, chunk in enumerate(chunks):
            content_headings = "## " + ", ".join(chunk.meta.headings)
            content_text = f"{content_headings}\n{chunk.text}"
            
            # Add token length check
            if self._estimate_tokens(content_text) > 450:  # Buffer below 512
                # Split large chunks further
                sub_chunks = self._split_large_chunk(content_text)
                for j, sub_chunk in enumerate(sub_chunks):
                    source = f"{chunk.meta.origin.filename}:{i}:{j}"
                    item = DataItem(content=sub_chunk, source=source)
                    items.append(item)
            else:
                source = f"{chunk.meta.origin.filename}:{i}"
                item = DataItem(content=content_text, source=source)
                items.append(item)
        
        return items

    def _estimate_tokens(self, text: str) -> int:
        # Rough estimate: ~0.75 tokens per word for English
        return len(text.split()) * 0.75

    def _split_large_chunk(self, text: str, max_tokens: int = 450) -> List[str]:
        # Simple splitting by sentences or paragraphs
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            if current_length + sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = [sentence]
                    current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks