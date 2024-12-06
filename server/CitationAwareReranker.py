import re
import os
from typing import Optional, Sequence
import operator
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from ScoredCrossEncoderReranker import ScoredCrossEncoderReranker

class CitationAwareReranker(ScoredCrossEncoderReranker):
    """Document reranker that considers both semantic similarity and citation relevance."""
    citation_weight: float = 0.3
    citation_patterns: list[str] = [
        # Standard formats
        r'Article\s*(\d+)(?:\s*\((\d+)\))?',       # Article 6(1)
        r'Art\.\s*(\d+)(?:\s*\((\d+)\))?',         # Art. 6(1)
        r'art\.\s*(\d+)(?:\s*\((\d+)\))?',         # art. 6(1)
        r'§\s*(\d+)(?:\s*\((\d+)\))?',             # § 6(1)
        
        # Asterisk/special character formats
        r'\*Art\.\s*(\d+)(?:\s*\((\d+)\))?\*',     # *Art. 6(1)*
        r'Art\.\s*(\d+)(?:\s*\((\d+)\))?',         # Art. 6(1)
        r'\*Article\s*(\d+)(?:\s*\((\d+)\))?\*',   # *Article 6(1)*
        
        # Different bracket styles
        r'Art(?:icle)?\s*(\d+)(?:\s*\[(\d+)\])',   # Article 6[1]
        r'Art(?:icle)?\s*(\d+)(?:\s*\{(\d+)\})',   # Article 6{1}
        
        # No space variations
        r'Art(?:icle)?(\d+)\((\d+)\)',             # Article6(1)
        r'Art(?:icle)?(\d+)\.(\d+)',               # Article6.1
        
        # Period notation
        r'Art(?:icle)?\s*(\d+)\.(\d+)',            # Article 6.1
        
        # Different letter cases
        r'ARTICLE\s*(\d+)(?:\s*\((\d+)\))?',       # ARTICLE 6(1)
        r'ART\.\s*(\d+)(?:\s*\((\d+)\))?',         # ART. 6(1)
        
        # Slash notation
        r'Art(?:icle)?\s*(\d+)/(\d+)',             # Article 6/1
        
        # With sub-letters
        r'Art(?:icle)?\s*(\d+)(?:\s*\((\d+[a-z])\))?',  # Article 6(1a)
        
        # With roman numerals in subsection
        r'Art(?:icle)?\s*(\d+)(?:\s*\(([ivxIVX]+)\))?'  # Article 6(iv)
    ]
    exact_match_weight=0.4
    context_weight=0.3
    proximity_weight=0.3
    context_window=200
    
    context_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    context_window = 200
    
    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize base parameters
        self.citation_weight = float(os.getenv('citation_weight', '0.3'))
        
        # Initialize weights for different components
        self.exact_match_weight = float(os.getenv('citation_exact_match_weight', '0.4'))
        self.context_weight = float(os.getenv('citation_context_weight', '0.3'))
        self.proximity_weight = float(os.getenv('citation_proximity_weight', '0.3'))
        
        # Initialize context encoder for semantic similarity
        self.context_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Window size for context extraction
        self.context_window = int(os.getenv('citation_context_window', '200'))

    def _get_citation_context(self, citation: str, text: str) -> str:
        """Extract context window around citation."""
        citation_pos = text.lower().find(citation.lower())
        if citation_pos == -1:
            return text  # Return full text if citation not found
            
        start = max(0, citation_pos - self.context_window)
        end = min(len(text), citation_pos + len(citation) + self.context_window)
        
        return text[start:end]

    def _extract_citations(self, text: str) -> list[str]:
        """
        Extract citations from text.
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List of normalized citation strings
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get the main article number and subsection
                article_num = match.group(1)
                subsection = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                
                # Create normalized citation format
                citation = f"article_{article_num}"
                if subsection:
                    citation += f"_{subsection}"
                    
                citations.append(citation.lower())
        
        return list(set(citations))  # Remove duplicates

    def _calculate_proximity_score(self, citation: str, content: str, query: str) -> float:
        """Calculate how close the citation is to query-relevant content."""
        citation_pos = content.lower().find(citation.lower())
        if citation_pos == -1:
            return 0.0
        
        # Get context window around citation
        window_size = self.context_window
        citation_window = content[max(0, citation_pos - window_size):
                                min(len(content), citation_pos + window_size)]
        
        # Calculate semantic similarity with query
        similarity = self.context_encoder.predict([query, citation_window])
        
        # Adjust score based on position in document
        relative_pos = citation_pos / len(content)
        position_weight = 1 - (abs(0.5 - relative_pos) * 0.5)  # Prefer middle content
        
        return float(similarity * position_weight)

    def _calculate_citation_score(self, query_citations: list[str], doc_citations: list[str], 
                                query: str, doc_content: str) -> float:
        """
        Calculate enhanced citation relevance score using semantic similarity and proximity.
        
        Args:
            query_citations: List of citations from query
            doc_citations: List of citations from document
            query: Original query text
            doc_content: Full document content
            
        Returns:
            Float score between 0 and 1
        """
        if not query_citations:
            return 1.0
        
        citation_scores = []
        
        for qc in query_citations:
            # Get semantic embeddings for contexts
            query_citation_context = self._get_citation_context(qc, query)
            best_match_score = 0.0
            
            for dc in doc_citations:
                doc_citation_context = self._get_citation_context(dc, doc_content)
                
                # Calculate three components:
                # 1. Direct citation match
                exact_match = 1.0 if qc.lower() == dc.lower() else 0.0
                
                # 2. Semantic similarity between citation contexts
                context_similarity = self.context_encoder.predict([
                    query_citation_context,
                    doc_citation_context
                ])
                
                # 3. Citation proximity to relevant content
                proximity_score = self._calculate_proximity_score(dc, doc_content, query)
                
                # Combine scores with weights
                combined_score = (
                    exact_match * self.exact_match_weight +
                    context_similarity * self.context_weight +
                    proximity_score * self.proximity_weight
                )
                
                best_match_score = max(best_match_score, combined_score)
            
            citation_scores.append(best_match_score)
        
        # Consider both average and maximum scores
        avg_score = sum(citation_scores) / len(citation_scores)
        max_score = max(citation_scores)
        
        return avg_score * 0.7 + max_score * 0.3

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using both semantic similarity and citation awareness.

        Args:
            documents: A sequence of documents to compress
            query: The query to use for compression
            callbacks: Callbacks for the compression process

        Returns:
            Reranked sequence of documents
        """
        # Get semantic scores using parent class's model
        semantic_scores = self.model.score([(query, doc.page_content) for doc in documents])
        
        # Extract citations from query and documents
        query_citations = self._extract_citations(query)
        doc_citations_list = [self._extract_citations(doc.page_content) for doc in documents]
        
        # Calculate citation scores with enhanced method
        citation_scores = [
            self._calculate_citation_score(
                query_citations, 
                doc_citations,
                query,
                doc.page_content
            )
            for doc, doc_citations in zip(documents, doc_citations_list)
        ]
        
        # Combine scores with weighting
        combined_scores = [
            (semantic_score * (1 - self.citation_weight) + 
             citation_score * self.citation_weight)
            for semantic_score, citation_score in zip(semantic_scores, citation_scores)
        ]
        
        # Combine documents with their scores
        docs_with_scores = list(zip(documents, combined_scores, semantic_scores, citation_scores))
        
        # Sort by combined score
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        
        # Return top_n documents with metadata
        return [
            doc.copy(update={"metadata": {
                **doc.metadata,
                "relevance_score": combined_score,
                "semantic_score": semantic_score,
                "citation_score": citation_score,
                "citations_found": doc_citations_list[i]
            }}) 
            for i, (doc, combined_score, semantic_score, citation_score) in enumerate(result[:self.top_n])
        ]
