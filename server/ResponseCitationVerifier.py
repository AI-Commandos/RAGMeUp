from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import re
from sentence_transformers import CrossEncoder
from langchain.schema import Document
import logging
import os

@dataclass
class ResponseCitation:
    """Citation found in LLM response."""
    text: str            # Full citation text as found
    reference: str       # Standardized reference
    context: str         # Surrounding text
    span: Tuple[int, int]  # Position in text

class ResponseVerifier:
    """Verifies citations in LLM responses against source documents."""
    
    def __init__(self):
        # Initialize cross-encoder for semantic similarity
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Basic article references
        article_base = [
            r'Article',      # Article 6(1)
            r'Art\.',        # Art. 6(1)
            r'Art',          # Art 6(1)
            r'article',      # article 6(1)
            r'art\.',        # art. 6(1)
            r'art',          # art 6(1)
            r'§',            # § 6(1)
        ]

        # Numbers and subsections patterns
        number_patterns = [
            r'\s*(\d+)',                            # Basic number: 6
            r'\s*(\d+[a-z])',                       # Letter suffix: 6a
            r'\s*(\d+[a-z]?)\s*\((\d+[a-z]?)\)',   # Number with subsection: 6(1) or 6a(1b)
            r'\s*(\d+[a-z]?)\s*\(([ivx]+)\)',      # Roman numeral subsection: 6(iv)
            r'\s*(\d+[a-z]?)\s*\(([A-Z])\)',       # Letter subsection: 6(A)
            r'\s*(\d+[a-z]?)\s*\[(\d+[a-z]?)\]',   # Square bracket notation: 6[1]
            r'\s*(\d+[a-z]?)\s*\{(\d+[a-z]?)\}',   # Curly bracket notation: 6{1}
            r'\s*(\d+[a-z]?)/(\d+[a-z]?)',         # Slash notation: 6/1
        ]

        # Build comprehensive patterns
        self.citation_patterns = []
        for base in article_base:
            for num_pattern in number_patterns:
                # Standard format
                self.citation_patterns.append(
                    rf'{base}{num_pattern}'  # Article 6(1)
                )
                # Parenthetical format
                self.citation_patterns.append(
                    rf'\({base}{num_pattern}\)'  # (Article 6(1))
                )
                # Format with comma or semicolon prefix
                self.citation_patterns.append(
                    rf'[,;]\s*{base}{num_pattern}'  # , Article 6(1)
                )
                # Format with "under", "in", "of", "per" prefix
                self.citation_patterns.append(
                    rf'(?:under|in|of|per)\s+{base}{num_pattern}'  # under Article 6(1)
                )

        # Additional special formats
        special_formats = [
            # Hash notation
            r'Art\s*#\s*(\d+[a-z]?)(?:\s*\((\d+[a-z]?)\))?',
            # Dash notation
            r'Art\s*-\s*(\d+[a-z]?)(?:\s*\((\d+[a-z]?)\))?',
            # No space between Art and number
            r'Art(\d+[a-z]?)(?:\((\d+[a-z]?)\))?',
            # Multiple subsections
            r'Article\s*(\d+[a-z]?)(?:\((\d+[a-z]?(?:,\s*\d+[a-z]?)*)\))',
            # Range of articles
            r'Articles?\s*(\d+[a-z]?)\s*(?:-|to|through)\s*(\d+[a-z]?)',
            # Combined articles
            r'Articles?\s*(\d+[a-z]?)\s*(?:and|&)\s*(\d+[a-z]?)',
            # Multilingual variations (for EU texts)
            r'(?:artikel|artikel|artículo|articolo|artigo)\s*(\d+[a-z]?)(?:\((\d+[a-z]?)\))?',
        ]
        self.citation_patterns.extend(special_formats)
        
        # Configuration
        self.context_window = 100  # Characters before/after citation
        self.context_window = int(os.getenv("citation_context_window"))
        self.similarity_threshold = 0.7
        self.similarity_threshold = float(os.getenv("citation_similarity_threshold"))
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def verify_response(self, 
                       response: str,
                       retrieved_docs: List[Document]) -> Tuple[str, Dict]:
        """
        Verify citations in LLM response against source documents.
        
        Args:
            response: The LLM's response text
            retrieved_docs: List of source documents to verify against
            
        Returns:
            Tuple containing:
            - Modified response with verification markers
            - Dictionary of verification results
        """
        try:
            # Extract citations from response
            citations = self._extract_citations(response)
            
            # Get citations from source documents
            source_citations = self._get_source_citations(retrieved_docs)
            
            verification_results = {}
            modified_response = response
            
            # Verify each citation
            for citation in citations:
                try:
                    # Find matching citations in sources
                    matches = self._find_matches(citation, source_citations)
                    
                    if not matches:
                        # Handle non-existent citation
                        verification_results[citation.text] = {
                            'status': 'not_found',
                            'message': f'Citation {citation.text} not found in source materials'
                        }
                        modified_response = self._mark_invalid(modified_response, citation)
                        continue
                    
                    # Check context similarity for existing citations
                    best_match = self._find_best_context_match(citation, matches)
                    
                    if best_match['score'] < self.similarity_threshold:
                        # Handle context mismatch
                        verification_results[citation.text] = {
                            'status': 'context_mismatch',
                            'message': 'Citation used in incorrect context',
                            'source': best_match['source'],
                            'original_context': best_match['context']
                        }
                        modified_response = self._mark_context_mismatch(
                            modified_response, citation, best_match['source']
                        )
                    else:
                        # Valid citation
                        verification_results[citation.text] = {
                            'status': 'verified',
                            'source': best_match['source']
                        }
                        modified_response = self._add_source_reference(
                            modified_response, citation, best_match['source']
                        )
                except Exception as e:
                    self.logger.error(f"Error processing citation {citation.text}: {str(e)}")
                    continue
            
            return modified_response, verification_results
            
        except Exception as e:
            self.logger.error(f"Error in verify_response: {str(e)}")
            return response, {'error': str(e)}
    
    def _extract_citations(self, text: str) -> List[ResponseCitation]:
        """Extract all citations from text with their context."""
        citations = []
        compiled_patterns = self._compile_patterns()
        
        for pattern in compiled_patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                context_start = max(0, start - self.context_window)
                context_end = min(len(text), end + self.context_window)
                
                # Extract the full reference
                main_num = match.group(1) if match.group(1) else ""
                sub_num = match.group(2) if len(match.groups()) > 1 and match.group(2) else ""
                
                # Handle special cases
                if "-" in main_num or "to" in text[start:end] or "through" in text[start:end]:
                    parts = main_num.replace("to", "-").replace("through", "-").split("-")
                    main_num = f"{parts[0]}-{parts[1]}"
                elif "and" in text[start:end] or "&" in text[start:end]:
                    parts = main_num.replace("and", "&").split("&")
                    main_num = f"{parts[0]}&{parts[1]}"
                
                reference = main_num + (f"({sub_num})" if sub_num else "")
                
                citations.append(ResponseCitation(
                    text=match.group(0).strip(),
                    reference=reference,
                    context=text[context_start:context_end],
                    span=(start, end)
                ))
        
        return self._deduplicate_citations(citations)
    
    def _get_source_citations(self, docs: List[Document]) -> List[Dict]:
        """Extract citations from source documents."""
        source_citations = []
        compiled_patterns = self._compile_patterns()
        
        for doc in docs:
            for pattern in compiled_patterns:
                for match in pattern.finditer(doc.page_content):
                    start, end = match.span()
                    context_start = max(0, start - self.context_window)
                    context_end = min(len(doc.page_content), end + self.context_window)
                    
                    main_num = match.group(1) if match.group(1) else ""
                    sub_num = match.group(2) if len(match.groups()) > 1 and match.group(2) else ""
                    reference = main_num + (f"({sub_num})" if sub_num else "")
                    
                    source_citations.append({
                        'reference': reference,
                        'context': doc.page_content[context_start:context_end],
                        'source': doc.metadata.get('source', 'unknown')
                    })
        
        return source_citations
    
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile all regex patterns with case insensitivity."""
        return [re.compile(pattern, re.IGNORECASE) for pattern in self.citation_patterns]
    
    def _find_matches(self, 
                     citation: ResponseCitation, 
                     source_citations: List[Dict]) -> List[Dict]:
        """Find matching citations in source documents."""
        return [sc for sc in source_citations if sc['reference'] == citation.reference]
    
    def _find_best_context_match(self, 
                                citation: ResponseCitation,
                                matches: List[Dict]) -> Dict:
        """Find best matching context using semantic similarity."""
        scores = []
        for match in matches:
            score = self.cross_encoder.predict([
                citation.context, match['context']
            ])
            scores.append({
                'score': score,
                'source': match['source'],
                'context': match['context']
            })
        return max(scores, key=lambda x: x['score'])
    
    def _deduplicate_citations(self, citations: List[ResponseCitation]) -> List[ResponseCitation]:
        """Remove duplicate citations while keeping the most complete reference."""
        seen_refs = {}
        for citation in citations:
            existing = seen_refs.get(citation.reference)
            if not existing or len(citation.text) > len(existing.text):
                seen_refs[citation.reference] = citation
        return list(seen_refs.values())
    
    def _mark_invalid(self, text: str, citation: ResponseCitation) -> str:
        """Mark invalid citations in text."""
        start, end = citation.span
        return (
            text[:start] + 
            f"⚠️ [Invalid citation: {citation.text}]" + 
            text[end:]
        )
    
    def _mark_context_mismatch(self, text: str, citation: ResponseCitation,
                              source: str) -> str:
        """Mark citations with context mismatch."""
        start, end = citation.span
        return (
            text[:start] + 
            f"{citation.text} [⚠️ Context mismatch, see {source}]" + 
            text[end:]
        )
    
    def _add_source_reference(self, text: str, citation: ResponseCitation,
                            source: str) -> str:
        """Add source reference to verified citations."""
        start, end = citation.span
        return (
            text[:end] + 
            f" [{source}]" + 
            text[end:]
        )
