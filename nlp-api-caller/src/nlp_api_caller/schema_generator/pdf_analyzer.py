"""
PDF Document Analyzer

Extracts entities and relationships from PDF documents to generate knowledge graph schemas.
Uses NER (Named Entity Recognition) and pattern matching.
"""

from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from collections import Counter
from .base import SchemaGenerator


class PDFAnalyzer(SchemaGenerator):
    """
    Generate knowledge graph schema by analyzing PDF documents.
    
    Usage:
        analyzer = PDFAnalyzer()
        
        # Single PDF
        schema = analyzer.generate('document.pdf')
        
        # Directory of PDFs
        schema = analyzer.generate('pdfs/', recursive=True)
        
        # List of PDFs
        schema = analyzer.generate(['doc1.pdf', 'doc2.pdf'])
        
        # Save to file
        analyzer.save_schema(schema, 'knowledge_graph_schema.json')
    """
    
    def get_schema_type(self) -> str:
        return 'knowledge_graph'
    
    def generate(self, source: Any, **kwargs) -> Dict[str, Any]:
        """
        Generate knowledge graph schema from PDF documents.
        
        Args:
            source: PDF path, directory, or list of paths
            **kwargs: Options:
                - recursive: Scan directories recursively (default: False)
                - use_ner: Use NLP for entity recognition (default: True)
                - entity_threshold: Min mentions to be considered entity type (default: 5)
                - sample_limit: Max files to analyze (default: 50)
        
        Returns:
            Knowledge graph schema dict
        """
        # Collect PDF files
        pdf_files = self._collect_pdf_files(source, **kwargs)
        
        if not pdf_files:
            raise ValueError("No PDF files found")
        
        # Limit samples
        sample_limit = kwargs.get('sample_limit', 50)
        if len(pdf_files) > sample_limit:
            print(f"⚠️  Analyzing first {sample_limit} of {len(pdf_files)} files")
            pdf_files = pdf_files[:sample_limit]
        
        # Extract text from PDFs
        print(f"Extracting text from {len(pdf_files)} PDF files...")
        documents = []
        for pdf_path in pdf_files:
            try:
                text = self._extract_pdf_text(pdf_path)
                if text:
                    documents.append({
                        'path': pdf_path,
                        'text': text
                    })
            except Exception as e:
                print(f"⚠️  Warning: Could not extract text from {pdf_path}: {e}")
        
        if not documents:
            raise ValueError("Could not extract text from any PDF files")
        
        print(f"✓ Extracted text from {len(documents)} documents")
        
        # Analyze documents
        use_ner = kwargs.get('use_ner', True)
        entity_threshold = kwargs.get('entity_threshold', 5)
        
        if use_ner:
            entities, relationships = self._analyze_with_ner(documents, entity_threshold)
        else:
            entities, relationships = self._analyze_with_patterns(documents, entity_threshold)
        
        # Build schema
        schema = {
            "type": "knowledge_graph",
            "version": "1.0.0",
            "metadata": {
                "generated_from": "pdf_document_analysis",
                "files_analyzed": len(documents),
                "entity_types": len(entities),
                "relationship_types": len(relationships),
                "method": "ner" if use_ner else "pattern_matching"
            },
            "entities": entities,
            "relationships": relationships
        }
        
        return schema
    
    def _collect_pdf_files(self, source: Any, **kwargs) -> List[str]:
        """Collect PDF file paths from source."""
        recursive = kwargs.get('recursive', False)
        
        if isinstance(source, str):
            path = Path(source)
            
            if path.is_file() and path.suffix.lower() == '.pdf':
                return [str(path)]
            elif path.is_dir():
                if recursive:
                    return [str(f) for f in path.rglob('*.pdf')]
                else:
                    return [str(f) for f in path.glob('*.pdf')]
            else:
                raise ValueError(f"Path does not exist or is not a PDF: {source}")
        
        elif isinstance(source, list):
            # List of file paths
            return [
                str(Path(f))
                for f in source
                if Path(f).exists() and Path(f).suffix.lower() == '.pdf'
            ]
        
        else:
            raise ValueError("Source must be file path, directory, or list of paths")
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            # Try PyPDF2 first (most common)
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
        except ImportError:
            try:
                # Fallback to pdfplumber
                import pdfplumber
                
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    return text
                    
            except ImportError:
                raise ImportError(
                    "PDF parsing requires PyPDF2 or pdfplumber. "
                    "Install with: pip install PyPDF2"
                )
    
    def _analyze_with_ner(
        self,
        documents: List[Dict[str, Any]],
        entity_threshold: int
    ) -> tuple:
        """
        Analyze documents using Named Entity Recognition.
        
        Uses spaCy for NER if available, otherwise falls back to pattern matching.
        """
        try:
            import spacy
            
            # Try to load English model
            try:
                nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("⚠️  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                print("Falling back to pattern matching...")
                return self._analyze_with_patterns(documents, entity_threshold)
            
            print("Using spaCy NER for entity extraction...")
            
            # Process all documents
            entity_mentions = {}  # entity_type -> Counter of mentions
            entity_examples = {}  # entity_type -> example mentions
            
            for doc_info in documents:
                text = doc_info['text']
                doc = nlp(text[:1000000])  # Limit to 1MB
                
                for ent in doc.ents:
                    entity_type = ent.label_
                    entity_text = ent.text
                    
                    if entity_type not in entity_mentions:
                        entity_mentions[entity_type] = Counter()
                        entity_examples[entity_type] = []
                    
                    entity_mentions[entity_type][entity_text] += 1
                    
                    if len(entity_examples[entity_type]) < 10:
                        entity_examples[entity_type].append(entity_text)
            
            # Build entity definitions
            entities = {}
            for entity_type, mentions in entity_mentions.items():
                total_mentions = sum(mentions.values())
                
                if total_mentions >= entity_threshold:
                    entities[entity_type] = {
                        "name": entity_type,
                        "description": self._get_ner_description(entity_type),
                        "properties": {
                            "name": {
                                "type": "string",
                                "required": True,
                                "description": f"{entity_type} name or identifier"
                            },
                            "mention_count": {
                                "type": "integer",
                                "required": False,
                                "description": "Number of times mentioned in documents"
                            }
                        },
                        "searchable_fields": ["name"],
                        "display_field": "name",
                        "examples": entity_examples[entity_type][:5]
                    }
            
            # Infer relationships (simple co-occurrence)
            relationships = self._infer_relationships_from_cooccurrence(
                documents,
                list(entities.keys()),
                nlp
            )
            
            return entities, relationships
            
        except ImportError:
            print("⚠️  spaCy not installed. Install with: pip install spacy")
            print("Falling back to pattern matching...")
            return self._analyze_with_patterns(documents, entity_threshold)
    
    def _get_ner_description(self, entity_type: str) -> str:
        """Get description for NER entity type."""
        descriptions = {
            'PERSON': 'Person names',
            'ORG': 'Organizations and companies',
            'GPE': 'Geographic locations (cities, countries)',
            'DATE': 'Dates and time periods',
            'MONEY': 'Monetary values',
            'PRODUCT': 'Products and services',
            'EVENT': 'Named events',
            'WORK_OF_ART': 'Titles of works (books, songs, etc.)',
            'LAW': 'Legal documents and laws',
            'LANGUAGE': 'Languages',
            'FAC': 'Facilities and buildings',
            'LOC': 'Locations',
            'NORP': 'Nationalities, religious, or political groups'
        }
        return descriptions.get(entity_type, f'{entity_type} entities')
    
    def _analyze_with_patterns(
        self,
        documents: List[Dict[str, Any]],
        entity_threshold: int
    ) -> tuple:
        """
        Analyze documents using pattern matching (fallback without NLP).
        
        Extracts:
        - Capitalized phrases (likely proper nouns)
        - Dates
        - Numbers with units
        - Email addresses
        - URLs
        """
        import re
        
        print("Using pattern matching for entity extraction...")
        
        # Define patterns
        patterns = {
            'Person': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # John Doe
            'Organization': r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
            'Date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            'Email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'Phone': r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'Money': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            'Percentage': r'\d+(?:\.\d+)?%',
            'URL': r'https?://[^\s]+',
        }
        
        # Extract entities
        entity_mentions = {entity_type: Counter() for entity_type in patterns}
        entity_examples = {entity_type: [] for entity_type in patterns}
        
        for doc_info in documents:
            text = doc_info['text']
            
            for entity_type, pattern in patterns.items():
                matches = re.findall(pattern, text)
                
                for match in matches:
                    entity_mentions[entity_type][match] += 1
                    
                    if len(entity_examples[entity_type]) < 10:
                        entity_examples[entity_type].append(match)
        
        # Build entity definitions
        entities = {}
        for entity_type, mentions in entity_mentions.items():
            total_mentions = sum(mentions.values())
            
            if total_mentions >= entity_threshold:
                entities[entity_type] = {
                    "name": entity_type,
                    "description": f"{entity_type} entities extracted from documents",
                    "properties": {
                        "value": {
                            "type": "string",
                            "required": True,
                            "description": f"{entity_type} value"
                        },
                        "mention_count": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of times mentioned"
                        }
                    },
                    "searchable_fields": ["value"],
                    "display_field": "value",
                    "examples": entity_examples[entity_type][:5]
                }
        
        # Simple relationships (documents mentioning entities)
        relationships = [
            {
                "name": "document_mentions_entity",
                "source_entity": "Document",
                "target_entity": "Entity",
                "type": "mentions",
                "description": "Document mentions an entity"
            }
        ]
        
        # Add Document entity
        entities['Document'] = {
            "name": "Document",
            "description": "PDF document",
            "properties": {
                "path": {
                    "type": "string",
                    "required": True,
                    "description": "File path"
                },
                "content": {
                    "type": "string",
                    "required": True,
                    "description": "Document text content"
                }
            },
            "searchable_fields": ["content"],
            "display_field": "path"
        }
        
        return entities, relationships
    
    def _infer_relationships_from_cooccurrence(
        self,
        documents: List[Dict[str, Any]],
        entity_types: List[str],
        nlp
    ) -> List[Dict[str, Any]]:
        """Infer relationships based on entity co-occurrence in sentences."""
        relationships = []
        relationship_set = set()
        
        # Common relationship verbs
        relation_verbs = [
            'works for', 'employed by', 'founded', 'manages', 'owns',
            'acquired', 'merged with', 'partnered with', 'invested in',
            'located in', 'based in', 'operates in'
        ]
        
        for doc_info in documents[:10]:  # Sample first 10 docs
            text = doc_info['text']
            doc = nlp(text[:100000])  # Limit size
            
            # Check sentences for entity pairs with verbs
            for sent in doc.sents:
                entities_in_sent = [(ent.text, ent.label_) for ent in sent.ents]
                
                if len(entities_in_sent) >= 2:
                    # Check for relationship verbs
                    sent_text = sent.text.lower()
                    
                    for i, (ent1, type1) in enumerate(entities_in_sent):
                        for ent2, type2 in entities_in_sent[i+1:]:
                            if type1 != type2:
                                # Check if relationship verb present
                                for verb in relation_verbs:
                                    if verb in sent_text:
                                        rel_key = f"{type1}-{verb}-{type2}"
                                        if rel_key not in relationship_set:
                                            relationships.append({
                                                "name": f"{type1.lower()}_{verb.replace(' ', '_')}_{type2.lower()}",
                                                "source_entity": type1,
                                                "target_entity": type2,
                                                "type": verb.replace(' ', '_'),
                                                "description": f"{type1} {verb} {type2}"
                                            })
                                            relationship_set.add(rel_key)
        
        # Add generic co-occurrence relationship if none found
        if not relationships:
            relationships = [
                {
                    "name": "co_occurs_with",
                    "source_entity": "Entity",
                    "target_entity": "Entity",
                    "type": "co_occurrence",
                    "description": "Entities mentioned together in documents"
                }
            ]
        
        return relationships


# Convenience function
def analyze_pdfs(
    source: Any,
    output_path: Optional[str] = None,
    auto_save: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to analyze PDFs and generate knowledge graph schema.
    
    Args:
        source: PDF file path, directory, or list of paths
        output_path: Output path (optional, defaults to schemas/knowledge_graph_schema.json)
        auto_save: Automatically save to schemas/ directory (default: True)
        **kwargs: Additional analysis options
    
    Usage:
        # Generate and auto-save to schemas/
        schema = analyze_pdfs('document.pdf')
        
        # Directory with NER
        schema = analyze_pdfs('pdfs/', recursive=True, use_ner=True)
        
        # Generate and save to custom location
        schema = analyze_pdfs('pdfs/', output_path='custom/kg_schema.json')
        
        # Generate without saving
        schema = analyze_pdfs('pdfs/', auto_save=False)
    """
    analyzer = PDFAnalyzer()
    schema = analyzer.generate(source, **kwargs)
    
    if auto_save or output_path:
        saved_path = analyzer.save_schema(schema, output_path)
        schema['_saved_path'] = saved_path
    
    return schema
