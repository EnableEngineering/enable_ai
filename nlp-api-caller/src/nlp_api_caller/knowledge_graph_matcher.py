"""
Knowledge Graph RAG Planner - For unstructured data search.

Supports:
- READ ONLY operations
- Vector search using embeddings
- RAG (Retrieval Augmented Generation)
"""

from typing import Dict, Any, Optional, List


class KnowledgeGraphMatcher:
    """
    Builds RAG query plans for knowledge graph search.
    
    For PDFs, documents, unstructured data.
    """
    
    def __init__(self):
        """Initialize knowledge graph matcher."""
        pass
    
    def build_rag_query(self, parsed: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build RAG query plan from parsed input.
        
        Args:
            parsed: {
                'intent': 'read',
                'resource': 'Person',
                'entities': {
                    'relationship': 'WORKS_AT',
                    'from_entity': 'Person',
                    'to_entity': 'Organization'
                }
            }
            schema: knowledge_graph schema
        
        Returns:
            {
                'entity_type': 'Person',
                'filters': {...},
                'relationships': ['WORKS_AT'],
                'vector_search_params': {...}
            }
        """
        intent = parsed.get('intent')
        resource = parsed.get('resource')  # Entity type
        entities = parsed.get('entities', {})
        
        # Only support read operations
        if intent not in ['read', 'search']:
            return {
                'error': 'Knowledge graph only supports READ/SEARCH operations'
            }
        
        if not resource or resource not in schema.get('entities', {}):
            # If no resource found, use generic search
            return self._build_generic_search(parsed, schema)
        
        # Build RAG query plan
        entity_def = schema['entities'][resource]
        
        # Extract filters (entity attributes)
        filters = {}
        for attr_name in entity_def.get('attributes', {}).keys():
            if attr_name in entities:
                filters[attr_name] = entities[attr_name]
        
        # Extract relationships
        relationships = []
        if 'relationship' in entities:
            relationships.append(entities['relationship'])
        
        # Extract related entities
        related_entities = {}
        if 'to_entity' in entities:
            related_entities['to'] = entities['to_entity']
        if 'from_entity' in entities:
            related_entities['from'] = entities['from_entity']
        
        return {
            'entity_type': resource,
            'filters': filters,
            'relationships': relationships,
            'related_entities': related_entities,
            'vector_search_params': {
                'top_k': 5,  # Return top 5 results
                'similarity_threshold': 0.7  # Minimum similarity score
            }
        }
    
    def _build_generic_search(self, parsed: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build generic search when no specific entity is identified.
        
        Uses full-text search across all entities.
        """
        return {
            'entity_type': None,
            'filters': {},
            'relationships': [],
            'related_entities': {},
            'vector_search_params': {
                'top_k': 10,  # Return more results for generic search
                'similarity_threshold': 0.6  # Lower threshold for broader results
            },
            'search_mode': 'generic'
        }
