"""
Intent Classifier - Fast rule-based intent classification without LLM.

This module provides deterministic, low-latency classification for obvious queries,
reducing LLM API calls and costs. Falls back to LLM parsing for ambiguous queries.

Features:
- Rule-based intent detection (list/create/update/delete/count)
- Resource matching via schema resource names and synonyms
- Entity extraction for obvious patterns (IDs, numbers)
- Confidence scoring to decide when to skip LLM

Domain-agnostic: works with any schema, no hardcoded entity names.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from .types import ClassificationResult
from .utils import setup_logger
from . import constants


class IntentClassifier:
    """
    Fast rule-based intent classifier.

    Use this BEFORE the LLM-based QueryParser to:
    1. Handle obvious queries without LLM ($0 cost, <10ms latency)
    2. Pre-populate context for ambiguous queries (improves LLM accuracy)
    3. Detect when to skip LLM entirely (high confidence)

    Integration:
        classifier = IntentClassifier(schema)
        result, confidence = classifier.classify(query)
        if confidence >= CLASSIFIER_MEDIUM_CONFIDENCE:
            # Skip LLM, use result directly
        else:
            # Use LLM parser with result as hint
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize classifier with API schema.

        Args:
            schema: Enable AI format schema with 'resources' and 'resource_hints'
        """
        self.schema = schema
        self.resources = list(schema.get('resources', {}).keys())
        self.resource_hints = schema.get('resource_hints', {}) or {}
        self.logger = setup_logger('enable_ai.classifier')

        # Build resource name variants for matching
        self._resource_variants = self._build_resource_variants()

        self.logger.info(
            f"IntentClassifier initialized with {len(self.resources)} resources"
        )

    def classify(self, query: str) -> Tuple[Optional[ClassificationResult], float]:
        """
        Classify query intent using rules (no LLM).

        Args:
            query: Natural language query string

        Returns:
            Tuple of (ClassificationResult or None, confidence float)
            - If confidence >= CLASSIFIER_MEDIUM_CONFIDENCE, result can be used directly
            - If confidence < CLASSIFIER_LOW_CONFIDENCE, defer to LLM
        """
        if not query or not query.strip():
            return None, 0.0

        query_lower = query.lower().strip()
        tokens = self._tokenize(query_lower)

        # Detect intent from keywords
        intent, intent_confidence = self._detect_intent(query_lower, tokens)

        # Detect question type (list vs count vs details)
        question_type = self._detect_question_type(query_lower)

        # Detect resource
        resource, resource_confidence = self._detect_resource(query_lower, tokens)

        # Extract obvious entities (IDs, numbers)
        entities = self._extract_entities(query_lower, tokens)
        filters = self._entities_to_filters(entities)

        # Calculate overall confidence
        confidence = self._calculate_confidence(
            intent, intent_confidence,
            resource, resource_confidence,
            entities
        )

        if confidence < constants.CLASSIFIER_LOW_CONFIDENCE:
            # Too uncertain - defer to LLM
            return None, confidence

        result = ClassificationResult(
            intent=intent or 'read',
            resource=resource,
            entities=entities,
            filters=filters,
            confidence=confidence,
            question_type=question_type,
            classified_by='rules',
        )

        self.logger.info(
            f"Classified: intent={result.intent}, resource={result.resource}, "
            f"confidence={confidence:.2f}"
        )

        return result, confidence

    def _detect_intent(self, query: str, tokens: List[str]) -> Tuple[Optional[str], float]:
        """
        Detect CRUD intent from keywords.

        Returns:
            (intent string, confidence)
        """
        # Check first few words for intent keywords
        first_tokens = set(tokens[:4])

        # Read intent (most common)
        if first_tokens & constants.CLASSIFIER_READ_KEYWORDS:
            return 'read', 0.9

        # Create intent
        if first_tokens & constants.CLASSIFIER_CREATE_KEYWORDS:
            return 'create', 0.9

        # Update intent
        if first_tokens & constants.CLASSIFIER_UPDATE_KEYWORDS:
            return 'update', 0.9

        # Delete intent
        if first_tokens & constants.CLASSIFIER_DELETE_KEYWORDS:
            return 'delete', 0.9

        # Count check (can appear anywhere)
        for count_phrase in constants.CLASSIFIER_COUNT_KEYWORDS:
            if count_phrase in query:
                return 'read', 0.85  # Count is still a read intent

        # Default to read for query-like sentences
        if '?' in query or any(w in first_tokens for w in ['what', 'which', 'who', 'where']):
            return 'read', 0.7

        return 'read', 0.5  # Default with low confidence

    def _detect_question_type(self, query: str) -> str:
        """
        Detect what type of response the user wants.

        Returns:
            'count', 'list', or 'details'
        """
        # Count queries
        for count_phrase in constants.CLASSIFIER_COUNT_KEYWORDS:
            if count_phrase in query:
                return 'count'

        # Detail queries
        detail_patterns = ['details', 'detail', 'info about', 'information about', 'tell me about']
        for pattern in detail_patterns:
            if pattern in query:
                return 'details'

        # Default to list
        return 'list'

    def _detect_resource(self, query: str, tokens: List[str]) -> Tuple[Optional[str], float]:
        """
        Detect resource from known resource names and synonyms.

        Returns:
            (resource name or None, confidence)
        """
        # Try exact matches first (highest priority)
        for resource, variants in self._resource_variants.items():
            for variant in variants:
                # Check if variant appears as a whole word
                if self._word_in_query(variant, query):
                    return resource, 0.95

        # Try partial matches
        for resource, variants in self._resource_variants.items():
            for variant in variants:
                if len(variant) >= 4 and variant in query:
                    return resource, 0.8

        return None, 0.0

    def _extract_entities(self, query: str, tokens: List[str]) -> Dict[str, Any]:
        """
        Extract obvious entities like IDs and numbers.

        Patterns detected:
        - "id 5", "id=5", "#5" -> {"id": 5}
        - "user 123" -> {"id": 123} (if after resource word)
        - Numbers in specific contexts
        """
        entities = {}

        # Pattern: "id X" or "id=X" or "id: X"
        id_patterns = [
            r'\bid\s*[=:]\s*(\d+)',  # id=5, id: 5
            r'\bid\s+(\d+)',          # id 5
            r'#(\d+)',                # #5
        ]

        for pattern in id_patterns:
            match = re.search(pattern, query)
            if match:
                entities['id'] = int(match.group(1))
                break

        # Pattern: "first N", "top N", "last N"
        limit_match = re.search(r'\b(?:first|top|last|next)\s+(\d+)', query)
        if limit_match:
            entities['_limit'] = int(limit_match.group(1))

        # Pattern: explicit numbers after certain words
        # "show me 10 users" -> limit 10
        show_num_match = re.search(r'\b(?:show|list|get|display)\s+(?:me\s+)?(\d+)\s+', query)
        if show_num_match and '_limit' not in entities:
            entities['_limit'] = int(show_num_match.group(1))

        return entities

    def _entities_to_filters(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert extracted entities to filter format.

        Args:
            entities: Dict of extracted entities

        Returns:
            Dict of filters in {field: {operator, value}} format
        """
        filters = {}

        for key, value in entities.items():
            # Skip internal keys
            if key.startswith('_'):
                continue

            filters[key] = {
                'operator': 'equals',
                'value': value
            }

        return filters

    def _calculate_confidence(
        self,
        intent: Optional[str],
        intent_confidence: float,
        resource: Optional[str],
        resource_confidence: float,
        entities: Dict[str, Any]
    ) -> float:
        """
        Calculate overall classification confidence.

        Factors:
        - Intent detection confidence
        - Resource detection confidence
        - Presence of specific entities (boosts confidence)
        """
        if not intent or not resource:
            # Can't classify without both
            if not resource:
                return intent_confidence * 0.5  # Halve if no resource
            return intent_confidence * 0.7

        # Weighted average
        base_confidence = (intent_confidence * 0.4 + resource_confidence * 0.6)

        # Boost for specific entities
        if entities.get('id'):
            base_confidence = min(base_confidence + 0.1, 1.0)

        return base_confidence

    def _build_resource_variants(self) -> Dict[str, set]:
        """
        Build all name variants for each resource.

        Variants include:
        - Original name
        - Singular/plural forms
        - Underscore/hyphen variants
        - Synonyms from resource_hints
        """
        variants = {}

        for resource in self.resources:
            res_variants = set()

            # Add original and case variants
            res_variants.add(resource.lower())

            # Add underscore/hyphen/space variants
            normalized = resource.lower()
            res_variants.add(normalized.replace('_', ' '))
            res_variants.add(normalized.replace('-', ' '))
            res_variants.add(normalized.replace('_', ''))
            res_variants.add(normalized.replace('-', ''))

            # Add singular form (simple: remove trailing 's')
            if normalized.endswith('s') and len(normalized) > 2:
                singular = normalized[:-1]
                res_variants.add(singular)
                res_variants.add(singular.replace('_', ' '))
                res_variants.add(singular.replace('-', ' '))

            # Add synonyms from resource_hints
            hints = self.resource_hints.get(resource, {})
            if isinstance(hints, dict):
                synonyms = hints.get('__resource_synonyms__', [])
                if isinstance(synonyms, list):
                    for syn in synonyms:
                        if isinstance(syn, str):
                            res_variants.add(syn.lower())

            variants[resource] = res_variants

        return variants

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text (should be lowercase)

        Returns:
            List of word tokens
        """
        # Split on whitespace and punctuation, keep alphanumeric
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _word_in_query(self, word: str, query: str) -> bool:
        """
        Check if word appears as a whole word in query.

        Args:
            word: Word to search for
            query: Query string (should be lowercase)

        Returns:
            True if word appears as whole word
        """
        # Use word boundary regex
        pattern = r'\b' + re.escape(word) + r'\b'
        return bool(re.search(pattern, query))

    def get_available_resources(self) -> List[str]:
        """Get list of all available resource names."""
        return self.resources.copy()

    def get_resource_synonyms(self, resource: str) -> List[str]:
        """
        Get synonyms for a specific resource.

        Args:
            resource: Resource name

        Returns:
            List of synonym strings
        """
        hints = self.resource_hints.get(resource, {})
        if isinstance(hints, dict):
            synonyms = hints.get('__resource_synonyms__', [])
            if isinstance(synonyms, list):
                return synonyms
        return []
