"""
JSON Structure Analyzer

Infers knowledge graph schemas from JSON files by analyzing structure and patterns.
"""

from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import json
from .base import SchemaGenerator


class JSONAnalyzer(SchemaGenerator):
    """
    Generate knowledge graph schema by analyzing JSON file structure.
    
    Usage:
        analyzer = JSONAnalyzer()
        
        # Single file
        schema = analyzer.generate('data.json')
        
        # Directory of files
        schema = analyzer.generate('data/', recursive=True)
        
        # List of files
        schema = analyzer.generate(['file1.json', 'file2.json'])
        
        # Save to file
        analyzer.save_schema(schema, 'knowledge_graph_schema.json')
    """
    
    def get_schema_type(self) -> str:
        return 'knowledge_graph'
    
    def generate(self, source: Any, **kwargs) -> Dict[str, Any]:
        """
        Generate knowledge graph schema from JSON files.
        
        Args:
            source: File path, directory, or list of paths
            **kwargs: Options:
                - recursive: Scan directories recursively (default: False)
                - entity_threshold: Min occurrences to consider as entity type (default: 2)
                - sample_limit: Max files to analyze (default: 100)
        
        Returns:
            Knowledge graph schema dict
        """
        # Load JSON files
        json_files = self._collect_json_files(source, **kwargs)
        
        if not json_files:
            raise ValueError("No JSON files found")
        
        # Limit samples
        sample_limit = kwargs.get('sample_limit', 100)
        if len(json_files) > sample_limit:
            print(f"⚠️  Analyzing first {sample_limit} of {len(json_files)} files")
            json_files = json_files[:sample_limit]
        
        # Parse all files
        all_data = []
        for file_path in json_files:
            try:
                data = self.load_json_file(file_path)
                all_data.append(data)
            except Exception as e:
                print(f"⚠️  Warning: Could not load {file_path}: {e}")
        
        if not all_data:
            raise ValueError("Could not load any JSON files")
        
        # Analyze structure
        entity_threshold = kwargs.get('entity_threshold', 2)
        entities, relationships = self._analyze_structure(all_data, entity_threshold)
        
        # Build schema
        schema = {
            "type": "knowledge_graph",
            "version": "1.0.0",
            "metadata": {
                "generated_from": "json_structure_analysis",
                "files_analyzed": len(all_data),
                "entity_types": len(entities),
                "relationship_types": len(relationships)
            },
            "entities": entities,
            "relationships": relationships
        }
        
        return schema
    
    def _collect_json_files(self, source: Any, **kwargs) -> List[str]:
        """Collect JSON file paths from source."""
        recursive = kwargs.get('recursive', False)
        
        if isinstance(source, str):
            path = Path(source)
            
            if path.is_file():
                return [str(path)]
            elif path.is_dir():
                if recursive:
                    return [str(f) for f in path.rglob('*.json')]
                else:
                    return [str(f) for f in path.glob('*.json')]
            else:
                raise ValueError(f"Path does not exist: {source}")
        
        elif isinstance(source, list):
            # List of file paths
            return [str(Path(f)) for f in source if Path(f).exists()]
        
        else:
            raise ValueError("Source must be file path, directory, or list of paths")
    
    def _analyze_structure(
        self,
        data_list: List[Any],
        entity_threshold: int
    ) -> tuple:
        """
        Analyze JSON structure to infer entities and relationships.
        
        Strategy:
        1. Identify object types by shared properties
        2. Detect nested objects as related entities
        3. Infer relationships from object references
        """
        # Collect all object structures
        structures = []
        
        for data in data_list:
            if isinstance(data, dict):
                structures.extend(self._extract_objects(data))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        structures.extend(self._extract_objects(item))
        
        # Group by similar structure (entity types)
        entity_groups = self._group_by_structure(structures, entity_threshold)
        
        # Build entity definitions
        entities = {}
        for entity_type, objects in entity_groups.items():
            entities[entity_type] = self._build_entity_definition(entity_type, objects)
        
        # Infer relationships
        relationships = self._infer_relationships(entity_groups)
        
        return entities, relationships
    
    def _extract_objects(self, data: Dict[str, Any], path: str = '') -> List[Dict[str, Any]]:
        """Recursively extract all objects from JSON structure."""
        objects = []
        
        if not isinstance(data, dict):
            return objects
        
        # Add current object
        objects.append({
            'data': data,
            'path': path,
            'keys': set(data.keys())
        })
        
        # Recursively extract nested objects
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                objects.extend(self._extract_objects(value, new_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        objects.extend(self._extract_objects(item, f"{new_path}[{i}]"))
        
        return objects
    
    def _group_by_structure(
        self,
        objects: List[Dict[str, Any]],
        threshold: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group objects by similar structure (shared keys)."""
        groups = {}
        
        for obj in objects:
            # Find best matching group
            best_match = None
            best_score = 0
            
            for group_name, group_objects in groups.items():
                # Calculate similarity (Jaccard index)
                if not group_objects:
                    continue
                
                reference_keys = group_objects[0]['keys']
                current_keys = obj['keys']
                
                intersection = len(reference_keys & current_keys)
                union = len(reference_keys | current_keys)
                
                if union > 0:
                    score = intersection / union
                    
                    if score > best_score and score > 0.5:  # 50% similarity threshold
                        best_match = group_name
                        best_score = score
            
            if best_match:
                groups[best_match].append(obj)
            else:
                # Create new group
                group_name = self._infer_entity_name(obj)
                groups[group_name] = [obj]
        
        # Filter groups by threshold
        filtered_groups = {
            name: objects
            for name, objects in groups.items()
            if len(objects) >= threshold
        }
        
        return filtered_groups
    
    def _infer_entity_name(self, obj: Dict[str, Any]) -> str:
        """Infer entity type name from object structure."""
        data = obj['data']
        path = obj['path']
        
        # Try to infer from path
        if path:
            parts = path.split('.')
            # Look for plural nouns (likely entity collections)
            for part in reversed(parts):
                if part.endswith('s') and len(part) > 2:
                    # Singularize
                    return part[:-1].title()
        
        # Try to infer from keys
        keys = obj['keys']
        
        # Common entity indicators
        if 'id' in keys or '_id' in keys:
            # Look for type hints in keys
            for key in keys:
                if key.endswith('_type') or key == 'type':
                    type_value = data.get(key)
                    if isinstance(type_value, str):
                        return type_value.title()
            
            # Look for name/title
            if 'name' in keys:
                return 'Entity'
            if 'title' in keys:
                return 'Document'
        
        return 'Unknown'
    
    def _build_entity_definition(
        self,
        entity_type: str,
        objects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build entity definition from object group."""
        # Collect all fields
        all_fields = {}
        
        for obj in objects:
            data = obj['data']
            
            for key, value in data.items():
                if key not in all_fields:
                    all_fields[key] = {
                        'types': set(),
                        'count': 0,
                        'examples': []
                    }
                
                all_fields[key]['count'] += 1
                all_fields[key]['types'].add(self._infer_type(value))
                
                if len(all_fields[key]['examples']) < 3:
                    all_fields[key]['examples'].append(value)
        
        # Build properties
        total_objects = len(objects)
        properties = {}
        
        for field_name, field_info in all_fields.items():
            # Determine primary type
            types = list(field_info['types'])
            primary_type = types[0] if types else 'string'
            
            # Determine if required (appears in >80% of objects)
            required = field_info['count'] / total_objects > 0.8
            
            properties[field_name] = {
                "type": primary_type,
                "required": required,
                "description": f"{field_name} field"
            }
        
        return {
            "name": entity_type,
            "description": f"{entity_type} entity type",
            "properties": properties,
            "searchable_fields": self._identify_searchable_fields(properties),
            "display_field": self._identify_display_field(properties)
        }
    
    def _infer_type(self, value: Any) -> str:
        """Infer schema type from value."""
        if isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, int):
            return 'integer'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, str):
            return 'string'
        elif isinstance(value, list):
            return 'array'
        elif isinstance(value, dict):
            return 'object'
        elif value is None:
            return 'string'
        else:
            return 'string'
    
    def _identify_searchable_fields(self, properties: Dict[str, Any]) -> List[str]:
        """Identify fields that should be searchable."""
        searchable = []
        
        for field_name, field_info in properties.items():
            field_type = field_info['type']
            
            # String fields are searchable
            if field_type == 'string':
                searchable.append(field_name)
            
            # Specific field names
            if field_name.lower() in ['name', 'title', 'description', 'content', 'text']:
                if field_name not in searchable:
                    searchable.append(field_name)
        
        return searchable
    
    def _identify_display_field(self, properties: Dict[str, Any]) -> Optional[str]:
        """Identify primary display field."""
        # Priority order
        priority = ['name', 'title', 'label', 'display_name', 'id']
        
        for field in priority:
            if field in properties:
                return field
        
        # Return first string field
        for field_name, field_info in properties.items():
            if field_info['type'] == 'string':
                return field_name
        
        return None
    
    def _infer_relationships(
        self,
        entity_groups: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Infer relationships between entity types."""
        relationships = []
        relationship_set = set()  # Track unique relationships
        
        entity_types = list(entity_groups.keys())
        
        for entity_type, objects in entity_groups.items():
            for obj in objects:
                data = obj['data']
                
                # Look for foreign key patterns
                for key, value in data.items():
                    # Pattern 1: *_id fields
                    if key.endswith('_id') and key != 'id' and key != '_id':
                        referenced_entity = key[:-3].title()  # Remove '_id'
                        
                        # Check if referenced entity exists
                        if referenced_entity in entity_types:
                            rel_key = f"{entity_type}->{referenced_entity}"
                            if rel_key not in relationship_set:
                                relationships.append({
                                    "name": f"{entity_type.lower()}_to_{referenced_entity.lower()}",
                                    "source_entity": entity_type,
                                    "target_entity": referenced_entity,
                                    "type": "has_one",
                                    "description": f"{entity_type} references {referenced_entity}"
                                })
                                relationship_set.add(rel_key)
                    
                    # Pattern 2: Nested objects
                    elif isinstance(value, dict):
                        # Check if it matches an entity type
                        value_keys = set(value.keys())
                        
                        for other_type, other_objects in entity_groups.items():
                            if other_type == entity_type:
                                continue
                            
                            if other_objects:
                                other_keys = other_objects[0]['keys']
                                
                                # Calculate similarity
                                intersection = len(value_keys & other_keys)
                                union = len(value_keys | other_keys)
                                
                                if union > 0 and intersection / union > 0.5:
                                    rel_key = f"{entity_type}->{other_type}"
                                    if rel_key not in relationship_set:
                                        relationships.append({
                                            "name": f"{entity_type.lower()}_has_{other_type.lower()}",
                                            "source_entity": entity_type,
                                            "target_entity": other_type,
                                            "type": "has_one",
                                            "description": f"{entity_type} contains {other_type}"
                                        })
                                        relationship_set.add(rel_key)
                    
                    # Pattern 3: Arrays of objects
                    elif isinstance(value, list) and value and isinstance(value[0], dict):
                        item_keys = set(value[0].keys())
                        
                        for other_type, other_objects in entity_groups.items():
                            if other_type == entity_type:
                                continue
                            
                            if other_objects:
                                other_keys = other_objects[0]['keys']
                                
                                # Calculate similarity
                                intersection = len(item_keys & other_keys)
                                union = len(item_keys | other_keys)
                                
                                if union > 0 and intersection / union > 0.5:
                                    rel_key = f"{entity_type}->>{other_type}"
                                    if rel_key not in relationship_set:
                                        relationships.append({
                                            "name": f"{entity_type.lower()}_has_many_{other_type.lower()}",
                                            "source_entity": entity_type,
                                            "target_entity": other_type,
                                            "type": "has_many",
                                            "description": f"{entity_type} contains multiple {other_type}"
                                        })
                                        relationship_set.add(rel_key)
        
        return relationships


# Convenience function
def analyze_json(
    source: Any,
    output_path: Optional[str] = None,
    auto_save: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to analyze JSON files and generate knowledge graph schema.
    
    Args:
        source: JSON file path, directory, or list of paths
        output_path: Output path (optional, defaults to schemas/knowledge_graph_schema.json)
        auto_save: Automatically save to schemas/ directory (default: True)
        **kwargs: Additional analysis options
    
    Usage:
        # Generate and auto-save to schemas/
        schema = analyze_json('data.json')
        
        # Directory
        schema = analyze_json('data/', recursive=True)
        
        # Generate and save to custom location
        schema = analyze_json('data/', output_path='custom/kg_schema.json')
        
        # Generate without saving
        schema = analyze_json('data/', auto_save=False)
    """
    analyzer = JSONAnalyzer()
    schema = analyzer.generate(source, **kwargs)
    
    if auto_save or output_path:
        saved_path = analyzer.save_schema(schema, output_path)
        schema['_saved_path'] = saved_path
    
    return schema
