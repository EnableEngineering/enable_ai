"""
Database Query Builder - Converts parsed NL to SQL/NoSQL queries.

Supports:
- SQL databases (PostgreSQL, MySQL, SQLite)
- NoSQL databases (MongoDB - JSON-like syntax)
"""

from typing import Dict, Any, Optional, List


class DatabaseMatcher:
    """
    Builds database queries from parsed natural language.
    
    Converts intent + entities → SQL/NoSQL queries
    """
    
    def __init__(self):
        """Initialize database matcher."""
        self.intent_to_operation = {
            'read': 'SELECT',
            'create': 'INSERT',
            'update': 'UPDATE',
            'delete': 'DELETE'
        }
    
    def build_query(self, parsed: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build database query from parsed input.
        
        Args:
            parsed: {
                'intent': 'read',
                'resource': 'employees',
                'entities': {'department': 'engineering', 'salary': 50000}
            }
            schema: database_schema
        
        Returns:
            {
                'table': 'employees',
                'operation': 'SELECT',
                'filters': {...},
                'sql_query': 'SELECT * FROM employees WHERE ...',
                'params': [...]
            }
        """
        intent = parsed.get('intent')
        resource = parsed.get('resource')  # Table name
        entities = parsed.get('entities', {})
        
        if not resource or resource not in schema.get('tables', {}):
            return None
        
        # Get table definition
        table_def = schema['tables'][resource]
        
        # Route to appropriate query builder
        if intent == 'read':
            return self._build_select_query(resource, entities, table_def)
        elif intent == 'create':
            return self._build_insert_query(resource, entities, table_def)
        elif intent == 'update':
            return self._build_update_query(resource, entities, table_def)
        elif intent == 'delete':
            return self._build_delete_query(resource, entities, table_def)
        else:
            return None
    
    # ========================================================================
    # SELECT QUERY BUILDER
    # ========================================================================
    
    def _build_select_query(
        self,
        table: str,
        filters: Dict[str, Any],
        table_def: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build SELECT query.
        
        Example:
            Input: table='employees', filters={'department': 'engineering', 'salary': 50000}
            Output: {
                'sql_query': 'SELECT * FROM employees WHERE department = ? AND salary >= ?',
                'params': ['engineering', 50000]
            }
        """
        columns = table_def.get('columns', {})
        
        # Build WHERE clause
        where_clauses = []
        params = []
        
        for col_name, col_value in filters.items():
            if col_name not in columns:
                continue
            
            # Check if column has comparison operator in value
            if isinstance(col_value, dict):
                # MongoDB-style operators: {'$gte': 50000}
                for op, val in col_value.items():
                    sql_op = self._convert_operator(op)
                    where_clauses.append(f"{col_name} {sql_op} ?")
                    params.append(val)
            else:
                # Simple equality
                where_clauses.append(f"{col_name} = ?")
                params.append(col_value)
        
        # Build SQL query
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        sql_query = f"SELECT * FROM {table} WHERE {where_clause}"
        
        return {
            'table': table,
            'operation': 'SELECT',
            'filters': filters,
            'sql_query': sql_query,
            'params': params
        }
    
    # ========================================================================
    # INSERT QUERY BUILDER
    # ========================================================================
    
    def _build_insert_query(
        self,
        table: str,
        data: Dict[str, Any],
        table_def: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build INSERT query.
        
        Example:
            Input: table='employees', data={'name': 'John', 'department': 'engineering'}
            Output: {
                'sql_query': 'INSERT INTO employees (name, department) VALUES (?, ?)',
                'params': ['John', 'engineering']
            }
        """
        columns = table_def.get('columns', {})
        
        # Filter only valid columns
        valid_data = {k: v for k, v in data.items() if k in columns}
        
        if not valid_data:
            return None
        
        # Build INSERT query
        col_names = ", ".join(valid_data.keys())
        placeholders = ", ".join(["?" for _ in valid_data])
        params = list(valid_data.values())
        
        sql_query = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"
        
        return {
            'table': table,
            'operation': 'INSERT',
            'data': valid_data,
            'sql_query': sql_query,
            'params': params
        }
    
    # ========================================================================
    # UPDATE QUERY BUILDER
    # ========================================================================
    
    def _build_update_query(
        self,
        table: str,
        entities: Dict[str, Any],
        table_def: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build UPDATE query.
        
        Assumes entities contain:
        - Identifier (id, primary key)
        - Fields to update
        
        Example:
            Input: table='employees', entities={'id': 5, 'salary': 60000, 'department': 'management'}
            Output: {
                'sql_query': 'UPDATE employees SET salary = ?, department = ? WHERE id = ?',
                'params': [60000, 'management', 5]
            }
        """
        columns = table_def.get('columns', {})
        
        # Find primary key
        primary_key = None
        for col_name, col_def in columns.items():
            if col_def.get('primary_key'):
                primary_key = col_name
                break
        
        if not primary_key or primary_key not in entities:
            return None
        
        # Separate identifier from update fields
        identifier_value = entities[primary_key]
        update_fields = {k: v for k, v in entities.items() if k != primary_key and k in columns}
        
        if not update_fields:
            return None
        
        # Build UPDATE query
        set_clauses = [f"{col} = ?" for col in update_fields.keys()]
        set_clause = ", ".join(set_clauses)
        params = list(update_fields.values()) + [identifier_value]
        
        sql_query = f"UPDATE {table} SET {set_clause} WHERE {primary_key} = ?"
        
        return {
            'table': table,
            'operation': 'UPDATE',
            'filters': {primary_key: identifier_value},
            'data': update_fields,
            'sql_query': sql_query,
            'params': params
        }
    
    # ========================================================================
    # DELETE QUERY BUILDER
    # ========================================================================
    
    def _build_delete_query(
        self,
        table: str,
        filters: Dict[str, Any],
        table_def: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build DELETE query.
        
        Example:
            Input: table='employees', filters={'id': 5}
            Output: {
                'sql_query': 'DELETE FROM employees WHERE id = ?',
                'params': [5]
            }
        """
        columns = table_def.get('columns', {})
        
        # Build WHERE clause (same as SELECT)
        where_clauses = []
        params = []
        
        for col_name, col_value in filters.items():
            if col_name not in columns:
                continue
            
            where_clauses.append(f"{col_name} = ?")
            params.append(col_value)
        
        if not where_clauses:
            # Don't allow DELETE without WHERE clause (safety)
            return None
        
        where_clause = " AND ".join(where_clauses)
        sql_query = f"DELETE FROM {table} WHERE {where_clause}"
        
        return {
            'table': table,
            'operation': 'DELETE',
            'filters': filters,
            'sql_query': sql_query,
            'params': params
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _convert_operator(self, mongo_op: str) -> str:
        """
        Convert MongoDB-style operators to SQL.
        
        Args:
            mongo_op: MongoDB operator (e.g., '$gte', '$lt')
        
        Returns:
            SQL operator (e.g., '>=', '<')
        """
        operator_map = {
            '$eq': '=',
            '$ne': '!=',
            '$gt': '>',
            '$gte': '>=',
            '$lt': '<',
            '$lte': '<=',
            '$in': 'IN',
            '$nin': 'NOT IN',
            '$like': 'LIKE'
        }
        return operator_map.get(mongo_op, '=')
