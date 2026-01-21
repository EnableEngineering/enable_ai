"""
Database Introspection Tool

Generates database schemas by connecting to databases and analyzing their structure.
Supports: PostgreSQL, MySQL, SQLite, MongoDB
"""

from typing import Dict, Any, List, Optional, Union
from .base import SchemaGenerator


class DatabaseInspector(SchemaGenerator):
    """
    Generate database schema by introspecting live databases.
    
    Usage:
        inspector = DatabaseInspector()
        
        # PostgreSQL
        schema = inspector.generate('postgresql://user:pass@localhost/mydb')
        
        # MySQL
        schema = inspector.generate('mysql://user:pass@localhost/mydb')
        
        # SQLite
        schema = inspector.generate('sqlite:///path/to/db.sqlite')
        
        # MongoDB
        schema = inspector.generate('mongodb://localhost:27017/mydb')
        
        # Save to file
        inspector.save_schema(schema, 'database_schema.json')
    """
    
    def get_schema_type(self) -> str:
        return 'database'
    
    def generate(self, source: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Generate database schema from connection string or config.
        
        Args:
            source: Connection string (str) or connection config (dict)
            **kwargs: Options:
                - table_filter: List of tables to include (default: all)
                - include_views: Include views (default: False)
                - sample_rows: Number of rows to sample for type inference (default: 100)
        
        Returns:
            Database schema dict
        """
        # Parse connection info
        if isinstance(source, str):
            conn_info = self._parse_connection_string(source)
        elif isinstance(source, dict):
            conn_info = source
        else:
            raise ValueError("Source must be connection string or config dict")
        
        db_type = conn_info['type']
        
        # Route to appropriate inspector
        if db_type in ['postgresql', 'postgres']:
            return self._inspect_postgresql(conn_info, **kwargs)
        elif db_type == 'mysql':
            return self._inspect_mysql(conn_info, **kwargs)
        elif db_type == 'sqlite':
            return self._inspect_sqlite(conn_info, **kwargs)
        elif db_type == 'mongodb':
            return self._inspect_mongodb(conn_info, **kwargs)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def _parse_connection_string(self, conn_str: str) -> Dict[str, Any]:
        """Parse database connection string."""
        # Format: type://user:pass@host:port/database
        # or sqlite:///path/to/db
        
        if conn_str.startswith('sqlite:///'):
            return {
                'type': 'sqlite',
                'database': conn_str.replace('sqlite:///', '')
            }
        
        # Parse standard connection string
        parts = conn_str.split('://')
        if len(parts) != 2:
            raise ValueError("Invalid connection string format")
        
        db_type = parts[0]
        rest = parts[1]
        
        # Split user:pass@host:port/database
        if '@' in rest:
            auth, location = rest.split('@', 1)
            if ':' in auth:
                user, password = auth.split(':', 1)
            else:
                user = auth
                password = ''
        else:
            user = ''
            password = ''
            location = rest
        
        # Split host:port/database
        if '/' in location:
            host_port, database = location.rsplit('/', 1)
        else:
            host_port = location
            database = ''
        
        # Split host:port
        if ':' in host_port:
            host, port = host_port.rsplit(':', 1)
            port = int(port)
        else:
            host = host_port
            port = self._get_default_port(db_type)
        
        return {
            'type': db_type,
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
    
    def _get_default_port(self, db_type: str) -> int:
        """Get default port for database type."""
        ports = {
            'postgresql': 5432,
            'postgres': 5432,
            'mysql': 3306,
            'mongodb': 27017
        }
        return ports.get(db_type, 5432)
    
    def _inspect_postgresql(self, conn_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Introspect PostgreSQL database."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError:
            raise ImportError(
                "PostgreSQL support requires psycopg2-binary. "
                "Install with: pip install psycopg2-binary"
            )
        
        connection = None
        try:
            # Connect
            connection = psycopg2.connect(
                host=conn_info.get('host', 'localhost'),
                port=conn_info.get('port', 5432),
                database=conn_info.get('database'),
                user=conn_info.get('user'),
                password=conn_info.get('password')
            )
            
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            # Get table list
            table_filter = kwargs.get('table_filter')
            include_views = kwargs.get('include_views', False)
            
            if include_views:
                table_types = "('BASE TABLE', 'VIEW')"
            else:
                table_types = "('BASE TABLE')"
            
            cursor.execute(f"""
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type IN {table_types}
                ORDER BY table_name
            """)
            
            tables_info = cursor.fetchall()
            
            # Filter tables
            if table_filter:
                tables_info = [t for t in tables_info if t['table_name'] in table_filter]
            
            # Introspect each table
            tables = {}
            for table_info in tables_info:
                table_name = table_info['table_name']
                table_type = 'view' if table_info['table_type'] == 'VIEW' else 'table'
                
                # Get columns
                cursor.execute("""
                    SELECT 
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    AND table_name = %s
                    ORDER BY ordinal_position
                """, (table_name,))
                
                columns_info = cursor.fetchall()
                
                # Get primary key
                cursor.execute("""
                    SELECT a.attname
                    FROM pg_index i
                    JOIN pg_attribute a ON a.attrelid = i.indrelid
                    AND a.attnum = ANY(i.indkey)
                    WHERE i.indrelid = %s::regclass
                    AND i.indisprimary
                """, (table_name,))
                
                pk_result = cursor.fetchone()
                primary_key = pk_result['attname'] if pk_result else None
                
                # Get foreign keys
                cursor.execute("""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                        AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_name = %s
                """, (table_name,))
                
                fk_results = cursor.fetchall()
                
                # Build table schema
                tables[table_name] = {
                    "name": table_name,
                    "type": table_type,
                    "columns": self._convert_columns(columns_info),
                    "primary_key": primary_key,
                    "foreign_keys": self._convert_foreign_keys(fk_results),
                    "relationships": []  # Will be populated later
                }
            
            # Build relationships
            self._build_relationships(tables)
            
            # Build schema
            schema = {
                "type": "database",
                "version": "1.0.0",
                "metadata": {
                    "database_type": "postgresql",
                    "database_name": conn_info.get('database'),
                    "generated_from": "database_introspection",
                    "table_count": len(tables)
                },
                "tables": tables
            }
            
            return schema
            
        finally:
            if connection:
                connection.close()
    
    def _inspect_mysql(self, conn_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Introspect MySQL database."""
        try:
            import mysql.connector
        except ImportError:
            raise ImportError(
                "MySQL support requires mysql-connector-python. "
                "Install with: pip install mysql-connector-python"
            )
        
        connection = None
        try:
            # Connect
            connection = mysql.connector.connect(
                host=conn_info.get('host', 'localhost'),
                port=conn_info.get('port', 3306),
                database=conn_info.get('database'),
                user=conn_info.get('user'),
                password=conn_info.get('password')
            )
            
            cursor = connection.cursor(dictionary=True)
            
            # Get table list
            table_filter = kwargs.get('table_filter')
            include_views = kwargs.get('include_views', False)
            
            if include_views:
                cursor.execute("SHOW FULL TABLES")
            else:
                cursor.execute("SHOW TABLES")
            
            tables_raw = cursor.fetchall()
            
            # Extract table names
            db_name = conn_info.get('database')
            table_names = []
            
            for row in tables_raw:
                if include_views:
                    table_name = row[f'Tables_in_{db_name}']
                    table_type = row['Table_type']
                    if table_type == 'BASE TABLE' or include_views:
                        table_names.append(table_name)
                else:
                    table_names.append(row[f'Tables_in_{db_name}'])
            
            # Filter tables
            if table_filter:
                table_names = [t for t in table_names if t in table_filter]
            
            # Introspect each table
            tables = {}
            for table_name in table_names:
                # Get columns
                cursor.execute(f"DESCRIBE {table_name}")
                columns_info = cursor.fetchall()
                
                # Get foreign keys
                cursor.execute(f"""
                    SELECT
                        COLUMN_NAME,
                        REFERENCED_TABLE_NAME,
                        REFERENCED_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = %s
                    AND TABLE_NAME = %s
                    AND REFERENCED_TABLE_NAME IS NOT NULL
                """, (db_name, table_name))
                
                fk_results = cursor.fetchall()
                
                # Find primary key
                primary_key = None
                for col in columns_info:
                    if col['Key'] == 'PRI':
                        primary_key = col['Field']
                        break
                
                # Build table schema
                tables[table_name] = {
                    "name": table_name,
                    "type": "table",
                    "columns": self._convert_mysql_columns(columns_info),
                    "primary_key": primary_key,
                    "foreign_keys": self._convert_mysql_foreign_keys(fk_results),
                    "relationships": []
                }
            
            # Build relationships
            self._build_relationships(tables)
            
            # Build schema
            schema = {
                "type": "database",
                "version": "1.0.0",
                "metadata": {
                    "database_type": "mysql",
                    "database_name": db_name,
                    "generated_from": "database_introspection",
                    "table_count": len(tables)
                },
                "tables": tables
            }
            
            return schema
            
        finally:
            if connection:
                connection.close()
    
    def _inspect_sqlite(self, conn_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Introspect SQLite database."""
        import sqlite3
        
        connection = None
        try:
            # Connect
            db_path = conn_info.get('database')
            connection = sqlite3.connect(db_path)
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            
            # Get table list
            table_filter = kwargs.get('table_filter')
            include_views = kwargs.get('include_views', False)
            
            if include_views:
                cursor.execute("""
                    SELECT name, type FROM sqlite_master
                    WHERE type IN ('table', 'view')
                    AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """)
            else:
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type = 'table'
                    AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """)
            
            tables_raw = cursor.fetchall()
            table_names = [row[0] for row in tables_raw]
            
            # Filter tables
            if table_filter:
                table_names = [t for t in table_names if t in table_filter]
            
            # Introspect each table
            tables = {}
            for table_name in table_names:
                # Get columns
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns_info = cursor.fetchall()
                
                # Get foreign keys
                cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                fk_results = cursor.fetchall()
                
                # Find primary key
                primary_key = None
                for col in columns_info:
                    if col['pk'] == 1:
                        primary_key = col['name']
                        break
                
                # Build table schema
                tables[table_name] = {
                    "name": table_name,
                    "type": "table",
                    "columns": self._convert_sqlite_columns(columns_info),
                    "primary_key": primary_key,
                    "foreign_keys": self._convert_sqlite_foreign_keys(fk_results),
                    "relationships": []
                }
            
            # Build relationships
            self._build_relationships(tables)
            
            # Build schema
            schema = {
                "type": "database",
                "version": "1.0.0",
                "metadata": {
                    "database_type": "sqlite",
                    "database_name": db_path,
                    "generated_from": "database_introspection",
                    "table_count": len(tables)
                },
                "tables": tables
            }
            
            return schema
            
        finally:
            if connection:
                connection.close()
    
    def _inspect_mongodb(self, conn_info: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Introspect MongoDB database."""
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "MongoDB support requires pymongo. "
                "Install with: pip install pymongo"
            )
        
        client = None
        try:
            # Connect
            host = conn_info.get('host', 'localhost')
            port = conn_info.get('port', 27017)
            user = conn_info.get('user')
            password = conn_info.get('password')
            
            if user and password:
                connection_string = f"mongodb://{user}:{password}@{host}:{port}/"
            else:
                connection_string = f"mongodb://{host}:{port}/"
            
            client = MongoClient(connection_string)
            db = client[conn_info.get('database')]
            
            # Get collection list
            collection_filter = kwargs.get('table_filter')
            sample_rows = kwargs.get('sample_rows', 100)
            
            collection_names = db.list_collection_names()
            
            # Filter collections
            if collection_filter:
                collection_names = [c for c in collection_names if c in collection_filter]
            
            # Introspect each collection
            tables = {}
            for coll_name in collection_names:
                collection = db[coll_name]
                
                # Sample documents to infer schema
                sample = list(collection.find().limit(sample_rows))
                
                if not sample:
                    continue
                
                # Infer schema from samples
                fields = self._infer_mongodb_schema(sample)
                
                tables[coll_name] = {
                    "name": coll_name,
                    "type": "collection",
                    "columns": fields,
                    "primary_key": "_id",
                    "foreign_keys": [],
                    "relationships": []
                }
            
            # Build schema
            schema = {
                "type": "database",
                "version": "1.0.0",
                "metadata": {
                    "database_type": "mongodb",
                    "database_name": conn_info.get('database'),
                    "generated_from": "database_introspection",
                    "table_count": len(tables),
                    "note": "MongoDB schema inferred from sample documents"
                },
                "tables": tables
            }
            
            return schema
            
        finally:
            if client:
                client.close()
    
    def _convert_columns(self, columns_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert PostgreSQL column info to schema format."""
        columns = []
        for col in columns_info:
            columns.append({
                "name": col['column_name'],
                "type": self._normalize_type(col['data_type']),
                "nullable": col['is_nullable'] == 'YES',
                "default": col['column_default']
            })
        return columns
    
    def _convert_mysql_columns(self, columns_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MySQL column info to schema format."""
        columns = []
        for col in columns_info:
            columns.append({
                "name": col['Field'],
                "type": self._normalize_type(col['Type']),
                "nullable": col['Null'] == 'YES',
                "default": col['Default']
            })
        return columns
    
    def _convert_sqlite_columns(self, columns_info: List) -> List[Dict[str, Any]]:
        """Convert SQLite column info to schema format."""
        columns = []
        for col in columns_info:
            columns.append({
                "name": col['name'],
                "type": self._normalize_type(col['type']),
                "nullable": col['notnull'] == 0,
                "default": col['dflt_value']
            })
        return columns
    
    def _normalize_type(self, db_type: str) -> str:
        """Normalize database-specific types to standard types."""
        db_type_lower = db_type.lower()
        
        # Integer types
        if any(t in db_type_lower for t in ['int', 'serial', 'bigint', 'smallint']):
            return 'integer'
        
        # String types
        if any(t in db_type_lower for t in ['char', 'text', 'varchar', 'string']):
            return 'string'
        
        # Float types
        if any(t in db_type_lower for t in ['float', 'double', 'real', 'numeric', 'decimal']):
            return 'float'
        
        # Boolean
        if 'bool' in db_type_lower:
            return 'boolean'
        
        # Date/Time
        if any(t in db_type_lower for t in ['date', 'time', 'timestamp']):
            return 'datetime'
        
        # JSON
        if 'json' in db_type_lower:
            return 'json'
        
        return 'string'
    
    def _convert_foreign_keys(self, fk_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert PostgreSQL foreign key info to schema format."""
        fks = []
        for fk in fk_results:
            fks.append({
                "column": fk['column_name'],
                "references_table": fk['foreign_table_name'],
                "references_column": fk['foreign_column_name']
            })
        return fks
    
    def _convert_mysql_foreign_keys(self, fk_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MySQL foreign key info to schema format."""
        fks = []
        for fk in fk_results:
            fks.append({
                "column": fk['COLUMN_NAME'],
                "references_table": fk['REFERENCED_TABLE_NAME'],
                "references_column": fk['REFERENCED_COLUMN_NAME']
            })
        return fks
    
    def _convert_sqlite_foreign_keys(self, fk_results: List) -> List[Dict[str, Any]]:
        """Convert SQLite foreign key info to schema format."""
        fks = []
        for fk in fk_results:
            fks.append({
                "column": fk['from'],
                "references_table": fk['table'],
                "references_column": fk['to']
            })
        return fks
    
    def _infer_mongodb_schema(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Infer schema from MongoDB documents."""
        # Collect all fields and their types
        field_types = {}
        
        for doc in documents:
            for field, value in doc.items():
                if field not in field_types:
                    field_types[field] = set()
                
                field_types[field].add(type(value).__name__)
        
        # Build field list
        fields = []
        for field_name, types in field_types.items():
            # Use most common type
            primary_type = list(types)[0] if types else 'string'
            
            fields.append({
                "name": field_name,
                "type": self._normalize_python_type(primary_type),
                "nullable": True,  # MongoDB fields are implicitly nullable
                "default": None
            })
        
        return fields
    
    def _normalize_python_type(self, python_type: str) -> str:
        """Convert Python type names to schema types."""
        type_map = {
            'int': 'integer',
            'float': 'float',
            'str': 'string',
            'bool': 'boolean',
            'dict': 'json',
            'list': 'array',
            'datetime': 'datetime',
            'NoneType': 'string'
        }
        return type_map.get(python_type, 'string')
    
    def _build_relationships(self, tables: Dict[str, Dict[str, Any]]) -> None:
        """Build relationships from foreign keys."""
        for table_name, table in tables.items():
            for fk in table.get('foreign_keys', []):
                # Add relationship
                table['relationships'].append({
                    "type": "belongs_to",
                    "target_table": fk['references_table'],
                    "foreign_key": fk['column']
                })


# Convenience function
def inspect_database(
    connection: Union[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    auto_save: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to introspect database and generate schema.
    
    Args:
        connection: Connection string or config dict
        output_path: Output path (optional, defaults to schemas/database_schema.json)
        auto_save: Automatically save to schemas/ directory (default: True)
        **kwargs: Additional inspection options
    
    Usage:
        # Generate and auto-save to schemas/
        schema = inspect_database('postgresql://user:pass@localhost/mydb')
        
        # Generate and save to custom location
        schema = inspect_database(
            'postgresql://user:pass@localhost/mydb',
            output_path='custom/database_schema.json'
        )
        
        # Generate without saving
        schema = inspect_database(
            'postgresql://user:pass@localhost/mydb',
            auto_save=False
        )
    """
    inspector = DatabaseInspector()
    schema = inspector.generate(connection, **kwargs)
    
    if auto_save or output_path:
        saved_path = inspector.save_schema(schema, output_path)
        schema['_saved_path'] = saved_path
    
    return schema
