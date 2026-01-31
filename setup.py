from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ''

setup(
    name='enable-ai',
    version='0.3.28',
    author='Enable Engineering',
    author_email='engineering@enableyou.co',
    description='AI-powered natural language interface for REST APIs with OpenAPI support and real-time streaming',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/EnableEngineering/enable_ai',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'requests>=2.25.0',
        'openai>=1.0.0',
        'langgraph>=0.2.0',
        'python-dotenv>=0.19.0',
        'langgraph-checkpoint>=1.0.0',
    ],
    extras_require={
        'mcp': [
            # MCP server support (optional)
            # Install MCP SDK manually if needed: pip install mcp
            # Note: MCP SDK may not be available on PyPI yet
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'enable-schema=enable_ai.schema_generator.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.7',
)