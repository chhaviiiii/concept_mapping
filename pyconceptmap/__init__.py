"""
PyConceptMap: An Open-Source Concept Mapping Tool in Python

A comprehensive Python implementation of concept mapping methodology,
inspired by RCMap (Bar & Mentch, 2017).

Author: PyConceptMap Development Team
License: MIT
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "PyConceptMap Development Team"
__email__ = "pyconceptmap@example.com"

from .core import ConceptMappingAnalysis
from .data_handler import DataHandler
from .visualizer import ConceptMapVisualizer
from .reporter import ReportGenerator
from .utils import validate_data, check_requirements, create_sample_data

__all__ = [
    'ConceptMappingAnalysis',
    'DataHandler', 
    'ConceptMapVisualizer',
    'ReportGenerator',
    'validate_data',
    'check_requirements',
    'create_sample_data'
]
