"""
Main entry point for the Backend API.
This file imports the app from gateway_main.py for test compatibility.
"""
import os
import uuid
from gateway.gateway_main import app

# Mock attributes for testing
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
job_status = {}

def run_tiling_job(*args, **kwargs):
    """Mock function for testing"""
    pass

# Make app available for import in tests
__all__ = ['app']
