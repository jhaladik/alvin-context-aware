"""
Wrapper script to run tests with proper path setup
"""
import sys
import os

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Now import and run the tests
import test_context_aware
