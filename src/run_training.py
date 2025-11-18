"""
Wrapper script to run training with proper path setup
"""
import sys
import os

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Now import and run the training
import train_context_aware
