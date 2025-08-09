#!/usr/bin/env python3
"""
Main entry point for the Emergent Communication Research Suite
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import and run the CLI
from src.cli.research_suite import main

if __name__ == "__main__":
    main()