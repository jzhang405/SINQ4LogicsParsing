#!/usr/bin/env python3
"""
Main entry point for SINQ quantization CLI.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cli.quantize_cli import main

if __name__ == "__main__":
    main()