#!/usr/bin/env python3
"""
Quick Examples for Emergent Communication Research Suite
"""

import os
import subprocess

def main():
    """Show usage examples"""
    print("EMERGENT COMMUNICATION RESEARCH SUITE")
    print("=" * 50)
    print()

    examples = [
        ("Start Training with Dashboard",
         "python research_suite.py train --dashboard --episodes 100"),

        ("Run Ablation Studies",
         "python research_suite.py ablation --category message_length --episodes 50"),

        ("Test Generalization",
         "python research_suite.py generalization --episodes 20"),

        ("View Dashboard",
         "python research_suite.py dashboard --mode training"),

        ("Full Pipeline",
         "python research_suite.py pipeline --episodes 200"),

        ("Check Status",
         "python research_suite.py status"),
    ]

    for i, (desc, cmd) in enumerate(examples, 1):
        print(f"{i}. {desc}")
        print(f"   {cmd}")
        print()

    print("For detailed help:")
    print("   python research_suite.py --help")
    print()

    choice = input("Enter example number to run (1-6), or 'q' to quit: ")

    if choice == 'q':
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(examples):
            cmd = examples[idx][1]
            print(f"\nRunning: {cmd}")
            subprocess.run(cmd.split())
        else:
            print("Invalid choice")
    except ValueError:
        print("Invalid input")

if __name__ == "__main__":
    main()