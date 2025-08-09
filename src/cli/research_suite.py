#!/usr/bin/env python3
"""
Research Suite Runner for Emergent Communication
Unified interface for training, ablation studies, generalization tests, and visualization
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

def print_banner():
    """Print research suite banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                 EMERGENT COMMUNICATION                       ║
║                    RESEARCH SUITE                            ║
║                                                              ║
║  🧠 Training  🔬 Ablation  🌐 Generalization  📊 Dashboard   ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def run_training(args):
    """Run training with enhanced monitoring"""
    print("🚀 Starting Enhanced Training...")

    # Start training
    cmd = ['python', 'scripts/train.py']
    if args.episodes:
        cmd.extend(['--episodes', str(args.episodes)])
    if args.device:
        cmd.extend(['--device', args.device])

    print(f"Running: {' '.join(cmd)}")

    # Run training in background if dashboard requested
    if args.dashboard:
        print("📊 Starting training with real-time dashboard...")

        # Start training process
        training_process = subprocess.Popen(cmd, cwd=project_root)

        # Wait a moment for training to start
        time.sleep(3)

        # Start dashboard
        dashboard_cmd = ['python', '-m', 'src.visualization.dashboard', '--mode', 'training']
        if args.training_log:
            dashboard_cmd.extend(['--training_log', args.training_log])

        try:
            subprocess.run(dashboard_cmd, cwd=project_root)
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped. Training continues in background.")

        # Wait for training to complete
        training_process.wait()
    else:
        # Run training normally
        subprocess.run(cmd, cwd=project_root)

    print("✅ Training completed!")

def run_ablation_studies(args):
    """Run ablation studies"""
    print("🔬 Starting Ablation Studies...")

    cmd = ['python', '-m', 'src.research.ablation']

    if args.category and args.category != 'all':
        cmd.extend(['--category', args.category])
    if args.episodes:
        cmd.extend(['--episodes', str(args.episodes)])
    if args.output_dir:
        cmd.extend(['--output_dir', args.output_dir])

    print(f"Running: {' '.join(cmd)}")

    if args.dashboard:
        print("📊 Starting ablation studies with monitoring dashboard...")

        # Start ablation process
        ablation_process = subprocess.Popen(cmd, cwd=project_root)

        # Wait for some results
        time.sleep(10)

        # Start dashboard
        dashboard_cmd = ['python', '-m', 'src.visualization.dashboard', '--mode', 'ablation']
        if args.output_dir:
            dashboard_cmd.extend(['--ablation_results', args.output_dir])

        try:
            subprocess.run(dashboard_cmd, cwd=project_root)
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped. Ablation studies continue in background.")

        # Wait for ablation to complete
        ablation_process.wait()
    else:
        subprocess.run(cmd, cwd=project_root)

    print("✅ Ablation studies completed!")

def run_generalization_tests(args):
    """Run generalization tests"""
    print("🌐 Starting Generalization Tests...")

    cmd = ['python', '-m', 'src.research.generalization']

    if args.model_path:
        cmd.extend(['--model_path', args.model_path])
    if args.episodes:
        cmd.extend(['--episodes', str(args.episodes)])
    if args.output_dir:
        cmd.extend(['--output_dir', args.output_dir])

    print(f"Running: {' '.join(cmd)}")

    if args.dashboard:
        print("📊 Starting generalization tests with monitoring dashboard...")

        # Start generalization process
        gen_process = subprocess.Popen(cmd, cwd=project_root)

        # Wait for some results
        time.sleep(5)

        # Start dashboard
        dashboard_cmd = ['python', '-m', 'src.visualization.dashboard', '--mode', 'generalization']
        if args.output_dir:
            results_file = os.path.join(args.output_dir, 'generalization_results.json')
            dashboard_cmd.extend(['--generalization_results', results_file])

        try:
            subprocess.run(dashboard_cmd, cwd=project_root)
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped. Generalization tests continue in background.")

        # Wait for generalization to complete
        gen_process.wait()
    else:
        subprocess.run(cmd, cwd=project_root)

    print("✅ Generalization tests completed!")

def run_dashboard_only(args):
    """Run dashboard only"""
    print("📊 Starting Research Dashboard...")

    cmd = ['python', '-m', 'src.visualization.dashboard']

    if args.mode:
        cmd.extend(['--mode', args.mode])
    if args.training_log:
        cmd.extend(['--training_log', args.training_log])
    if args.ablation_results:
        cmd.extend(['--ablation_results', args.ablation_results])
    if args.generalization_results:
        cmd.extend(['--generalization_results', args.generalization_results])
    if args.update_interval:
        cmd.extend(['--update_interval', str(args.update_interval)])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=project_root)

def run_full_research_pipeline(args):
    """Run complete research pipeline"""
    print("🔄 Starting Full Research Pipeline...")

    # Step 1: Training
    print("\n" + "="*60)
    print("STEP 1: TRAINING")
    print("="*60)

    training_args = argparse.Namespace(
        episodes=args.episodes or 1000,
        device=args.device,
        dashboard=False,  # No dashboard during pipeline
        training_log=None
    )
    run_training(training_args)

    # Step 2: Ablation Studies
    print("\n" + "="*60)
    print("STEP 2: ABLATION STUDIES")
    print("="*60)

    ablation_args = argparse.Namespace(
        category='all',
        episodes=args.episodes or 500,
        output_dir='ablation_results',
        dashboard=False
    )
    run_ablation_studies(ablation_args)

    # Step 3: Generalization Tests
    print("\n" + "="*60)
    print("STEP 3: GENERALIZATION TESTS")
    print("="*60)

    # Find the latest trained model
    model_path = 'results/architect_builder_v1/models/final_model.pt'
    if not os.path.exists(model_path):
        # Look for any available model
        for root, dirs, files in os.walk('results'):
            for file in files:
                if file.endswith('.pt') and 'final' in file:
                    model_path = os.path.join(root, file)
                    break

    gen_args = argparse.Namespace(
        model_path=model_path,
        episodes=args.episodes or 50,
        output_dir='generalization_results',
        dashboard=False
    )
    run_generalization_tests(gen_args)

    # Step 4: Comprehensive Analysis
    print("\n" + "="*60)
    print("STEP 4: COMPREHENSIVE ANALYSIS")
    print("="*60)

    # Run integrated evaluation
    if os.path.exists(model_path):
        eval_cmd = ['python', '-m', 'src.evaluation.integrated', '--model_path', model_path]
        print(f"Running: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, cwd=project_root)

    print("\n🎉 Full research pipeline completed!")
    print("\n📋 RESULTS SUMMARY:")
    print("  📁 Training results: results/architect_builder_v1/")
    print("  📁 Ablation results: ablation_results/")
    print("  📁 Generalization results: generalization_results/")
    print("  📁 Evaluation results: evaluation_results/")

    # Offer to start comparison dashboard
    print("\n📊 Starting comparison dashboard to view all results...")
    dashboard_args = argparse.Namespace(
        mode='comparison',
        training_log='results/architect_builder_v1/logs/training_log.json',
        ablation_results='ablation_results',
        generalization_results='generalization_results/generalization_results.json',
        update_interval=5000
    )
    run_dashboard_only(dashboard_args)

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = [
        'torch', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'tqdm'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False

    return True

def show_status():
    """Show current project status"""
    print("📊 PROJECT STATUS:")
    print("-" * 40)

    # Check for trained models
    models_found = []
    for root, dirs, files in os.walk('results'):
        for file in files:
            if file.endswith('.pt'):
                models_found.append(os.path.join(root, file))

    if models_found:
        print(f"✅ Found {len(models_found)} trained model(s):")
        for model in models_found[:3]:  # Show first 3
            print(f"   📦 {model}")
        if len(models_found) > 3:
            print(f"   ... and {len(models_found) - 3} more")
    else:
        print("❌ No trained models found")

    # Check for ablation results
    ablation_dir = Path('ablation_results')
    if ablation_dir.exists():
        result_files = list(ablation_dir.glob('*_results.json'))
        print(f"✅ Found {len(result_files)} ablation result file(s)")
    else:
        print("❌ No ablation results found")

    # Check for generalization results
    gen_file = Path('generalization_results/generalization_results.json')
    if gen_file.exists():
        print("✅ Generalization results available")
    else:
        print("❌ No generalization results found")

    # Check for evaluation results
    eval_dir = Path('evaluation_results')
    if eval_dir.exists():
        eval_files = list(eval_dir.glob('*.json'))
        print(f"✅ Found {len(eval_files)} evaluation result file(s)")
    else:
        print("❌ No evaluation results found")

def main():
    """Main function with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(
        description='Emergent Communication Research Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run training with dashboard
  python -m src.cli.research_suite train --dashboard --episodes 1000

  # Run ablation studies for message length
  python -m src.cli.research_suite ablation --category message_length --episodes 500

  # Run generalization tests
  python -m src.cli.research_suite generalization --model_path results/model.pt

  # Start dashboard only
  python -m src.cli.research_suite dashboard --mode training

  # Run full research pipeline
  python -m src.cli.research_suite pipeline --episodes 500

  # Show project status
  python -m src.cli.research_suite status
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Training command
    train_parser = subparsers.add_parser('train', help='Run training')
    train_parser.add_argument('--episodes', type=int, help='Number of training episodes')
    train_parser.add_argument('--device', type=str, help='Training device (cuda/cpu)')
    train_parser.add_argument('--dashboard', action='store_true', help='Show real-time dashboard')
    train_parser.add_argument('--training_log', type=str, help='Training log file path')

    # Ablation command
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation studies')
    ablation_parser.add_argument('--category', type=str,
                                choices=['message_length', 'vocab_size', 'architecture',
                                        'communication', 'training', 'environment', 'all'],
                                default='all', help='Ablation category')
    ablation_parser.add_argument('--episodes', type=int, help='Episodes per experiment')
    ablation_parser.add_argument('--output_dir', type=str, default='ablation_results',
                                help='Output directory')
    ablation_parser.add_argument('--dashboard', action='store_true', help='Show monitoring dashboard')

    # Generalization command
    gen_parser = subparsers.add_parser('generalization', help='Run generalization tests')
    gen_parser.add_argument('--model_path', type=str,
                           default='results/architect_builder_v1/models/final_model.pt',
                           help='Path to trained model')
    gen_parser.add_argument('--episodes', type=int, help='Episodes per test scenario')
    gen_parser.add_argument('--output_dir', type=str, default='generalization_results',
                           help='Output directory')
    gen_parser.add_argument('--dashboard', action='store_true', help='Show monitoring dashboard')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Start dashboard only')
    dashboard_parser.add_argument('--mode', type=str,
                                 choices=['training', 'ablation', 'generalization', 'comparison'],
                                 default='training', help='Dashboard mode')
    dashboard_parser.add_argument('--training_log', type=str, help='Training log file')
    dashboard_parser.add_argument('--ablation_results', type=str, help='Ablation results directory')
    dashboard_parser.add_argument('--generalization_results', type=str, help='Generalization results file')
    dashboard_parser.add_argument('--update_interval', type=int, help='Update interval (ms)')

    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full research pipeline')
    pipeline_parser.add_argument('--episodes', type=int, help='Episodes for each stage')
    pipeline_parser.add_argument('--device', type=str, help='Training device')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show project status')

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Check dependencies
    if not check_dependencies():
        return

    # Handle commands
    if args.command == 'train':
        run_training(args)
    elif args.command == 'ablation':
        run_ablation_studies(args)
    elif args.command == 'generalization':
        run_generalization_tests(args)
    elif args.command == 'dashboard':
        run_dashboard_only(args)
    elif args.command == 'pipeline':
        run_full_research_pipeline(args)
    elif args.command == 'status':
        show_status()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()