#!/usr/bin/env python3
"""
CLI script for quantizing Logics-Parsing model using SINQ.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.pipeline import QuantizationPipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quantization.log')
        ]
    )


def main():
    """Main function for CLI script."""
    parser = argparse.ArgumentParser(
        description="Quantize Logics-Parsing model using SINQ"
    )
    
    parser.add_argument(
        "--model-path", 
        type=str,
        required=True,
        help="Path to the original model directory"
    )
    
    parser.add_argument(
        "--output-path", 
        type=str,
        default="/Users/zhangcz/.cache/modelscope/hub/models/Alibaba-DT/Logics-Parsing-SINQ/",
        help="Path to save the quantized model"
    )
    
    parser.add_argument(
        "--bits", 
        type=int,
        choices=[2, 3, 4, 5, 6, 8],
        default=4,
        help="Quantization bit precision"
    )
    
    parser.add_argument(
        "--group-size", 
        type=int,
        choices=[64, 128],
        default=64,
        help="Group size for quantization"
    )
    
    parser.add_argument(
        "--tiling-mode", 
        type=str,
        choices=["1D", "2D"],
        default="1D",
        help="Tiling strategy"
    )
    
    parser.add_argument(
        "--method", 
        type=str,
        choices=["sinq", "asinq"],
        default="sinq",
        help="Quantization method"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        pipeline = QuantizationPipeline()
        
        # Run quantization
        logger.info("Starting quantization pipeline...")
        
        quantized_model = pipeline.quantize_model(
            model_path=Path(args.model_path),
            bits=args.bits,
            output_path=Path(args.output_path),
            group_size=args.group_size,
            tiling_mode=args.tiling_mode,
            method=args.method
        )
        
        logger.info("Quantization completed successfully!")
        logger.info(f"Quantized model saved to: {args.output_path}")
        
        # Run comprehensive validation
        logger.info("Running comprehensive validation...")
        
        results = pipeline.run_comprehensive_pipeline(
            model_path=Path(args.model_path),
            bits=args.bits,
            output_path=Path(args.output_path),
            validate=True
        )
        
        if results['success']:
            logger.info("Validation completed successfully!")
            
            # Print results
            print("\n" + "="*50)
            print("QUANTIZATION RESULTS")
            print("="*50)
            
            if 'validation_results' in results:
                print(f"\nValidation Results:")
                for key, value in results['validation_results'].items():
                    print(f"  {key}: {value:.4f}")
            
            if 'compression_metrics' in results:
                print(f"\nCompression Metrics:")
                for key, value in results['compression_metrics'].items():
                    if key.endswith('_mb'):
                        print(f"  {key}: {value:.2f} MB")
                    elif key.endswith('_ratio') or key.endswith('_reduction'):
                        print(f"  {key}: {value:.2%}")
                    else:
                        print(f"  {key}: {value:.4f}")
            
            print(f"\nOverall Status: {results['message']}")
            print("="*50)
        else:
            logger.error(f"Validation failed: {results['error']}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()