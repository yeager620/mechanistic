"""
Quick status checker for the phi-2 TDA pipeline.
"""
from pathlib import Path
import h5py
from utils import logger

def check_pipeline_status():
    """Check the current status of the pipeline."""
    logger.info("🔍 PIPELINE STATUS CHECK")
    logger.info("=" * 40)
    
    # Check 1: Downloaded weights
    weights_file = Path("phi2_weights.h5")
    if weights_file.exists():
        try:
            with h5py.File(weights_file, "r") as f:
                num_matrices = len(f.keys())
                file_size = weights_file.stat().st_size / (1024**2)
                logger.info(f"✅ Weights downloaded: {num_matrices} matrices, {file_size:.1f} MB")
                
                # Show some matrix names
                matrix_names = sorted(list(f.keys()))
                logger.info(f"   Sample matrices: {matrix_names[:5]}...")
        except Exception as e:
            logger.error(f"❌ Weights file corrupted: {e}")
    else:
        logger.info("❌ Weights not downloaded yet")
        logger.info("   Run: uv run python 00_download.py")
    
    # Check 2: Point clouds
    pc_dir = Path("point_clouds")
    if pc_dir.exists():
        pc_files = list(pc_dir.glob("*.npz"))
        if pc_files:
            logger.info(f"✅ Point clouds extracted: {len(pc_files)} files")
            
            # Show some file names
            file_names = [f.stem for f in pc_files if not f.name.startswith('summary')]
            logger.info(f"   Sample strata: {file_names[:5]}...")
        else:
            logger.info("❌ Point cloud directory empty")
    else:
        logger.info("❌ Point clouds not extracted yet")
        logger.info("   Run: uv run python 01_extract.py")
    
    # Check 3: TDA results
    tda_dir = Path("tda_results")
    if tda_dir.exists():
        tda_files = list(tda_dir.glob("*"))
        if tda_files:
            logger.info(f"✅ TDA computed: {len(tda_files)} result files")
            
            # Check for key files
            key_files = ['persistence_diagrams.pkl', 'betti_analysis.json']
            for key_file in key_files:
                if (tda_dir / key_file).exists():
                    logger.info(f"   ✅ {key_file}")
                else:
                    logger.info(f"   ❌ {key_file}")
        else:
            logger.info("❌ TDA results directory empty")
    else:
        logger.info("❌ TDA not computed yet")
        logger.info("   Run: uv run python 02_tda.py")
    
    # Check 4: Test files
    test_files = ["test_weights.h5", "test_point_clouds", "test_tda_results"]
    test_exists = [Path(f).exists() for f in test_files]
    if any(test_exists):
        logger.info(f"ℹ️  Test files present: {sum(test_exists)}/3")
        logger.info("   (These are from pipeline testing)")
    
    # Recommendations
    logger.info("\n📋 NEXT STEPS:")
    if not weights_file.exists():
        logger.info("1. Download weights: uv run python 00_download.py")
    elif weights_file.exists() and not pc_dir.exists():
        logger.info("1. Explore weights: uv run python explore_weights.py")
        logger.info("2. Extract point clouds: uv run python 01_extract.py")
    elif pc_dir.exists() and not tda_dir.exists():
        logger.info("1. Explore point clouds: uv run python explore_pointclouds.py")
        logger.info("2. Compute TDA: uv run python 02_tda.py")
    else:
        logger.info("1. Explore results: jupyter notebook 03_viz.ipynb")
        logger.info("2. Or continue with analysis...")

if __name__ == "__main__":
    check_pipeline_status()