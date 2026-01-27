import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir="output/logs", level="INFO", debug=False):
    """Setup logging with file and console handlers
    
    Creates a timestamped log file for each run: pipeline_YYYYMMDD_HHMMSS.log
    
    Args:
        log_dir: Directory for log files
        level: Logging level (INFO, DEBUG, etc.)
        debug: If True, enable detailed logging like test mode
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Store debug mode globally
    logger = logging.getLogger("pipeline")
    logger.debug_mode = debug
    logger.setLevel(logging.DEBUG if debug else getattr(logging, level))
    logger.handlers.clear()
    
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"pipeline_{timestamp}.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    logger.info(f"Logging to: {log_file}")
    
    return logger

def get_logger():
    return logging.getLogger("pipeline")
