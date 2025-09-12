import subprocess
import sys
import os
import time
from datetime import datetime, timedelta

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

from adan_trading_bot.common.utils import get_logger

logger = get_logger(__name__)

def run_endurance_test(duration_hours: int = 24, model_path: str = "models/rl_agents/ppo_model_v1.zip", initial_capital: float = 1000.0):
    """
    Runs the online_learning_agent.py script for a specified duration in paper trading mode.

    Args:
        duration_hours: The duration of the test in hours.
        model_path: Path to the pre-trained PPO model.
        initial_capital: Initial capital for the paper trading.
    """
    logger.info(f"Starting endurance test for {duration_hours} hours...")
    logger.info(f"Using model: {model_path}")
    logger.info(f"Initial capital: {initial_capital}")

    # Construct the command to run online_learning_agent.py
    # We assume online_learning_agent.py is in the same 'scripts' directory
    online_learning_script = os.path.join(current_dir, "online_learning_agent.py")

    # Ensure the model path is relative to the project root if it's not absolute
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    command = [
        sys.executable,  # Use the current Python interpreter
        online_learning_script,
        "--exec_profile", "cpu",  # Assuming CPU for endurance test, can be changed
        "--model_path", model_path,
        "--initial_capital", str(initial_capital),
        "--max_iterations", "999999999",  # Effectively run indefinitely until duration is met
        "--sleep_seconds", "60",  # Check every minute
        "--learning_frequency", "10", # Learn every 10 decisions
        "--exploration_rate", "0.05" # Low exploration for endurance
    ]

    logger.info(f"Executing command: {' '.join(command)}")

    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)

    process = None
    try:
        # Start the process in a non-blocking way
        process = subprocess.Popen(command, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info(f"Online learning agent started with PID: {process.pid}")

        # Monitor the process and duration
        while datetime.now() < end_time:
            # Read stdout and stderr to prevent buffer overflow
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()
            if stdout_line:
                logger.info(f"AGENT_STDOUT: {stdout_line.strip()}")
            if stderr_line:
                logger.error(f"AGENT_STDERR: {stderr_line.strip()}")

            # Check if the process has terminated
            if process.poll() is not None:
                logger.error(f"Online learning agent terminated unexpectedly with exit code {process.returncode}")
                break
            time.sleep(10) # Check every 10 seconds

    except KeyboardInterrupt:
        logger.info("Endurance test interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during the endurance test: {e}")
    finally:
        if process and process.poll() is None:
            logger.info("Terminating online learning agent process...")
            process.terminate()
            try:
                process.wait(timeout=5) # Give it some time to terminate
            except subprocess.TimeoutExpired:
                logger.warning("Process did not terminate gracefully, killing it.")
                process.kill()
        logger.info("Endurance test finished.")

if __name__ == "__main__":
    # Example usage: run for 24 hours
    # You might need to adjust the model_path based on your actual pre-trained model location
    # For a real test, ensure you have a valid model at the specified path.
    run_endurance_test(duration_hours=24, model_path="models/rl_agents/ppo_model_v1.zip", initial_capital=1000.0)
