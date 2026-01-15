#!/usr/bin/env python3
"""
Sweep through MOE kernel configurations, launch servers, and run benchmarks.

Error handling:
- Each configuration is isolated - failures don't affect subsequent runs
- Server cleanup happens in finally blocks, ensuring no zombie processes
- Ctrl+C is handled gracefully with proper cleanup
"""

import subprocess
import time
import signal
import sys
import os
import requests
from dataclasses import dataclass

# Global flag for interrupt handling
_interrupted = False
_current_server_process = None


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _interrupted
    print("\n\n>>> Interrupt received! Cleaning up before exit...")
    _interrupted = True
    
    # Clean up current server if running
    if _current_server_process is not None:
        try:
            _current_server_process.terminate()
            _current_server_process.wait(timeout=5)
        except Exception:
            try:
                _current_server_process.kill()
            except Exception:
                pass
    
    # Kill any remaining processes on the port
    subprocess.run(
        f"lsof -ti:{SERVER_PORT} | xargs -r kill -9",
        shell=True,
        check=False,
        capture_output=True
    )
    subprocess.run(
        "pkill -9 -f 'sglang.launch_server' || true",
        shell=True,
        check=False,
        capture_output=True
    )
    
    print("Cleanup complete. Exiting.")
    sys.exit(1)

@dataclass
class Config:
    name: str
    quantization: str
    command: str
    model_for_benchmark: str

# Configuration list extracted from the table
CONFIGS = [
    Config(
        name="trtllm_fp4_block_scale_moe_w4a16",
        quantization="w4a16",
        command="python -m sglang.launch_server --model openai/gpt-oss-20b --flashinfer-mxfp4-moe-precision bf16",
        model_for_benchmark="openai/gpt-oss-20b",
    ),
    Config(
        name="flashinfer_cutlass_fused_moe_w16a16",
        quantization="w16a16",
        command="python -m sglang.launch_server --model openai/gpt-oss-20b --moe-runner-backend flashinfer_cutlass",
        model_for_benchmark="openai/gpt-oss-20b",
    ),
    Config(
        name="matmul_ogs_w4a16",
        quantization="w4a16",
        command="python -m sglang.launch_server --model openai/gpt-oss-20b --moe-runner-backend triton_kernel",
        model_for_benchmark="openai/gpt-oss-20b",
    ),
    Config(
        name="fused_moe_kernel_w16a16",
        quantization="w16a16",
        command="python -m sglang.launch_server --model openai/gpt-oss-20b --moe-runner-backend triton",
        model_for_benchmark="openai/gpt-oss-20b",
    ),
    Config(
        name="trtllm_fp4_block_scale_moe_w4a8",
        quantization="w4a8",
        command="python -m sglang.launch_server --model openai/gpt-oss-20b",
        model_for_benchmark="openai/gpt-oss-20b",
    ),
    # flashinfer_cutlass_fused_moe w4a8 - skipped, no command provided
    Config(
        name="cutlass_fp4_group_w4a4",
        quantization="w4a4",
        command="python -m sglang.launch_server --model shanjiaz/gpt-oss-20b-nvfp4-modelopt --quantization modelopt_fp4 --moe-runner-backend cutlass",
        model_for_benchmark="shanjiaz/gpt-oss-20b-nvfp4-modelopt",
    ),
    Config(
        name="flashinfer_cutedsl_moe_masked_w4a4",
        quantization="w4a4",
        command="python -m sglang.launch_server --model-path shanjiaz/gpt-oss-20b-nvfp4-modelopt --quantization modelopt_fp4 --moe-runner-backend flashinfer_cutedsl",
        model_for_benchmark="shanjiaz/gpt-oss-20b-nvfp4-modelopt",
    ),
    Config(
        name="flashinfer_cutlass_fused_moe_w4a4",
        quantization="w4a4",
        command="python -m sglang.launch_server --model-path shanjiaz/gpt-oss-20b-nvfp4-modelopt --quantization modelopt_fp4 --moe-runner-backend flashinfer_cutlass",
        model_for_benchmark="shanjiaz/gpt-oss-20b-nvfp4-modelopt",
    ),
]

# Server configuration
SERVER_HOST = "localhost"
SERVER_PORT = 30000
HEALTH_ENDPOINT = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
MAX_WAIT_TIME = 600  # 10 minutes max wait for server startup
POLL_INTERVAL = 5    # Check every 5 seconds

# Benchmark configuration
BENCHMARK_CMD_TEMPLATE = (
    "python -m benchmarking.unified.benchmark {model} "
    "-i 1024 -o 1024 --disable_early_concurrency_stop "
    "--concurrency 1,4,8,16,32,64,128 --llm_api sglang --csv {csv_name}"
)

# Output directory for results
OUTPUT_DIR = "benchmark_results"


def wait_for_server_ready(timeout: int = MAX_WAIT_TIME) -> bool:
    """Wait for the server to be ready by polling the health endpoint."""
    start_time = time.time()
    print(f"Waiting for server to be ready at {HEALTH_ENDPOINT}...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        elapsed = int(time.time() - start_time)
        print(f"  Still waiting... ({elapsed}s elapsed)")
        time.sleep(POLL_INTERVAL)
    
    print(f"Server did not become ready within {timeout} seconds")
    return False


def kill_server_process(process: subprocess.Popen):
    """Kill the server process and all its children."""
    print("Shutting down server...")
    try:
        # Try graceful termination first
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Force kill if graceful termination didn't work
            process.kill()
            process.wait()
        print("Server shut down successfully")
    except Exception as e:
        print(f"Error shutting down server: {e}")
        # Try to kill by port as fallback
        try:
            subprocess.run(
                f"lsof -ti:{SERVER_PORT} | xargs -r kill -9",
                shell=True,
                check=False
            )
        except Exception:
            pass


def run_benchmark(config: Config, output_dir: str) -> bool:
    """Run the benchmark for a given configuration."""
    csv_name = os.path.join(output_dir, f"{config.name}.csv")
    benchmark_cmd = BENCHMARK_CMD_TEMPLATE.format(
        model=config.model_for_benchmark,
        csv_name=csv_name
    )
    
    print(f"\n{'='*60}")
    print(f"Running benchmark: {config.name}")
    print(f"Command: {benchmark_cmd}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            benchmark_cmd,
            shell=True,
            check=True,
            cwd="/home/ec2-user/performance"  # Adjust if benchmark is elsewhere
        )
        print(f"Benchmark completed successfully. Results saved to {csv_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed with return code {e.returncode}")
        return False


def cleanup_server(process: subprocess.Popen = None):
    """Comprehensive cleanup to ensure server is fully stopped."""
    if process is not None:
        kill_server_process(process)
    
    # Extra cleanup: kill any remaining processes on the port
    subprocess.run(
        f"lsof -ti:{SERVER_PORT} | xargs -r kill -9",
        shell=True,
        check=False,
        capture_output=True
    )
    
    # Also try pkill for any sglang processes that might be orphaned
    subprocess.run(
        "pkill -9 -f 'sglang.launch_server' || true",
        shell=True,
        check=False,
        capture_output=True
    )
    
    time.sleep(5)  # Wait for cleanup to complete


def run_config(config: Config, output_dir: str) -> tuple[bool, str]:
    """Run a single configuration: start server, run benchmark, stop server.
    
    Returns:
        (success: bool, error_message: str) - error_message is empty on success
    """
    global _current_server_process, _interrupted
    
    # Check if we were interrupted
    if _interrupted:
        return False, "Interrupted by user"
    
    print(f"\n{'#'*60}")
    print(f"# Configuration: {config.name}")
    print(f"# Quantization: {config.quantization}")
    print(f"# Server command: {config.command}")
    print(f"{'#'*60}\n")
    
    server_process = None
    
    try:
        # Ensure any previous server is killed before starting
        cleanup_server()
        
        # Start the server
        server_cmd = f"{config.command} --port {SERVER_PORT}"
        print(f"Starting server: {server_cmd}")
        
        server_process = subprocess.Popen(
            server_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid  # Create new process group for cleanup
        )
        _current_server_process = server_process  # Track for signal handler
        
        # Wait for server to be ready
        if not wait_for_server_ready():
            error_msg = "Server failed to start within timeout"
            print(f"ERROR: {error_msg}")
            return False, error_msg
        
        # Give it a moment to fully initialize
        time.sleep(5)
        
        # Run benchmark
        success = run_benchmark(config, output_dir)
        
        if success:
            return True, ""
        else:
            return False, "Benchmark command failed"
        
    except subprocess.SubprocessError as e:
        error_msg = f"Subprocess error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {e}"
        print(f"ERROR: {error_msg}")
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}: {e}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg
        
    finally:
        # ALWAYS clean up the server, no matter what happened
        print("Cleaning up server...")
        cleanup_server(server_process)
        _current_server_process = None  # Clear global reference
        print("Cleanup complete. Ready for next configuration.")


def main():
    # Register signal handler for graceful interrupt
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Sweep MOE kernel benchmarks")
    parser.add_argument(
        "--output-dir", 
        default=OUTPUT_DIR,
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Run only a specific config by name (partial match)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available configurations"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    args = parser.parse_args()
    
    # List configs if requested
    if args.list:
        print("Available configurations:")
        for i, config in enumerate(CONFIGS):
            print(f"  {i+1}. {config.name} ({config.quantization})")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Filter configs if specified
    configs_to_run = CONFIGS
    if args.config:
        configs_to_run = [c for c in CONFIGS if args.config.lower() in c.name.lower()]
        if not configs_to_run:
            print(f"No configs matching '{args.config}'")
            return
    
    # Track results
    results = []
    
    print(f"Will run {len(configs_to_run)} configuration(s)")
    print(f"Results will be saved to: {args.output_dir}/")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - Commands that would be executed:]\n")
        for config in configs_to_run:
            print(f"Config: {config.name}")
            print(f"  Server: {config.command} --port {SERVER_PORT}")
            benchmark_cmd = BENCHMARK_CMD_TEMPLATE.format(
                model=config.model_for_benchmark,
                csv_name=os.path.join(args.output_dir, f"{config.name}.csv")
            )
            print(f"  Benchmark: {benchmark_cmd}")
            print()
        return
    
    # Run each configuration - errors in one don't affect others
    for i, config in enumerate(configs_to_run):
        # Check if interrupted
        if _interrupted:
            print("\nInterrupted! Skipping remaining configurations.")
            break
            
        print(f"\n{'*'*60}")
        print(f"* Running configuration {i+1}/{len(configs_to_run)}: {config.name}")
        print(f"{'*'*60}")
        
        try:
            success, error_msg = run_config(config, args.output_dir)
            results.append((config.name, success, error_msg))
        except Exception as e:
            # Catch-all for any truly unexpected errors
            error_msg = f"Critical error: {type(e).__name__}: {e}"
            print(f"CRITICAL ERROR in {config.name}: {error_msg}")
            import traceback
            traceback.print_exc()
            results.append((config.name, False, error_msg))
            
            # Make sure we clean up even after critical errors
            print("Attempting emergency cleanup...")
            try:
                cleanup_server()
            except Exception:
                pass
        
        print(f"\n>>> Completed {config.name}, moving to next configuration...")
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SWEEP SUMMARY")
    print(f"{'='*60}")
    for name, success, error_msg in results:
        if success:
            print(f"  ✓ SUCCESS: {name}")
        else:
            print(f"  ✗ FAILED:  {name}")
            if error_msg:
                print(f"             Error: {error_msg}")
    
    successful = sum(1 for _, s, _ in results if s)
    failed = len(results) - successful
    print(f"\nTotal: {successful}/{len(results)} configurations completed successfully")
    if failed > 0:
        print(f"       {failed} configuration(s) failed")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

