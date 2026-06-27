from pathlib import Path  # Explanation: imports Path to locate the local src package directory.
import sys  # Explanation: imports sys so this legacy wrapper can add src to Python's import path.

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # Explanation: lets this script run before kgcl-retro is installed.

from kgcl_retro.cli.train import build_arg_parser, build_model_config, cli_main, main, save_checkpoint, test, train_epoch  # Explanation: re-exports packaged training entry points and helpers.

__all__ = [  # Explanation: preserves the public API that existed in the original top-level training script.
    "build_arg_parser",  # Explanation: exposes parser construction for reusable CLIs.
    "build_model_config",  # Explanation: exposes model configuration assembly.
    "cli_main",  # Explanation: exposes the console entry point.
    "main",  # Explanation: exposes the training driver.
    "save_checkpoint",  # Explanation: exposes checkpoint writing.
    "test",  # Explanation: exposes validation evaluation.
    "train_epoch",  # Explanation: exposes one training epoch.
]  # Explanation: closes the legacy public API list.


if __name__ == "__main__":  # Explanation: runs the compatibility entry point only when executed as a script.
    cli_main()  # Explanation: invokes the packaged training command.
