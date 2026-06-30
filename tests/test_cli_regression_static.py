import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative_path: str) -> str:
    return ROOT.joinpath(relative_path).read_text()


def parse_source(relative_path: str) -> ast.Module:
    return ast.parse(read_source(relative_path), filename=relative_path)


class CliRegressionStaticTests(unittest.TestCase):
    def test_train_parser_keeps_cli_lr_and_parses_num_workers_as_int(self):
        tree = parse_source("src/kgcl_retro/cli/train.py")
        source = read_source("src/kgcl_retro/cli/train.py")

        def assigns_lr(node, value):
            return (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "args"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "lr"
                    for target in node.targets
                )
                and isinstance(node.value, ast.Constant)
                and node.value.value == value
            )

        lr_default_guard = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == "args.get('lr') is None"
            ),
            None,
        )
        self.assertIsNotNone(lr_default_guard)
        self.assertTrue(any(assigns_lr(node, 0.0001) for node in ast.walk(lr_default_guard)))
        self.assertTrue(any(assigns_lr(node, 0.001) for node in ast.walk(lr_default_guard)))
        self.assertIn("parser.add_argument('--lr', type=float, default=None", source)
        self.assertIn("parser.add_argument('--num_workers', type=int", source)

    def test_prepare_data_skips_empty_final_batch(self):
        source = read_source("src/kgcl_retro/cli/prepare_data.py")

        self.assertIn("if batch_graphs:", source)
        self.assertIn("No valid reactions to save", source)

    def test_eval_scripts_accept_checkpoint_argument(self):
        for relative_path in [
            "src/kgcl_retro/cli/eval_50k.py",
            "src/kgcl_retro/cli/eval_full.py",
            "src/kgcl_retro/cli/eval_roundtrip.py",
        ]:
            source = read_source(relative_path)
            self.assertIn("--checkpoint", source, relative_path)
            self.assertIn("args.checkpoint", source, relative_path)

    def test_roundtrip_creates_prediction_directory(self):
        source = read_source("src/kgcl_retro/cli/eval_roundtrip.py")

        self.assertIn("pred_text_dir", source)
        self.assertIn("os.makedirs(pred_text_dir, exist_ok=True)", source)

    def test_beam_search_clamps_step_topk_to_available_actions(self):
        source = read_source("src/kgcl_retro/models/beam_search.py")

        self.assertIn("min(self.step_beam_size, int(edit_logits.numel()))", source)

    def test_eval_reported_topk_values_are_limited_by_beam_size(self):
        for relative_path in [
            "src/kgcl_retro/cli/eval_50k.py",
            "src/kgcl_retro/cli/eval_full.py",
            "src/kgcl_retro/cli/eval_roundtrip.py",
        ]:
            source = read_source(relative_path)
            self.assertIn("report_top_ks", source, relative_path)
            self.assertIn("if k <= args.beam_size", source, relative_path)


if __name__ == "__main__":
    unittest.main()
