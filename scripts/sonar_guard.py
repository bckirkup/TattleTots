#!/usr/bin/env python3
"""Fast, conservative checks for Sonar patterns without upstream lint rules."""

from __future__ import annotations

import argparse
import ast
import io
import tokenize
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    rule: str
    message: str


def _float_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, float)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _float_literal(node.operand)
    return False


def _approx_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "approx"
    )


def _assert_float_comparisons(tree: ast.AST, path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        for comparison in ast.walk(node.test):
            if not isinstance(comparison, ast.Compare):
                continue
            if not any(isinstance(op, (ast.Eq, ast.NotEq)) for op in comparison.ops):
                continue
            operands = [comparison.left, *comparison.comparators]
            if any(_float_literal(operand) for operand in operands) and not any(
                _approx_call(operand) for operand in operands
            ):
                findings.append(
                    Finding(
                        path,
                        comparison.lineno,
                        "S1244",
                        "float equality in assert; use pytest.approx",
                    )
                )
                break
    return findings


def _numpy_random_imports(tree: ast.AST) -> set[str]:
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 0 or node.module != "numpy.random":
            continue
        imported_names.update(
            alias.asname or alias.name for alias in node.names if alias.name != "*"
        )
    return imported_names


def _random_call_name(call: ast.Call, imported_names: set[str]) -> str | None:
    function = call.func
    if isinstance(function, ast.Name) and function.id in imported_names:
        return function.id
    if not isinstance(function, ast.Attribute):
        return None
    random_module = function.value
    if not (
        isinstance(random_module, ast.Attribute)
        and random_module.attr == "random"
        and isinstance(random_module.value, ast.Name)
        and random_module.value.id == "np"
    ):
        return None
    return function.attr


def _bare_random_calls(tree: ast.AST, path: Path) -> list[Finding]:
    findings: list[Finding] = []
    imported_names = _numpy_random_imports(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _random_call_name(node, imported_names)
        if name is None:
            continue
        if name == "default_rng":
            has_seed = bool(node.args) or any(keyword.arg == "seed" for keyword in node.keywords)
            if has_seed:
                continue
        findings.append(
            Finding(
                path,
                node.lineno,
                "S6709",
                f"bare NumPy random call {name!r}; use a seeded Generator",
            )
        )
    return findings


def _comment_lines(source: str) -> set[int]:
    comments: set[int] = set()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    with suppress(tokenize.TokenError):
        comments.update(token.start[0] for token in tokens if token.type == tokenize.COMMENT)
    return comments


def _uncommented_passes(tree: ast.AST, source: str, path: Path) -> list[Finding]:
    comments = _comment_lines(source)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Pass):
            continue
        if node.lineno in comments or node.lineno - 1 in comments:
            continue
        findings.append(
            Finding(path, node.lineno, "S1186", "empty pass stub needs an explanatory comment")
        )
    return findings


def _check_file(path: Path) -> list[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        print(f"{path}: unable to inspect ({error})")
        return []
    return [
        *_assert_float_comparisons(tree, path),
        *_bare_random_calls(tree, path),
        *_uncommented_passes(tree, source, path),
    ]


def _python_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.add(path)
        elif path.is_dir():
            files.update(candidate for candidate in path.rglob("*.py") if candidate.is_file())
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Python files or directories (defaults to src, tests, scripts, and Large Experiments)",
    )
    args = parser.parse_args()
    paths = args.paths or [
        Path("src"),
        Path("tests"),
        Path("scripts"),
        Path("Large Experiments"),
    ]
    findings = [finding for path in _python_files(paths) for finding in _check_file(path)]
    for finding in findings:
        print(f"{finding.path}:{finding.line}:1: {finding.rule} {finding.message}")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
