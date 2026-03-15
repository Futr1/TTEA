from __future__ import annotations

import importlib


def has_dependency(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def import_torch():
    if not has_dependency("torch"):
        return None
    return importlib.import_module("torch")


def import_transformers():
    if not has_dependency("transformers"):
        return None
    return importlib.import_module("transformers")


def import_langchain_core():
    if has_dependency("langchain_core"):
        return importlib.import_module("langchain_core")
    if has_dependency("langchain"):
        return importlib.import_module("langchain")
    return None


def import_playwright_sync():
    if not has_dependency("playwright.sync_api"):
        return None
    return importlib.import_module("playwright.sync_api")


def import_gymnasium():
    if not has_dependency("gymnasium"):
        return None
    return importlib.import_module("gymnasium")


def import_datasets():
    if not has_dependency("datasets"):
        return None
    return importlib.import_module("datasets")


def import_evaluate():
    if not has_dependency("evaluate"):
        return None
    return importlib.import_module("evaluate")


def import_mauve():
    if has_dependency("mauve"):
        return importlib.import_module("mauve")
    if has_dependency("mauve_text"):
        return importlib.import_module("mauve_text")
    return None


def import_sacrebleu():
    if not has_dependency("sacrebleu"):
        return None
    return importlib.import_module("sacrebleu")


def import_rouge_score():
    if not has_dependency("rouge_score"):
        return None
    return importlib.import_module("rouge_score")
