"""pytest 配置"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-cuda", action="store_true", default=False, help="Run CUDA tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
