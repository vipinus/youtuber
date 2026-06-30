from pathlib import Path
from src.config import Config
from src.llm import FakeLLM
from src.context import Ctx


def test_ctx_job_path(tmp_path):
    ctx = Ctx(slug="roswell-1947", config=Config({}), root=tmp_path, llm=FakeLLM())
    assert ctx.job == tmp_path / "roswell-1947"
