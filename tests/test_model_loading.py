import os

import pytest

from viva_tellurium.processes import _load_roadrunner


def test_missing_model_file_fails_loud(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as ei:
        _load_roadrunner(model_source='', model_file='does_not_exist.xml')
    msg = str(ei.value)
    assert 'does_not_exist.xml' in msg
    assert os.path.join(os.getcwd(), 'does_not_exist.xml') in msg
    assert os.getcwd() in msg
    assert 'absolute path' in msg
