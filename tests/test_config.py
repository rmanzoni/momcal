import pytest, yaml
from scalesmear.config import load_config, physics_hash


def test_defaults_env_and_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("NT", "/x/y")
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(dict(inputs=dict(data="${NT}/rs.root", mc=["${NT}/sig.root"]))))
    c = load_config(str(p), ["fit.seed=7", "model.start.s0_2=-2e-3"])
    assert c["inputs"]["data"] == ["/x/y/rs.root"] and c["fit"]["seed"] == 7 and c["model"]["start"]["s0_2"] == -2e-3
    assert c["model"]["eta_edges"] == [0.0, 0.7, 1.2, 2.5]
    c2 = load_config(str(p)); assert physics_hash(c) != physics_hash(c2)


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(dict(inputs=dict(data="a", mc="b"), fitt={})))
    with pytest.raises(KeyError):
        load_config(str(p))
