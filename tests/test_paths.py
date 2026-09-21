from latentmas_reprop.infrastructure.paths import PathResolver, get_path_resolver


def test_path_resolver_singleton():
    resolver = get_path_resolver()
    assert resolver is not None
    assert isinstance(resolver, PathResolver)
    assert resolver.root.is_dir()
    assert (resolver.root / "pyproject.toml").is_file()


def test_path_resolver_directories():
    resolver = get_path_resolver()
    assert resolver.cache_dir.name == ".cache"
    assert resolver.data_dir.name == "data"
    assert resolver.medqa_json_path.name == "medqa.json"
    assert resolver.medqa_json_path.is_file()


def test_path_resolver_cache_layers():
    resolver = get_path_resolver()
    layer_path = resolver.get_cache_layer_dir("models/realign")
    assert layer_path.is_dir()
    assert ".cache" in str(layer_path)
    assert "models/realign" in str(layer_path)


def test_path_resolver_relative_resolution():
    resolver = get_path_resolver()
    resolved = resolver.resolve("data/medqa.json")
    assert resolved == resolver.medqa_json_path
    assert resolved.is_absolute()


def test_path_resolver_configs():
    resolver = get_path_resolver()
    assert resolver.configs_dir.is_dir()
    resolved = resolver.resolve_config_path("lm_q30.6_gsm8k")
    assert resolved.is_file()
    assert resolved.name == "lm_q30.6_gsm8k.yaml"
