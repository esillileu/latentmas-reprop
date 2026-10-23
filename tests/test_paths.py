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
    assert resolver.storage_dir.name == "storage"
    assert resolver.experiments_dir.name == "experiments"
    assert resolver.tensors_dir.name == "tensors"
    assert resolver.mlflow_dir.name == "mlflow"
    assert resolver.artifacts_dir.name == "artifacts"
    assert resolver.medqa_json_path.name == "medqa.json"
    assert resolver.medqa_json_path.is_file()


def test_path_resolver_safe_cache_path(tmp_path):
    resolver = PathResolver(tmp_path)
    safe_runtime = resolver.get_cache_layer_dir("runtime")
    assert resolver.is_safe_cache_path(safe_runtime)

    # Protected paths must NOT be safe cache paths
    assert not resolver.is_safe_cache_path(resolver.storage_dir)
    assert not resolver.is_safe_cache_path(resolver.experiments_dir)
    assert not resolver.is_safe_cache_path(resolver.mlflow_dir)
    assert not resolver.is_safe_cache_path(resolver.data_dir)
    assert not resolver.is_safe_cache_path(resolver.root)


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


def test_path_resolver_nested_config_without_extension(tmp_path):
    resolver = PathResolver(tmp_path)
    config = resolver.configs_dir / "secret_digit" / "lm_q3-0.6_l0.yaml"
    config.parent.mkdir(parents=True)
    config.touch()

    resolved = resolver.resolve_config_path("secret_digit/lm_q3-0.6_l0")

    assert resolved == config.resolve()
