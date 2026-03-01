from src.modules.utils import (
    DatasetPaths,
    ModelPaths,
    OptunaPaths,
    TextVectorizerConfig,
)


def test_dataset_paths_enum():
    """Test DatasetPaths enum values."""
    assert hasattr(DatasetPaths, "RAW_DATA")
    # Enum values are strings (str, Enum inheritance converts Path to str)
    assert isinstance(DatasetPaths.RAW_DATA.value, str)
    assert str(DatasetPaths.RAW_DATA.value).endswith(".csv")


def test_model_paths_enum():
    """Test ModelPaths enum values."""
    # Check that all required paths exist
    assert hasattr(ModelPaths, "MODEL_BUILDER_CONFIG")
    assert hasattr(ModelPaths, "MODEL_TRAINER_CONFIG")
    assert hasattr(ModelPaths, "TRAINED_MODEL")
    assert hasattr(ModelPaths, "INFERENCE_MODEL")
    assert hasattr(ModelPaths, "TRANSFORMER_MODEL")

    # Check that paths are strings (str, Enum inheritance converts Path to str)
    assert isinstance(ModelPaths.MODEL_BUILDER_CONFIG.value, str)
    assert isinstance(ModelPaths.MODEL_TRAINER_CONFIG.value, str)
    assert isinstance(ModelPaths.TRAINED_MODEL.value, str)
    assert isinstance(ModelPaths.INFERENCE_MODEL.value, str)
    assert isinstance(ModelPaths.TRANSFORMER_MODEL.value, str)

    # Check file extensions
    assert str(ModelPaths.MODEL_BUILDER_CONFIG.value).endswith(".json")
    assert str(ModelPaths.MODEL_TRAINER_CONFIG.value).endswith(".json")
    assert str(ModelPaths.TRAINED_MODEL.value).endswith(".keras")
    assert str(ModelPaths.INFERENCE_MODEL.value).endswith(".keras")
    assert str(ModelPaths.TRANSFORMER_MODEL.value).endswith(".keras")


def test_optuna_paths_enum():
    """Test OptunaPaths enum values."""
    assert hasattr(OptunaPaths, "OPTUNA_CONFIG")
    assert hasattr(OptunaPaths, "OPTUNA_MODEL")

    # Enum values are strings (str, Enum inheritance converts Path to str)
    assert isinstance(OptunaPaths.OPTUNA_CONFIG.value, str)
    assert isinstance(OptunaPaths.OPTUNA_MODEL.value, str)


def test_text_vectorizer_config_enum():
    """Test TextVectorizerConfig enum values."""
    assert hasattr(TextVectorizerConfig, "max_tokens")
    assert hasattr(TextVectorizerConfig, "output_sequence_length")

    # Check values are positive integers
    assert TextVectorizerConfig.max_tokens.value > 0
    assert TextVectorizerConfig.output_sequence_length.value > 0

    # Check specific default values
    assert TextVectorizerConfig.max_tokens.value == 20000
    assert TextVectorizerConfig.output_sequence_length.value == 500


def test_paths_structure():
    """Test that all paths follow expected structure."""
    # All model paths should start with "src/"
    assert str(ModelPaths.MODEL_BUILDER_CONFIG.value).startswith("src/")
    assert str(ModelPaths.MODEL_TRAINER_CONFIG.value).startswith("src/")
    assert str(ModelPaths.TRAINED_MODEL.value).startswith("src/")

    # Configuration paths should be in "src/configurations/"
    assert "configurations" in str(ModelPaths.MODEL_BUILDER_CONFIG.value)
    assert "configurations" in str(ModelPaths.MODEL_TRAINER_CONFIG.value)
    assert "configurations" in str(OptunaPaths.OPTUNA_CONFIG.value)

    # Model paths should be in "src/models/"
    assert "models" in str(ModelPaths.TRAINED_MODEL.value)
    assert "models" in str(ModelPaths.INFERENCE_MODEL.value)
    assert "models" in str(OptunaPaths.OPTUNA_MODEL.value)


def test_enum_uniqueness():
    """Test that enum values are unique."""
    # Check ModelPaths
    model_values = [item.value for item in ModelPaths]
    assert len(model_values) == len(
        set(model_values)
    ), "ModelPaths has duplicate values"

    # Check OptunaPaths
    optuna_values = [item.value for item in OptunaPaths]
    assert len(optuna_values) == len(
        set(optuna_values)
    ), "OptunaPaths has duplicate values"


def test_enum_string_conversion():
    """Test that enum values can be converted to strings."""
    # Test that Path objects can be converted to strings
    assert isinstance(str(DatasetPaths.RAW_DATA.value), str)
    assert isinstance(str(ModelPaths.TRAINED_MODEL.value), str)
    assert isinstance(str(OptunaPaths.OPTUNA_CONFIG.value), str)


def test_enum_membership():
    """Test enum membership checks."""
    # Check that specific values are in the enums
    assert "RAW_DATA" in DatasetPaths.__members__
    assert "TRAINED_MODEL" in ModelPaths.__members__
    assert "OPTUNA_CONFIG" in OptunaPaths.__members__
    assert "max_tokens" in TextVectorizerConfig.__members__


def test_path_components():
    """Test that paths have expected components."""
    # Test that configurations are in the right directory
    config_path = str(ModelPaths.MODEL_BUILDER_CONFIG.value)
    assert config_path.startswith("src")
    assert "configurations" in config_path

    # Test that models are in the right directory
    model_path = str(ModelPaths.TRAINED_MODEL.value)
    assert model_path.startswith("src")
    assert "models" in model_path
