import pytest

from src.models.asset_model import AssetType, QualityLevel, StylePreference
from src.utils.validators import AssetValidator, TextValidator, ValidationException


def test_text_validator_valid_description() -> None:
    description = "A detailed description of a sword with glowing runes."
    sanitized = TextValidator.validate_description(description)
    assert sanitized == description.strip(), "Valid description should be returned sanitized"


def test_text_validator_too_short() -> None:
    with pytest.raises(ValidationException, match="Description must be at least 10 characters"):
        TextValidator.validate_description("Short")


def test_text_validator_too_long() -> None:
    with pytest.raises(ValidationException, match="Description cannot exceed 2000 characters"):
        TextValidator.validate_description("x" * 2001)


def test_text_validator_security() -> None:
    with pytest.raises(ValidationException, match="Potential security threat detected"):
        TextValidator.validate_description("<script>alert('hack')</script>")


def test_asset_validator() -> None:
    assert AssetValidator.validate_asset_type("weapon") == AssetType.WEAPON, "Valid asset type should be returned"
    assert AssetValidator.validate_style_preference("realistic") == StylePreference.REALISTIC, (
        "Valid style should be returned"
    )
    assert AssetValidator.validate_quality_level("standard") == QualityLevel.STANDARD, (
        "Valid quality level should be returned"
    )


def test_asset_validator_invalid() -> None:
    with pytest.raises(ValidationException, match="Asset type must be one of"):
        AssetValidator.validate_asset_type("invalid")
    with pytest.raises(ValidationException, match="Style preference must be one of"):
        AssetValidator.validate_style_preference("invalid")
    with pytest.raises(ValidationException, match="Quality level must be one of"):
        AssetValidator.validate_quality_level("invalid")
