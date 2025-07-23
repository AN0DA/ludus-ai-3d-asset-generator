from src.models.asset_model import AssetMetadata, AssetType, FileFormat, GenerationStatus, QualityLevel, StylePreference


class TestAssetType:
    """Test suite for AssetType enum."""

    def test_asset_type_values(self) -> None:
        """Test that AssetType has expected values."""
        assert AssetType.WEAPON.value == "weapon"
        assert AssetType.ARMOR.value == "armor"
        assert AssetType.POTION.value == "potion"
        assert AssetType.TOOL.value == "tool"
        assert AssetType.ENVIRONMENT.value == "environment"

    def test_asset_type_enum_membership(self) -> None:
        """Test that all expected values are in AssetType enum."""
        expected_types = {"weapon", "armor", "potion", "tool", "environment"}
        actual_types = {member.value for member in AssetType}
        assert actual_types == expected_types


class TestStylePreference:
    """Test suite for StylePreference enum."""

    def test_style_preference_values(self) -> None:
        """Test that StylePreference has expected values."""
        assert StylePreference.REALISTIC.value == "realistic"
        assert StylePreference.SCULPTURE.value == "sculpture"

    def test_style_preference_enum_membership(self) -> None:
        """Test that all expected values are in StylePreference enum."""
        expected_styles = {"realistic", "sculpture"}
        actual_styles = {member.value for member in StylePreference}
        assert actual_styles == expected_styles


class TestFileFormat:
    """Test suite for FileFormat enum."""

    def test_file_format_values(self) -> None:
        """Test that FileFormat has expected values."""
        assert FileFormat.GLB.value == "glb"
        assert FileFormat.FBX.value == "fbx"
        assert FileFormat.OBJ.value == "obj"
        assert FileFormat.USDZ.value == "usdz"

    def test_file_format_enum_membership(self) -> None:
        """Test that all expected values are in FileFormat enum."""
        expected_formats = {"glb", "fbx", "obj", "usdz"}
        actual_formats = {member.value for member in FileFormat}
        assert actual_formats == expected_formats


class TestQualityLevel:
    """Test suite for QualityLevel enum."""

    def test_quality_level_values(self) -> None:
        """Test that QualityLevel has expected values."""
        assert QualityLevel.DRAFT.value == "draft"
        assert QualityLevel.STANDARD.value == "standard"
        assert QualityLevel.HIGH.value == "high"
        assert QualityLevel.ULTRA.value == "ultra"

    def test_quality_level_enum_membership(self) -> None:
        """Test that all expected values are in QualityLevel enum."""
        expected_levels = {"draft", "standard", "high", "ultra"}
        actual_levels = {member.value for member in QualityLevel}
        assert actual_levels == expected_levels


class TestGenerationStatus:
    """Test suite for GenerationStatus enum."""

    def test_generation_status_values(self) -> None:
        """Test that GenerationStatus has expected values."""
        assert GenerationStatus.PENDING.value == "pending"
        assert GenerationStatus.IN_PROGRESS.value == "in_progress"
        assert GenerationStatus.COMPLETED.value == "completed"
        assert GenerationStatus.FAILED.value == "failed"
        assert GenerationStatus.CANCELLED.value == "cancelled"
        assert GenerationStatus.RATE_LIMITED.value == "rate_limited"
        assert GenerationStatus.RETRYING.value == "retrying"

    def test_generation_status_enum_membership(self) -> None:
        """Test that all expected values are in GenerationStatus enum."""
        expected_statuses = {"pending", "in_progress", "completed", "failed", "cancelled", "rate_limited", "retrying"}
        actual_statuses = {member.value for member in GenerationStatus}
        assert actual_statuses == expected_statuses


class TestAssetMetadata:
    """Test suite for AssetMetadata model."""

    def test_asset_metadata_creation(self) -> None:
        """Test that AssetMetadata can be created with valid data."""
        metadata = AssetMetadata(
            name="Test Sword",
            original_description="A sharp medieval sword",
            enhanced_description={"enhanced": "A masterfully crafted medieval sword"},
            asset_type=AssetType.WEAPON,
            style_preferences=[StylePreference.REALISTIC],
            quality_level=QualityLevel.HIGH,
            generation_service="meshy",
            session_id="test-session-123",
            metadata={"tags": ["medieval", "weapon"]},
        )

        assert metadata.name == "Test Sword"
        assert metadata.original_description == "A sharp medieval sword"
        assert metadata.enhanced_description == {"enhanced": "A masterfully crafted medieval sword"}
        assert metadata.asset_type == AssetType.WEAPON
        assert metadata.style_preferences == [StylePreference.REALISTIC]
        assert metadata.quality_level == QualityLevel.HIGH
        assert metadata.generation_service == "meshy"
        assert metadata.session_id == "test-session-123"
        assert metadata.metadata == {"tags": ["medieval", "weapon"]}

        # Test that asset_id is automatically generated
        assert metadata.asset_id is not None
        assert isinstance(metadata.asset_id, str)
        assert len(metadata.asset_id) > 0

    def test_asset_metadata_with_custom_asset_id(self) -> None:
        """Test that AssetMetadata accepts custom asset_id."""
        custom_id = "custom-asset-id-123"
        metadata = AssetMetadata(
            asset_id=custom_id,
            name="Test Armor",
            original_description="A heavy armor set",
            enhanced_description={"enhanced": "A robust armor set"},
            asset_type=AssetType.ARMOR,
            style_preferences=[StylePreference.SCULPTURE],
            quality_level=QualityLevel.STANDARD,
            generation_service="meshy",
            session_id="test-session-456",
            metadata={},
        )

        assert metadata.asset_id == custom_id

    def test_asset_metadata_multiple_style_preferences(self) -> None:
        """Test that AssetMetadata accepts multiple style preferences."""
        metadata = AssetMetadata(
            name="Test Tool",
            original_description="A useful tool",
            enhanced_description={"enhanced": "A well-designed tool"},
            asset_type=AssetType.TOOL,
            style_preferences=[StylePreference.REALISTIC, StylePreference.SCULPTURE],
            quality_level=QualityLevel.DRAFT,
            generation_service="meshy",
            session_id="test-session-789",
            metadata={},
        )

        assert len(metadata.style_preferences) == 2
        assert StylePreference.REALISTIC in metadata.style_preferences
        assert StylePreference.SCULPTURE in metadata.style_preferences

    def test_asset_metadata_empty_style_preferences(self) -> None:
        """Test that AssetMetadata accepts empty style preferences list."""
        metadata = AssetMetadata(
            name="Test Environment",
            original_description="A vast landscape",
            enhanced_description={"enhanced": "An expansive landscape"},
            asset_type=AssetType.ENVIRONMENT,
            style_preferences=[],
            quality_level=QualityLevel.ULTRA,
            generation_service="meshy",
            session_id="test-session-000",
            metadata={},
        )

        assert metadata.style_preferences == []

    def test_asset_metadata_complex_enhanced_description(self) -> None:
        """Test that AssetMetadata accepts complex enhanced_description."""
        complex_enhanced = {
            "enhanced": "A detailed description",
            "technical_specs": {"material": "steel", "weight": "2.5kg", "length": "90cm"},
            "visual_elements": ["engravings", "leather_grip", "pommel"],
        }

        metadata = AssetMetadata(
            name="Complex Weapon",
            original_description="A weapon",
            enhanced_description=complex_enhanced,
            asset_type=AssetType.WEAPON,
            style_preferences=[StylePreference.REALISTIC],
            quality_level=QualityLevel.HIGH,
            generation_service="meshy",
            session_id="test-session-complex",
            metadata={"complexity": "high"},
        )

        assert metadata.enhanced_description == complex_enhanced
        assert metadata.enhanced_description["technical_specs"]["material"] == "steel"
        assert "engravings" in metadata.enhanced_description["visual_elements"]

    def test_asset_metadata_serialization(self) -> None:
        """Test that AssetMetadata can be serialized to dict."""
        metadata = AssetMetadata(
            name="Serialization Test",
            original_description="Test description",
            enhanced_description={"enhanced": "Enhanced test description"},
            asset_type=AssetType.POTION,
            style_preferences=[StylePreference.REALISTIC],
            quality_level=QualityLevel.STANDARD,
            generation_service="test_service",
            session_id="test-session-serial",
            metadata={"test": True},
        )

        # Convert to dict (Pydantic model_dump)
        data_dict = metadata.model_dump()

        assert isinstance(data_dict, dict)
        assert data_dict["name"] == "Serialization Test"
        assert data_dict["asset_type"] == "potion"
        assert data_dict["style_preferences"] == ["realistic"]
        assert data_dict["quality_level"] == "standard"
        assert data_dict["metadata"]["test"] is True
