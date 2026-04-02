import pytest
import os
from omop_llm import LLMClient
from typing import Callable, Dict, Any, Optional

from omop_graph.oaklib_interface.omop_factory import OMOP_DATABASE_ENV_VAR
from omop_graph.oaklib_interface.omop_implementation import (
    OMOPAlchemyImplementation,
    ANNOTATIONS_DOMAINS_KEY,
    ANNOTATIONS_PARENT_ID_KEY,
    ANNOTATIONS_VOCABS_KEY,
    ANNOTATIONS_SPLIT_CHAR
)

OLLAMA_API_BASE_ENV_VAR = "OMOP_OLLAMA_API_BASE"

NAMES_CASES = [
    pytest.param("Hodgkin lymphoma", 4038835, "Hodgkin's disease (clinical)", id="hodgkin-lymphoma"),
    pytest.param("Multiple myeloma", 437233, "Multiple myeloma", id="multiple-myeloma"),
    pytest.param("Acute lymphoblastic leukaemia", 134305, "Acute lymphoid leukemia", id="acute-lymphoblastic-leukaemia"),
    pytest.param("Acute myeloid leukaemia",140352,"Acute myeloid leukemia, disease", id="acute-myeloid-leukaemia"),
    pytest.param("Myelodysplasia", 138994, "Myelodysplastic syndrome (clinical)", id="myelodysplasia",),
]

ORGAN_CASES = [
    pytest.param("Ovarian cancer", 4181351, "Malignant neoplasm of ovary", id="ovarian-cancer"),
    pytest.param("Placenta cancer", 4221190, "Gestational choriocarcinoma", id="placenta-cancer"),
    pytest.param("Prostate cancer", 4163261, "Malignant neoplasm of prostate", id="prostate-cancer"),
    pytest.param("Testicular cancer", 4177115, "Malignant tumor of testis", id="testicular-cancer"),
    pytest.param("Kidney cancer", 196653, "Malignant tumor of kidney", id="kidney-cancer"),
    pytest.param("Bladder cancer", 197508, "Malignant neoplasm of urinary bladder", id="bladder-cancer"),
    pytest.param("Eye cancer", 4246808, "Malignant tumor of ciliary body", id="eye-cancer"),
    pytest.param("Brain tumor", 443588, "Malignant neoplasm of brain", id="brain-tumor"),
    pytest.param("Thyroid cancer", 4178976, "Malignant tumor of thyroid gland", id="thyroid-cancer"),
]

class LinkMLAnnotationMimic:
    "Mimics the LinkML Annotation type. Simply requires the value property."
    def __init__(
        self,
        value
    ):
        self.value = value

@pytest.fixture
def cancer_annotations() -> Dict[str, Any]:
    """Returns the annotation object that would be found in a LinkML.
    These include restrictions about:
    - parent_id: the ancestor of the concept
    - vocabs: allowed vocabs 
    - domanins: allowed domains
    """
    return {
        ANNOTATIONS_DOMAINS_KEY: LinkMLAnnotationMimic(ANNOTATIONS_SPLIT_CHAR.join(["Condition"])),
        ANNOTATIONS_PARENT_ID_KEY: LinkMLAnnotationMimic(ANNOTATIONS_SPLIT_CHAR.join(["OMOP:443392"])), # Malignant neoplastic disease
        ANNOTATIONS_VOCABS_KEY: LinkMLAnnotationMimic(ANNOTATIONS_SPLIT_CHAR.join(["SNOMED", "ICD10CM", "HemOnc"]))
    }


@pytest.fixture(scope="session")
def get_omop_implementation() -> Callable[[], OMOPAlchemyImplementation]:
    def _get_instance() -> OMOPAlchemyImplementation:
        engine_string = os.getenv(OMOP_DATABASE_ENV_VAR)
        omop = OMOPAlchemyImplementation(engine_string=engine_string)
        return omop
    return _get_instance


@pytest.fixture(scope="session")
def get_embedding_model_instance():
    def _get_instance(model: str, api_base: Optional[str] = None) -> LLMClient:
        ollama_api_base = api_base or os.getenv(OLLAMA_API_BASE_ENV_VAR)
        assert ollama_api_base is not None, (
            f"{OLLAMA_API_BASE_ENV_VAR} environment variable must be set to use embedding model "
            "or api_base must be passed explicitly"
        )
        embedding_client = LLMClient(
            model=model,
            system_message="",
            api_base=ollama_api_base
        )
        return embedding_client
    return _get_instance


class TestGrounding:              
    @pytest.mark.parametrize("input_text, expected_concept_id, expected_concept_name", NAMES_CASES)
    def test_grounding_cancers_named_no_embd(self, input_text, expected_concept_id, expected_concept_name, cancer_annotations, get_omop_implementation):
        omop: OMOPAlchemyImplementation = get_omop_implementation()
        expected_concept_id_with_prefix = f"OMOP:{expected_concept_id}"
        annotation_iterator = omop.annotate_text(
            text=input_text,
            text_embedding=None,
            text_embedding_model=None,
            embedding_client=None,
            configuration=None,
            annotations=cancer_annotations,
        )

        all_predictions = list(annotation_iterator)
        prediction = all_predictions[0]

        assert prediction.object_id is not None, f"Failed to ground '{input_text}' (returned None)"
        assert prediction.object_id == expected_concept_id_with_prefix, (
            f"Expected:{expected_concept_id_with_prefix} [{expected_concept_name}]\n"
            f"Got: {prediction.object_id} [{prediction.object_label}]"
        )

    @pytest.mark.parametrize("input_text, expected_concept_id, expected_concept_name", ORGAN_CASES)
    def test_grounding_cancers_organ_no_embd(self, input_text, expected_concept_id, expected_concept_name, cancer_annotations, get_omop_implementation):
        omop: OMOPAlchemyImplementation = get_omop_implementation()
        expected_concept_id_with_prefix = f"OMOP:{expected_concept_id}"
        annotation_iterator = omop.annotate_text(
            text=input_text,
            text_embedding=None,
            text_embedding_model=None,
            embedding_client=None,
            configuration=None,
            annotations=cancer_annotations,
        )

        all_predictions = list(annotation_iterator)
        prediction = all_predictions[0]

        assert prediction.object_id is not None, f"Failed to ground '{input_text}' (returned None)"
        assert prediction.object_id == expected_concept_id_with_prefix, (
            f"Expected:{expected_concept_id_with_prefix} [{expected_concept_name}]\n"
            f"Got: {prediction.object_id} [{prediction.object_label}]"
        )

    @pytest.mark.parametrize("input_text, expected_concept_id, expected_concept_name", NAMES_CASES)
    def test_grounding_cancers_named_with_embd(self, input_text, expected_concept_id, expected_concept_name, cancer_annotations, get_omop_implementation, get_embedding_model_instance):
        omop: OMOPAlchemyImplementation = get_omop_implementation()
        embedding_client = get_embedding_model_instance(model="omop-embedding-model")
        expected_concept_id_with_prefix = f"OMOP:{expected_concept_id}"
        annotation_iterator = omop.annotate_text(
            text=input_text,
            text_embedding=None,
            text_embedding_model="omop-embedding-model",
            embedding_client=embedding_client,
            configuration=None,
            annotations=cancer_annotations,
        )

        all_predictions = list(annotation_iterator)
        prediction = all_predictions[0]

        assert prediction.object_id is not None, f"Failed to ground '{input_text}' (returned None)"
        assert prediction.object_id == expected_concept_id_with_prefix, (
            f"Expected:{expected_concept_id_with_prefix} [{expected_concept_name}]\n"
            f"Got: {prediction.object_id} [{prediction.object_label}]"
        )
