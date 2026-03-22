import pytest
import os
from omop_llm import LLMClient
from typing import Callable, Dict, Any

from omop_graph import KnowledgeGraph
from omop_graph.oaklib_interface.omop_factory import OMOP_DATABASE_ENV_VAR
from omop_graph.oaklib_interface.omop_implementation import (
    OMOPAlchemyImplementation,
    ANNOTATIONS_DOMAINS_KEY,
    ANNOTATIONS_PARENT_ID_KEY,
    ANNOTATIONS_VOCABS_KEY,
    ANNOTATIONS_SPLIT_CHAR
)

@pytest.fixture
def cancers_named() -> tuple[tuple[str, tuple[int, str]], ...]:
    """Return cancers listed in Table A1 of the 2023 NSW Data dictionary with a name that is not just an organ. Each element in the returned tuple is composed of the following:
    <cancer in dictionary>, <expected concept id>, <expected concept name>
    
    https://www.cancer.nsw.gov.au/research-and-data/cancer-data-and-statistics/data-available-on-request/request-unit-record-data-for-research#Accordion
    """
    return tuple([
        ("Hodgkin lymphoma", (4038835, "Hodgkin's disease (clinical)")),
        ("Multiple myeloma", (437233, "Multiple myeloma")),
        ("Acute lymphoblastic leukaemia", (134305, "Acute lymphoid leukemia")),
        ("Acute myeloid leukaemia", (140352, "Acute myeloid leukemia, disease")),
        ("Myelodysplasia", (138994, "Myelodysplastic syndrome (clinical)")),
    ])

@pytest.fixture
def cancers_organ() -> tuple[tuple[str, tuple[int, str]], ...]:
    """Return cancers listed in Table A1 of the 2023 NSW Data dictionary with a name that is just an organ. Each element in the returned tuple is composed of the following:
    <cancer in dictionary>, <expected concept id>, <expected concept name>
    
    https://www.cancer.nsw.gov.au/research-and-data/cancer-data-and-statistics/data-available-on-request/request-unit-record-data-for-research#Accordion
    """
    return tuple([
        ("Ovarian cancer", (4181351, "Malignant neoplasm of ovary")),
        ("Placenta cancer", (4221190, "Gestational choriocarcinoma")),
        ("Prostate cancer", (4163261, "Malignant neoplasm of prostate")),
        ("Testicular cancer", (4177115, "Malignant tumor of testis")),
        ("Kidney cancer", (196653, "Malignant tumor of kidney")),
        ("Bladder cancer", (197508, "Malignant neoplasm of urinary bladder")),
        ("Eye cancer", (4246808, "Malignant tumor of ciliary body")),  # Eye cancer does not really exist
        ("Brain tumor", (443588, "Malignant neoplasm of brain")),  # Brain `cancer` does not really exist
        ("Thyroid cancer", (4178976, "Malignant tumor of thyroid gland")),
    ])

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
    def _get_instance(model: str) -> LLMClient:
        embedding_client = LLMClient(
            model=model,
            system_message="",
            api_base="http://ollama:11434/v1"  # TODO: Make this .env variable or something?
        )
        return embedding_client
    return _get_instance


class TestGrounding:              
    def test_grounding_cancers_named_no_embd(self, cancers_named, cancer_annotations, get_omop_implementation, subtests):
        
        omop: OMOPAlchemyImplementation = get_omop_implementation()
        for i, (input_text, (expected_concept_id, expected_concept_name)) in enumerate(cancers_named):

            expected_concept_id_with_prefix = f"OMOP:{expected_concept_id}"
            with subtests.test(msg=input_text, i=i):
                annotation_iterator = omop.annotate_text(
                    text=input_text,
                    text_embedding=None,
                    text_embedding_model=None,
                    embedding_client=None,
                    configuration=None,  # Not being tested at this stage
                    annotations=cancer_annotations,
                )

                all_predictions = list(annotation_iterator) # Just for debugging
                prediction = all_predictions[0]
                
                assert prediction.object_id is not None, f"Failed to ground '{input_text}' (returned None)"
                assert prediction.object_id == expected_concept_id_with_prefix, \
                    f"Expected:{expected_concept_id_with_prefix} [{expected_concept_name}]\nGot: {prediction.object_id} [{prediction.object_label}]"
                
    def test_grounding_cancers_organ_no_embd(self, cancers_organ, cancer_annotations, get_omop_implementation, subtests):
        omop: OMOPAlchemyImplementation = get_omop_implementation()
        for i, (input_text, (expected_concept_id, expected_concept_name)) in enumerate(cancers_organ):

            expected_concept_id_with_prefix = f"OMOP:{expected_concept_id}"
            with subtests.test(msg=input_text, i=i):
                annotation_iterator = omop.annotate_text(
                    text=input_text,
                    text_embedding=None,
                    text_embedding_model=None,
                    embedding_client=None,
                    configuration=None,  # Not being tested at this stage
                    annotations=cancer_annotations,
                )

                all_predictions = list(annotation_iterator) # Just for debugging
                prediction = all_predictions[0]
                
                assert prediction.object_id is not None, f"Failed to ground '{input_text}' (returned None)"
                assert prediction.object_id == expected_concept_id_with_prefix, \
                    f"Expected:{expected_concept_id_with_prefix} [{expected_concept_name}]\nGot: {prediction.object_id} [{prediction.object_label}]"

