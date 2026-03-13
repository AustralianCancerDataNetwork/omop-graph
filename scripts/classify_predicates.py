import pathlib as pl
import pandas as pd
import os

from omop_graph.oaklib_interface.omop_factory import OMOP_DATABASE_ENV_VAR, omop_resource
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation
from omop_graph.graph import KnowledgeGraph, PredicateKind

PATH_TO_CSV = pl.Path(__file__).parent.parent / "docs" / "predicate_classification.csv"


def classify_all_predicates_generator(kg: KnowledgeGraph):
    def get_all_relationship_ids(_kg: KnowledgeGraph):
        from omop_alchemy.cdm.model import Relationship
        from sqlalchemy import select

        q = select(Relationship.relationship_id, Relationship.relationship_name).distinct()
        result = _kg.session.execute(q).fetchall()
        for row in result:
            yield row[0], row[1]
    
    for relationship_id, relationship_name in get_all_relationship_ids(kg):
        p = kg.predicate(relationship_id)
        kind = p.classify_predicate()
        yield relationship_id, relationship_name, kind

def csv_export(classification_generator):
    col_names = ["r_id", "r_name", "old_cat", "parent_cat", "subcategory", "classification_notes", "additional_info", "link"]
    rows = []

    for relationship_id, relationship_name, kind in classification_generator:
        rows.append({
            "r_id": relationship_id,
            "r_name": relationship_name,
            "old_cat": str(kind).strip("PredicateKind."),
            "parent_cat": None,
            "subcategory": None,
            "classification_notes": None,
            "additional_info": None,
            "link": None,
        })

    df = pd.DataFrame(rows, columns=col_names)
    df.to_csv(PATH_TO_CSV, index=False)

def set_unused_predicates_to_uncategorised(kg: KnowledgeGraph):
    from omop_alchemy.cdm.model import Concept_Relationship
    from sqlalchemy import select

    df = pd.read_csv(PATH_TO_CSV)

    used_predicates_query = select(Concept_Relationship.relationship_id).distinct()
    used_predicates = set([row[0] for row in kg.session.execute(used_predicates_query).fetchall()])
    mask = ~df["r_id"].isin(used_predicates)

    # 2. Update multiple columns at once for those specific rows
    df.loc[mask, ["parent_cat", "subcategory", "classification_notes"]] = [
        "Uncategorised", 
        "Uncategorised",
        "Unused in OMOP CDM"
    ]
    df.to_csv(PATH_TO_CSV, index=False)
    

if __name__ == "__main__":

    resource = omop_resource(url=os.getenv(OMOP_DATABASE_ENV_VAR))
    kg = OMOPAlchemyImplementation(resource=resource).kg

    if False:
        #  For initial generation of the CSV file for manual review and categorisation
        generator = classify_all_predicates_generator(kg)
        csv_export(generator)

    set_unused_predicates_to_uncategorised(kg)