import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, Session

from orm_loader.helpers import create_db, bulk_load_context
from orm_loader.loaders.loader_interface import PandasLoader
from orm_loader.helpers.metadata import Base
from omop_alchemy.cdm.handlers import (
    install_fulltext_columns,
    populate_fulltext_columns,
)
from omop_alchemy.cdm.base import CDMTableBase
from omop_alchemy.cdm.model.health_system import Location, Care_Site, Provider, Visit_Occurrence
from omop_alchemy.cdm.model.clinical import (
    Person, 
    Condition_Occurrence,
    Death, 
    Measurement, 
)
from omop_alchemy.cdm.model.structural.episode import Episode
from omop_alchemy.cdm.model.structural.episode_event import Episode_Event
from omop_alchemy.cdm.model.derived import Observation_Period
from omop_alchemy.cdm.model.vocabulary import (
    Domain,
    Vocabulary,
    Concept_Class,
    Relationship,
    Concept,
    Concept_Ancestor,
    Concept_Relationship,
    Concept_Synonym
)
from omop_graph.extensions.omop_alchemy import RelationshipClass, RelationshipMapping
from typing import Annotated, Union, Optional
import pandas as pd
import os
from pathlib import Path
from random import randint, choice
import numpy as np
from datetime import date, timedelta
from dotenv import load_dotenv
import typer
import logging

app = typer.Typer()
logger = logging.getLogger(__name__)

ATHENA_INITIAL_LOAD = [
    Domain,
    Vocabulary,
    Concept_Class,
    Relationship,
    Concept
]


ATHENA_SUBSEQUENT_LOAD = [
    Concept_Ancestor,
    Concept_Relationship,
    Concept_Synonym,
]

ATHENA_RELATIONSHIP_CLASSIFICATION_LOAD = [
    RelationshipClass,
    RelationshipMapping
]

def configure_logging_level(verbosity: int, reduce_logging: bool = True) -> None:
    """Configure global logging."""
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    log_level = level_map.get(min(verbosity, 2), logging.DEBUG)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    if reduce_logging:
        existing_loggers = [
            logging.getLogger(name) for name in logging.root.manager.loggerDict
        ]
        exempt_loggers = ["omop_graph", "omop_emb"]
        for logger_instance in existing_loggers:
            if not any(
                logger_instance.name.startswith(exempt) for exempt in exempt_loggers
            ):
                logger_instance.setLevel(logging.WARNING)


def _enable_fulltext_sidecars(engine: sa.Engine, regconfig: str) -> None:
    install_fulltext_columns(engine)
    populate_fulltext_columns(engine, regconfig=regconfig)

def _populate_reference_data(
    session: Session,
    avail_country: list[int],
    avail_place_of_service: list[int],
    avail_provider: list[int],
    avail_gender: list[int],
):
    
    loc_ids = Location.allocator(session)
    cs_ids = Care_Site.allocator(session)
    pro_ids = Provider.allocator(session)
    
    location_data = [{'location_id': loc_ids.next(), 'country_concept_id': choice(avail_country), 'city': f'City {idx}'} for idx in range(10)]
    locations = [Location(**row) for row in location_data]
    care_site_data = [{'care_site_id': cs_ids.next(), 'care_site_name': f'Care Site {idx}', 'location_id': choice(locations).location_id, 'place_of_service_concept_id': choice(avail_place_of_service)} for idx in range(30)]
    care_sites = [Care_Site(**row) for row in care_site_data]
    provider_data = [{'provider_id': pro_ids.next(), 'specialty_concept_id': choice(avail_provider), 'gender_concept_id': choice(avail_gender), 'care_site_id': choice(care_sites).care_site_id} for _ in range(50)]
    providers = [Provider(**row) for row in provider_data]

    session.add_all(locations)
    session.add_all(care_sites)
    session.add_all(providers)
    session.commit()

    return locations, care_sites, providers

def _populate_people_and_visits(
        session: Session,
        care_sites: list[Care_Site],
        avail_gender: list[int],
        avail_race: list[int],
        avail_ethnicity: list[int],
        avail_place_of_service: list[int],
    ):
    
    person_ids = Person.allocator(session)
    visit_ids = Visit_Occurrence.allocator(session)
    
    person_data = [{'person_id': person_ids.next(), 'year_of_birth': randint(1950, 2020), 'month_of_birth': randint(1, 12), 'gender_concept_id':choice(avail_gender), 'race_concept_id':choice(avail_race), 'ethnicity_concept_id':choice(avail_ethnicity)} for idx in range(1000)]
    people = [Person(**row) for row in person_data]

    visits = []
    for person in people:
        cs = choice(care_sites)
        visit_num = randint(1, 3)
        for v in range(visit_num):
            days_delay = randint(0, 365)
            visit_date = date(2020, 1, 1) + timedelta(days_delay)
            visit = Visit_Occurrence(
                visit_occurrence_id=visit_ids.next(),
                person_id=person.person_id,
                care_site_id=cs.care_site_id,
                visit_concept_id=choice(avail_place_of_service),
                visit_start_date=visit_date,
                visit_end_date=visit_date,
            )
            visits.append(visit)
    session.add_all(people)
    session.add_all(visits)
    session.commit()
    return people, visits

def _populate_observation_periods(
        session: Session,
        avail_types: list[int],
    ):
    op_ids = Observation_Period.allocator(session)
    deaths = []
    rows = (
        session.query(
            Visit_Occurrence.person_id,
            sa.func.min(Visit_Occurrence.visit_start_date).label("start"),
            sa.func.max(Visit_Occurrence.visit_end_date).label("end"),
            Death.death_date,
            Observation_Period.observation_period_id
        )
        .join(Death, Death.person_id==Visit_Occurrence.person_id, isouter=True)
        .join(Observation_Period, Observation_Period.person_id==Visit_Occurrence.person_id, isouter=True)
        .filter(Observation_Period.observation_period_id==None)
        .group_by(Visit_Occurrence.person_id)
        .all()
    )
    obs = []
    for idx, r in enumerate(rows):
        deceased = np.random.choice([True, False], p=[0.05, 0.95])
        if deceased:
            death_date = r.end + timedelta(days=randint(1, 365))
            deaths.append(
                Death(
                    person_id=r.person_id,
                    death_date=death_date,
                    death_type_concept_id=choice(avail_types),
                )
            )
            obs_end = death_date
        else:
            obs_end = r.end
        obs.append(
            Observation_Period(
                observation_period_id=op_ids.next(),
                person_id=r.person_id,
                observation_period_start_date=r.start,
                observation_period_end_date=obs_end,
                period_type_concept_id=choice(avail_types),
            )
        )
    session.add_all(deaths)
    session.add_all(obs)
    session.commit()
    return obs

def _populate_conditions_and_modifiers(
        session: Session,
        staging_sets: dict[str, pd.DataFrame],
        cancers: list[int],
        avail_types: list[int],
    ):
    cond_ids = Condition_Occurrence.allocator(session)
    meas_ids = Measurement.allocator(session)
    ep_ids   = Episode.allocator(session)
    rows = (
        session.query(
            Observation_Period, Death, Condition_Occurrence
        )
        .join(Death, Observation_Period.person_id==Death.person_id, isouter=True)
        .join(Condition_Occurrence, Observation_Period.person_id==Condition_Occurrence.person_id, isouter=True)
        .all()
    )
    conditions = []
    measurements = []
    episodes = []
    episode_events = []
    for obs, death, condition in rows:
        if condition:
            continue
        t = choice(list(staging_sets['T'].concept_id))
        n = choice(list(staging_sets['N'].concept_id))
        m = choice(list(staging_sets['M'].concept_id))
        # don't worry abt overall stage for now as it should be calculated
        condition_concept = choice(cancers)
        condition = Condition_Occurrence(
            condition_occurrence_id=cond_ids.next(),
            condition_concept_id = condition_concept,
            condition_start_date = obs.observation_period_start_date,
            condition_type_concept_id = choice(avail_types),
            person_id = obs.person_id,
            condition_status_concept_id = 32902
        )
        conditions.append(condition)
        episode = Episode(
            episode_id=ep_ids.next(),
            person_id=obs.person_id,
            episode_concept_id=32533,  # Episode of care
            episode_object_concept_id=condition.condition_concept_id,
            episode_start_date=condition.condition_start_date,
            episode_end_date=(
                death.death_date if death else obs.observation_period_end_date
            ),
            episode_type_concept_id=choice(avail_types),  # EHR / registry / derived
        )
        episodes.append(episode)

        for stage in [t, n, m]:
            measurement = Measurement(
                person_id = obs.person_id,
                measurement_id = meas_ids.next(),
                measurement_concept_id = stage,
                measurement_event_id = condition.condition_occurrence_id,
                meas_event_field_concept_id = 1147127, # condition_occurrence.condition_occurrence_id
                measurement_date = condition.condition_start_date,
                measurement_type_concept_id = choice(avail_types),
                value_as_number = 1
            )
            measurements.append(measurement)
            episode_events.append(
                Episode_Event(
                    episode_id=episode.episode_id,
                    event_id=measurement.measurement_id,
                    episode_event_field_concept_id=1147138,  # measurement.measurement_id
                )
            )
        episode_events.append(
            Episode_Event(
                episode_id=episode.episode_id,
                event_id=condition.condition_occurrence_id,
                episode_event_field_concept_id=1147127,  # condition_occurrence.condition_occurrence_id
            )
        )
    session.add_all(conditions)
    session.add_all(measurements)
    session.add_all(episodes)
    session.add_all(episode_events)
    session.commit()

def _populate_test_data(session):
    """Brute force addition of test data for development/testing purposes."""

    # Data
    concept_by_domain = pd.DataFrame(
        session.query(
            *Concept.__table__.columns
        )
        .filter(
            sa.or_(
                Concept.domain_id.in_(['Gender', 'Ethnicity', 'Race', 'Visit', 'Location', 'Provider', 'Type Concept']),
                sa.and_(
                    Concept.domain_id == 'Condition',
                    Concept.vocabulary_id == 'ICDO3'
                )
            )
        )
    )

    avail_gender = list(concept_by_domain[concept_by_domain.domain_id=='Gender'].concept_id)
    avail_ethnicity = list(concept_by_domain[concept_by_domain.domain_id=='Ethnicity'].concept_id)
    avail_race = list(concept_by_domain[concept_by_domain.domain_id=='Race'].concept_id)
    avail_place_of_service = list(concept_by_domain[concept_by_domain.domain_id=='Visit'].concept_id)
    avail_country = list(concept_by_domain[concept_by_domain.concept_class_id=='Location'].concept_id)
    avail_provider = list(concept_by_domain[concept_by_domain.domain_id=='Provider'].concept_id)
    avail_types = list(concept_by_domain[concept_by_domain.domain_id=='Type Concept'].concept_id)
    cancers = list(concept_by_domain[(concept_by_domain.domain_id=='Condition')&(concept_by_domain.vocabulary_id=='ICDO3') & (concept_by_domain.concept_code.str.contains('/3'))].concept_id)

    staging_parents = pd.DataFrame(
        session.query(
            *Concept.__table__.columns
        )
        .join(Concept_Ancestor, Concept.concept_id==Concept_Ancestor.descendant_concept_id)
        .filter(Concept_Ancestor.ancestor_concept_id==734320)
        .filter(Concept_Ancestor.max_levels_of_separation==1)
    )

    staging_sets = {}

    for axis in ['T', 'N', 'M', 'Stage']:
        parents = list(staging_parents[staging_parents.concept_name.str.contains(axis)].concept_id)
        s = pd.DataFrame(
            session.query(
                *Concept.__table__.columns
            )
            .join(Concept_Ancestor, Concept.concept_id==Concept_Ancestor.descendant_concept_id)
            .filter(Concept_Ancestor.ancestor_concept_id.in_(parents))
            .filter(Concept.concept_code.ilike('%8th%'))
            .filter(~Concept.concept_code.ilike('%yp%'))
        )
        staging_sets[axis] = s


    # Care sites
    _populate_reference_data(session, avail_country, avail_place_of_service, avail_provider, avail_gender)
    session.commit()
    care_sites = session.query(Care_Site).all()
    
    # People and visits
    _populate_people_and_visits(session, care_sites, avail_gender, avail_race, avail_ethnicity, avail_place_of_service)
    _populate_observation_periods(session, avail_types)
    _populate_conditions_and_modifiers(session, staging_sets, cancers, avail_types)


@app.command()
def omop_cdm(
    add_test_data: Annotated[bool, typer.Option(help="Whether to add synthetic test data after loading Athena data. Omit if not used.")] = False,
    chunk_size: Annotated[int, typer.Option(
        "--chunk-size", "-c", 
        help="Number of rows to process in each chunk when loading large tables with fallback pandas loader.")] = 5000,
    pred_class_dir: Annotated[Optional[str], typer.Option(help="Path to the directory containing `predicate_classification.csv` and `predicate_mapping.csv`.")] = None,
    fulltext: Annotated[bool, typer.Option("--fulltext/--no-fulltext", help="Install and populate PostgreSQL full-text sidecars after loading the vocabulary tables.")] = False,
    fulltext_regconfig: Annotated[str, typer.Option("--fulltext-regconfig", help="PostgreSQL text search configuration to use when populating the full-text sidecars.")] = "english",
    verbosity: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity (up to two levels)")] = 0,
):
    """
    Instantiate the database from scratch by loading the Athena vocabularies.
    IMPORTANT: This will wipe the entire existing database in the db container.
    """
    configure_logging_level(verbosity)
    load_dotenv()

    engine_string = os.getenv('OMOP_DATABASE_URL')
    if engine_string is None:
        raise RuntimeError("OMOP_DATABASE_URL environment variable not set.")
    
    engine = sa.create_engine(engine_string, future=True, echo=False)

    # Drop all existing tables for a fresh bootstrap
    metadata = Base.metadata
    metadata.reflect(bind=engine)
    metadata.drop_all(engine)
    
    # Re-init tables
    create_db(engine)

    Session = sessionmaker(bind=engine, future=True)
    session = Session()

    loader = PandasLoader()

    athena_db_path = os.getenv('SOURCE_PATH')
    if athena_db_path is None:
        raise RuntimeError("SOURCE_PATH environment variable not set. Please set it in your .env file to point to the Athena CSV files base directory.")
    base_path = Path(athena_db_path).resolve()
    assert base_path.exists(), f"Source path {base_path} does not exist"

    with bulk_load_context(session):
       for model in ATHENA_INITIAL_LOAD:
           model.load_csv(
               session,
               base_path / f"{model.__tablename__.upper()}.csv",
               dedupe=True,
               merge_strategy="upsert",
               loader=loader
           )
       session.commit()

    with bulk_load_context(session):
        for model in ATHENA_SUBSEQUENT_LOAD:
            model.load_csv(
                session,
                base_path / f"{model.__tablename__.upper()}.csv",
                dedupe=True,
                chunksize=chunk_size,
                merge_strategy="replace",
                loader=loader
            )
            session.commit()

    if fulltext:
        try:
            _enable_fulltext_sidecars(engine, fulltext_regconfig)
        except Exception as exc:
            logger.error(f"Failed to enable PostgreSQL full-text sidecars: {exc}")
            logger.info("Continuing with bootstrap without full-text sidecars. You can rerun omop-maint fulltext install and omop-maint fulltext populate later.")

    try:
       relationship_classification(pred_class_dir)
    except Exception as e:
       logger.error(f"Failed to ingest predicate classifications: {e}")
       logger.info("Continuing with bootstrap without predicate classifications. Re-run cli `ingest-classification` command once the issue is resolved.")

    if add_test_data:
        _populate_test_data(session)

@app.command()
def relationship_classification(
    pred_class_dir: Annotated[Optional[str], typer.Option(help="Path to the directory containing `predicate_classification.csv` and `predicate_mapping.csv`.")] = None,
    verbosity: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity (up to two levels)")] = 0,
):
    """
    Method to get the pre-classified predicates into the database.
    """
    configure_logging_level(verbosity)
    load_dotenv()

    if pred_class_dir is None:
        pred_class_dir = str((Path(__file__).parent.parent.parent / "docs").resolve())

    pred_class_dir_pl = Path(pred_class_dir)

    pred_mapping_file = pred_class_dir_pl / "predicate_mapping.csv"
    if not pred_mapping_file.is_file():
        raise FileNotFoundError(f"`predicate_mapping.csv` not found in {pred_class_dir_pl}")
    pred_class_file = pred_class_dir_pl / "predicate_classification.csv"
    if not pred_class_file:
        raise FileNotFoundError(f"`predicate_classification.csv` not found in {pred_class_dir_pl}")

    df_class = pd.read_csv(pred_class_file)
    df_mapping = pd.read_csv(pred_mapping_file)
    

    # 1. RelationshipClass
    df_rel_cls = df_class.rename(columns={"class": "class_id", "subclass": "subclass_id"})

    # Only allow that a subclass_id maps exactly to one semantic and inference description
    check = df_rel_cls.groupby(["class_id", "subclass_id"])[["description", "semantics", "inference"]].nunique(dropna=True)
    violations = check[(check > 1).any(axis=1)]
    if not violations.empty:
        conflicting_data = df_rel_cls[df_rel_cls["subclass_id"].isin(violations.index)].sort_values("subclass_id")
        logger.error(f"Validation Failed! {len(violations)} subclass_ids have conflicting definitions: {conflicting_data}")        
        raise AttributeError("Validation not passed")
    df_rel_cls_to_export = df_rel_cls.groupby(["class_id", "subclass_id"], as_index=False).first()

    # 2. RelationshipMapping
    df_rel_mapping = df_mapping.rename(columns={"class": "class_id", "subclass": "subclass_id", "r_id": "relationship_id"})
    # Same order as relationship_class.py
    df_rel_mapping = df_rel_mapping[["relationship_id", "class_id", "subclass_id"]].dropna(subset=['class_id', 'subclass_id'], how='all')
    invalid_mask = df_rel_mapping[['class_id', 'subclass_id']].isna().any(axis=1)
    dropped_ids = df_rel_mapping.loc[invalid_mask, 'relationship_id'].unique().tolist()

    if dropped_ids:
        logger.warning(f"Dropping {len(dropped_ids)} relationships due to missing parent or child class: {dropped_ids}")
    df_rel_mapping = df_rel_mapping.dropna(subset=['class_id', 'subclass_id'], how='any')
    df_rel_mapping_to_export = df_rel_mapping.drop_duplicates(subset=["relationship_id", "class_id", "subclass_id"])

    # Save and then load again
    athena_db_path = os.getenv('SOURCE_PATH')
    if athena_db_path is None:
        raise RuntimeError("SOURCE_PATH environment variable not set. Please set it in your .env file to point to the Athena CSV files base directory.")
    base_path = Path(athena_db_path).resolve()
    assert base_path.exists(), f"Source path {base_path} does not exist"

    engine_string = os.getenv('OMOP_DATABASE_URL')
    if engine_string is None:
        raise RuntimeError("OMOP_DATABASE_URL environment variable not set. Please set it in your .env file to point to your database.")
    
    engine = sa.create_engine(engine_string, future=True, echo=False)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()

    # Drop the tables
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {RelationshipMapping.staging_tablename()} CASCADE"))  # type: ignore
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {RelationshipClass.staging_tablename()} CASCADE"))  # type: ignore
        conn.execute(sa.text("DROP TYPE IF EXISTS classidenum CASCADE;"))

    tables_to_drop = [
        RelationshipMapping.__table__, 
        RelationshipClass.__table__
    ]
    Base.metadata.drop_all(bind=engine, tables=tables_to_drop, checkfirst=True)  # type: ignore
    Base.metadata.create_all(bind=engine, tables=tables_to_drop)  # type: ignore

    for model, df in zip([RelationshipClass, RelationshipMapping], [df_rel_cls_to_export, df_rel_mapping_to_export]):
        csv_path = base_path / f"{model.__tablename__.upper()}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} records to `{csv_path}` for model `{model.__name__}`")

        with bulk_load_context(session):
            model.load_csv(  # type: ignore
                session,
                csv_path,
                dedupe=True,
                merge_strategy="replace",
                loader=PandasLoader()
            )
            session.commit()

if __name__ == "__main__":
    app()