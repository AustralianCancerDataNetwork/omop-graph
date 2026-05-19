from random import randint, choice
import numpy as np
from datetime import date, timedelta
import pandas as pd

import sqlalchemy as sa
from sqlalchemy.orm import Session

from omop_alchemy.cdm.model.structural.episode import Episode
from omop_alchemy.cdm.model.structural.episode_event import Episode_Event
from omop_alchemy.cdm.model.derived import Observation_Period
from omop_alchemy.cdm.model.health_system import Location, Care_Site, Provider, Visit_Occurrence
from omop_alchemy.cdm.model.clinical import (
    Person, 
    Condition_Occurrence,
    Death, 
    Measurement, 
)
from omop_alchemy.cdm.model.vocabulary import (
    Concept,
    Concept_Ancestor
)


def populate_reference_data(
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

def populate_people_and_visits(
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

def populate_observation_periods(
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
        .filter(Observation_Period.observation_period_id.is_(None))
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

def populate_conditions_and_modifiers(
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

def populate_test_data(session):
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
    populate_reference_data(session, avail_country, avail_place_of_service, avail_provider, avail_gender)
    session.commit()
    care_sites = session.query(Care_Site).all()
    
    # People and visits
    populate_people_and_visits(session, care_sites, avail_gender, avail_race, avail_ethnicity, avail_place_of_service)
    populate_observation_periods(session, avail_types)
    populate_conditions_and_modifiers(session, staging_sets, cancers, avail_types)
