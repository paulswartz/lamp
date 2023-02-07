from typing import Dict, List

import binascii
import numpy
import pandas
import sqlalchemy as sa

from .l0_rt_trip_updates import process_tu_files
from .l0_rt_vehicle_positions import process_vp_files
from .logging_utils import ProcessLogger
from .postgres_schema import MetadataLog, TempHashCompare, VehicleEvents
from .postgres_utils import DatabaseManager, get_unprocessed_files
from .s3_utils import get_utc_from_partition_path


def get_gtfs_rt_paths(db_manager: DatabaseManager) -> List[Dict[str, List]]:
    """
    get all of the gtfs_rt files and group them by timestamp

    """
    grouped_files = {}

    vp_files = get_unprocessed_files("RT_VEHICLE_POSITIONS", db_manager)
    for record in vp_files:
        timestamp = get_utc_from_partition_path(record["paths"][0])
        grouped_files[timestamp] = {
            "ids": record["ids"],
            "vp_paths": record["paths"],
            "tu_paths": [],
        }

    tu_files = get_unprocessed_files("RT_TRIP_UPDATES", db_manager)
    for record in tu_files:
        timestamp = get_utc_from_partition_path(record["paths"][0])
        if timestamp in grouped_files:
            grouped_files[timestamp]["ids"] += record["ids"]
            grouped_files[timestamp]["tu_paths"] += record["paths"]
        else:
            grouped_files[timestamp] = {
                "ids": record["ids"],
                "tu_paths": record["paths"],
                "vp_paths": [],
            }

    return [
        grouped_files[timestamp] for timestamp in sorted(grouped_files.keys())
    ]


def collapse_events(
    vp_events: pandas.DataFrame, tu_events: pandas.DataFrame
) -> pandas.DataFrame:
    """
    collapse the vp events and tu events into a single vehicle events df

    the vehicle events will have one or both of vp_move_time and
    vp_stop_time entries. the trip updates events will have a
    tu_stop_time entry. many of these events should have overlap in their
    trip_stop_hash entries, implying they should be associated together.

    join the dataframes and collapse rows representing the same trip and
    stop pairs.
    """
    # sort by trip stop hash and then tu stop timestamp
    events = pandas.merge(
        vp_events[["trip_stop_hash","vp_stop_timestamp","vp_move_timestamp",]],
        tu_events[["trip_stop_hash","tu_stop_timestamp"]],
        how="outer",
        on="trip_stop_hash",
        validate="one_to_one",
    )

    details_columns = [
        "stop_sequence",
        "stop_id",
        "parent_station",
        "direction_id",
        "route_id",
        "start_date",
        "start_time",
        "vehicle_id",
        "fk_static_timestamp",
        "trip_stop_hash",
    ]

    events_details = pandas.concat(
        [
            vp_events[details_columns],
            tu_events[details_columns],
        ]
    ).drop_duplicates(subset="trip_stop_hash")

    return events.merge(events_details, how="left", on="trip_stop_hash", validate="one_to_one")


def compute_metrics(events: pandas.DataFrame) -> pandas.DataFrame:
    """
    generate dwell times, stop times, and headways metrics for events
    """
    # TODO(zap / ryan) - figure out how 2 compute these. the update
    # logic will also need to be updated.
    return events


def upload_to_database(
    events: pandas.DataFrame, db_manager: DatabaseManager
) -> None:
    """
    add vehicle event data to the database

    pull existing events from the database that overlap with the proccessed
    events. split the processed events into those whose trip_stop_hash is
    already in the database and those that are brand new. update preexisting
    events where appropriate and insert the new ones.
    """
    # remove everything from the temp hash table and insert the trip stop hashs
    # from the new events. then pull events from the VehicleEvents table by
    # matching those hashes, which will be the events that will potentially be
    # updated.

    # convert hash hex to bytes for DB compare
    hash_bytes = events["trip_stop_hash"].str.decode("hex")

    db_manager.truncate_table(TempHashCompare)
    db_manager.execute_with_data(
        sa.insert(TempHashCompare.__table__), hash_bytes.to_frame("trip_stop_hash")
    )

    database_events = db_manager.select_as_dataframe(
        sa.select(
            VehicleEvents.pk_id,
            VehicleEvents.trip_stop_hash,
            VehicleEvents.vp_move_timestamp.label("vp_move_db"),
            VehicleEvents.vp_stop_timestamp.label("vp_stop_db"),
        ).join(
            TempHashCompare,
            TempHashCompare.trip_stop_hash == VehicleEvents.trip_stop_hash,
        )
    )

    if database_events.shape[0] == 0:
        database_events = pandas.DataFrame(columns=["trip_stop_hash","pk_id","vp_move_db","vp_stop_db"])
    else:
        # convert hash bytes to hex
        database_events["trip_stop_hash"] = database_events["trip_stop_hash"].apply(binascii.hexlify).str.decode("utf-8")

    # combine the existing vehicle events with the new events. sort them by
    # trip stop hash so that vehicle events from the same trip and stop will be
    # consecutive. the existing events will have a pk id while the new ones
    # will not. sorting those with na=last ensures they are ordered existing
    # first and new second
    all_events = pandas.merge(
        events,
        database_events,
        how="left",
        on="trip_stop_hash",
        validate="one_to_one",
    )

    all_events["vp_move_timestamp"] = numpy.where(
        (
            (all_events["pk_id"].notna())
            & (all_events["vp_move_db"].notna())
            & (all_events["vp_move_timestamp"].isna())
        ),
        all_events["vp_move_db"],
        all_events["vp_move_timestamp"],
    )

    all_events["vp_move_timestamp"] = numpy.where(
        (
            (all_events["pk_id"].notna())
            & (all_events["vp_move_db"].notna())
            & (all_events["vp_move_timestamp"].notna())
            & (all_events["vp_move_timestamp"] > all_events["vp_move_db"])
        ),
        all_events["vp_move_db"],
        all_events["vp_move_timestamp"],
    )

    all_events["vp_stop_timestamp"] = numpy.where(
        (
            (all_events["pk_id"].notna())
            & (all_events["vp_stop_db"].notna())
            & (all_events["vp_stop_timestamp"].isna())
        ),
        all_events["vp_stop_db"],
        all_events["vp_stop_timestamp"],
    )

    all_events["vp_stop_timestamp"] = numpy.where(
        (
            (all_events["pk_id"].notna())
            & (all_events["vp_stop_db"].notna())
            & (all_events["vp_stop_timestamp"].notna())
            & (all_events["vp_stop_timestamp"] > all_events["vp_stop_db"])
        ),
        all_events["vp_stop_db"],
        all_events["vp_stop_timestamp"],
    )

    update_mask = (
        (all_events["pk_id"].notna())
        & (
            (all_events["tu_stop_timestamp"].notna())
            | (all_events["vp_stop_timestamp"] != all_events["vp_stop_db"])
            | (all_events["vp_move_timestamp"] != all_events["vp_move_db"])
        )
    )

    all_events = all_events.drop(columns=["vp_move_db","vp_stop_db"])

    all_events = all_events.fillna(numpy.nan).replace([numpy.nan], [None])

    # convert hash hex to bytes
    all_events["trip_stop_hash"] = all_events["trip_stop_hash"].str.decode("hex")

    # update events are more complicated. find trip stop pairs that are
    # duplicated, implying the came both from the gtfs_rt files and the
    # VehicleEvents table. next, only get the first ones that come from the
    # VehicleEvents table, where the pk_id is set. lastly, leave only the
    # events that need any of their vp_move, vp_stop, or tu_stop times
    # updated.
    if update_mask.sum() > 0:
        update_cols = [
            "pk_id",
            "vp_move_timestamp",
            "vp_stop_timestamp",
            "tu_stop_timestamp",
        ]
        db_manager.execute_with_data(
            sa.update(VehicleEvents.__table__).where(
                VehicleEvents.pk_id == sa.bindparam("b_pk_id")
            ),
            all_events.loc[update_mask, update_cols].rename(columns={"pk_id": "b_pk_id"}),
        )

    # events that aren't duplicated came exclusively from the gtfs_rt files or the
    # VehicleEvents table. filter out events without pk_id's to remove events from
    # the VehicleEvents table.
    insert_mask = all_events["pk_id"].isna()

    if insert_mask.sum() > 0:
        insert_cols = list(set(all_events.columns) - {"pk_id"})

        db_manager.execute_with_data(
            sa.insert(VehicleEvents.__table__), 
            all_events.loc[insert_mask, insert_cols],
        )


def process_gtfs_rt_files(db_manager: DatabaseManager) -> None:
    """
    process vehicle position and trip update gtfs_rt files

    convert all of the entries in these files into vehicle events that
    represent unique trip and stop pairs. these events can have a vp_move_time
    (when the vehicle started moving towards the stop), a vp_stop_time (when
    the vehicle arrived at the stop as derrived from vehicle position data),
    and a tu_stop_time (when the vehicle arrived at the stop as derrived from
    the trip updates data).

    these vehicle events will then be combined with existing vehicle events in
    the database where we can compute travel time, dwell time, and headway
    metrics for each event.

    these events are either inserted into the Vehicle Events table or used to
    update existing rows in the table.

    finally, the MetadataLog table will be updated, marking the files as
    processed upon success. if a failure happens in processing, the failure
    will be logged and the file will be marked with a process_fail in the
    MetadataLog table.
    """
    hours_to_process = 6
    process_logger = ProcessLogger("l0_tables_loader")
    process_logger.log_start()

    for files in get_gtfs_rt_paths(db_manager):
        if hours_to_process == 0:
            break
        hours_to_process -= 1

        subprocess_logger = ProcessLogger(
            "l0_table_loader",
            tu_file_count=len(files["tu_paths"]),
            vp_file_count=len(files["vp_paths"]),
        )
        subprocess_logger.log_start()

        try:
            vp_events = process_vp_files(files["vp_paths"], db_manager)
            tu_events = process_tu_files(files["tu_paths"], db_manager)

            events = collapse_events(vp_events, tu_events)
            upload_to_database(events, db_manager)

            db_manager.execute(
                sa.update(MetadataLog.__table__)
                .where(MetadataLog.pk_id.in_(files["ids"]))
                .values(processed=1)
            )
            subprocess_logger.add_metadata(event_count=events.shape[0])
            subprocess_logger.log_complete()
        except Exception as error:
            db_manager.execute(
                sa.update(MetadataLog.__table__)
                .where(MetadataLog.pk_id.in_(files["ids"]))
                .values(processed=1, process_fail=1)
            )
            subprocess_logger.log_failure(error)

    process_logger.log_complete()
