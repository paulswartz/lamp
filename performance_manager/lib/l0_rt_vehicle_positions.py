from typing import List, Union

import pathlib
import numpy
import pandas
import sqlalchemy as sa

from .gtfs_utils import start_time_to_seconds, add_event_hash_column
from .logging_utils import ProcessLogger
from .postgres_schema import (
    StaticStops,
    VehicleEvents,
    MetadataLog,
    StaticFeedInfo,
    TempHashCompare,
)
from .postgres_utils import get_unprocessed_files, DatabaseManager
from .s3_utils import read_parquet


def get_vp_dataframe(to_load: Union[str, List[str]]) -> pandas.DataFrame:
    """
    return a dataframe from a vehicle position parquet file (or list of files)
    with expected columns
    """
    vehicle_position_cols = [
        "current_status",
        "current_stop_sequence",
        "stop_id",
        "vehicle_timestamp",
        "direction_id",
        "route_id",
        "start_date",
        "start_time",
        "vehicle_id",
    ]
    vehicle_position_filters = [
        ("current_status", "!=", "None"),
        ("current_stop_sequence", ">=", 0),
        ("stop_id", "!=", "None"),
        ("vehicle_timestamp", ">", 0),
        ("direction_id", "in", (0, 1)),
        ("route_id", "!=", "None"),
        ("start_date", "!=", "None"),
        ("start_time", "!=", "None"),
        ("vehicle_id", "!=", "None"),
    ]
    return read_parquet(to_load, columns=vehicle_position_cols, filters=vehicle_position_filters)


def transform_vp_dtyes(vehicle_positions: pandas.DataFrame) -> pandas.DataFrame:
    """
    ingest dataframe of vehicle position data from parquet file and transform
    column datatypes
    """
    # current_staus: 1 = MOVING, 0 = STOPPED_AT
    vehicle_positions["is_moving"] = numpy.where(
        vehicle_positions["current_status"] != "STOPPED_AT", True, False
    ).astype(numpy.bool8)
    vehicle_positions = vehicle_positions.drop(columns=["current_status"])

    # rename current_stop_sequence as stop_sequence
    # and convert to int64 [not nullable]
    vehicle_positions = vehicle_positions.rename(
        columns={"current_stop_sequence": "stop_sequence"}
    )
    vehicle_positions["stop_sequence"] = pandas.to_numeric(
        vehicle_positions["stop_sequence"]
    ).astype("int64")

    # store vehicle_timestamp as Int64 [nullable]
    vehicle_positions["vehicle_timestamp"] = pandas.to_numeric(
        vehicle_positions["vehicle_timestamp"]
    ).astype("Int64")

    # store direction_id as bool [not nullable]
    vehicle_positions["direction_id"] = pandas.to_numeric(
        vehicle_positions["direction_id"]
    ).astype(numpy.bool8)

    # store start_date as int64 [not nullable]
    vehicle_positions["start_date"] = pandas.to_numeric(
        vehicle_positions["start_date"]
    ).astype("int64")

    # store start_time as int64 [not nullable] seconds from start of day
    vehicle_positions["start_time"] = (
        vehicle_positions["start_time"]
        .apply(start_time_to_seconds)
        .astype("int64")
    )

    return vehicle_positions


def join_gtfs_static(
    vehicle_positions: pandas.DataFrame, db_manager: DatabaseManager
) -> pandas.DataFrame:
    """
    join vehicle position dataframe to gtfs static records

    adds "fk_static_timestamp" and "parent_station" columns to dataframe
    """
    # get unique "start_date" values from vehicle position dataframe
    # with associated minimum "vehicle_timestamp"
    date_groups = vehicle_positions.groupby(by="start_date")[
        "vehicle_timestamp"
    ].min()

    # dictionary used to match minimum "vehicle_timestamp" values to
    # "timestamp" from StaticFeedInfo table, to be used as foreign key
    timestamp_lookup = {}
    for (date, min_timestamp) in date_groups.iteritems():
        date = int(date)
        min_timestamp = int(min_timestamp)
        # "start_date" from vehicle timestamps must be between "feed_start_date"
        # and "feed_end_date" in StaticFeedInfo
        # minimum "vehicle_timestamp" must also be less than "timestamp" from
        # StaticFeedInfo, order by "timestamp" descending and limit to 1 result
        # this should deal with multiple static schedules with possible
        # overlapping times of applicability
        feed_timestamp_query = (
            sa.select(StaticFeedInfo.timestamp)
            .where(
                (StaticFeedInfo.feed_start_date <= date)
                & (StaticFeedInfo.feed_end_date >= date)
                # & (StaticFeedInfo.timestamp < min_timestamp)
            )
            .order_by(StaticFeedInfo.timestamp.desc())
            .limit(1)
        )
        result = db_manager.select_as_list(feed_timestamp_query)
        # if this query does not produce a result, no static schedule info
        # exists for this vehicle position data, so the vehicle position data
        # should not be processed until valid static schedule data exists
        if len(result) == 0:
            raise IndexError(
                f"StaticFeedInfo table has no matching schedule for start_date={date}, timestamp={min_timestamp}"
            )
        timestamp_lookup[min_timestamp] = int(result[0]["timestamp"])

    vehicle_positions["fk_static_timestamp"] = timestamp_lookup[
        min(timestamp_lookup.keys())
    ]
    # add "fk_static_timestamp" column to vehicle positions dataframe
    # loop is to handle batches vehicle position batches that are applicable to
    # overlapping static gtfs data
    for min_timestamp in sorted(timestamp_lookup.keys()):
        timestamp_mask = vehicle_positions["vehicle_timestamp"] >= min_timestamp
        vehicle_positions.loc[
            timestamp_mask, "fk_static_timestamp"
        ] = timestamp_lookup[min_timestamp]

    # unique list of "fk_static_timestamp" values for pulling parent stations
    static_timestamps = list(set(timestamp_lookup.values()))

    # pull parent station data for joining to vehicle position events
    parent_station_query = sa.select(
        StaticStops.timestamp.label("fk_static_timestamp"),
        StaticStops.stop_id,
        StaticStops.parent_station,
    ).where(StaticStops.timestamp.in_(static_timestamps))
    parent_stations = db_manager.select_as_dataframe(parent_station_query)

    # join parent stations to vehicle positions on "stop_id" and gtfs static
    # timestamp foreign key
    vehicle_positions = vehicle_positions.merge(
        parent_stations, how="left", on=["fk_static_timestamp", "stop_id"]
    )
    # is parent station is not provided, transfer "stop_id" value to
    # "parent_station" column
    vehicle_positions["parent_station"] = numpy.where(
        vehicle_positions["parent_station"].isna(),
        vehicle_positions["stop_id"],
        vehicle_positions["parent_station"],
    )

    return vehicle_positions


def transform_vp_timestamps(
    vehicle_positions: pandas.DataFrame,
) -> pandas.DataFrame:
    """
    convert raw vehicle position data with one timestamp field into
    vehicle position events with vp_move_timestamp and vp_stop_timestamp
    """
    vehicle_positions = add_event_hash_column(
        vehicle_positions, 
        hash_column_name="trip_stop_hash", 
    )

    vehicle_positions = (
            vehicle_positions.sort_values(
                by=["trip_stop_hash","is_moving","vehicle_timestamp"], 
                ascending=True
            ).drop_duplicates(
                subset=["trip_stop_hash", "is_moving"], 
                keep="first"
            )
        )

    vehicle_positions["vp_stop_timestamp"] = numpy.where(
        (vehicle_positions["is_moving"] == False), 
        vehicle_positions["vehicle_timestamp"],
        None,
    )

    vehicle_positions["vp_move_timestamp"] = numpy.where(
        (vehicle_positions["is_moving"] == True), 
        vehicle_positions["vehicle_timestamp"],
        None,
    )

    vehicle_positions["vp_move_timestamp"] = numpy.where(
        (
            (vehicle_positions["is_moving"] == False) 
            & (vehicle_positions["trip_stop_hash"] == vehicle_positions["trip_stop_hash"].shift(-1))
        ), 
        vehicle_positions["vp_move_timestamp"].shift(-1),
        vehicle_positions["vp_move_timestamp"],
    )

    vehicle_positions = (
        vehicle_positions.drop_duplicates(
            subset=["trip_stop_hash"], 
            keep="first"
        ).drop(
            columns=["is_moving", "vehicle_timestamp"]
        )
    )

    vehicle_positions.insert(0, "pk_id", None)

    return vehicle_positions


def get_event_overlap(
    new_events: pandas.DataFrame,
    db_manager: DatabaseManager,
) -> pandas.DataFrame:
    """
    get a dataframe consisting of the new events and any overlapping events from
    the vehicle positions events table, sorted by vehicle id and then timestamp
    """

    db_manager.truncate_table(TempHashCompare)

    db_manager.execute_with_data(
        sa.insert(TempHashCompare.__table__),
        new_events[["trip_stop_hash"]].rename(columns={"trip_stop_hash":"hash"})
    )

    get_db_events = sa.select(
        VehicleEvents.pk_id,
        VehicleEvents.trip_stop_hash,
        VehicleEvents.vp_move_timestamp,
        VehicleEvents.vp_stop_timestamp,
    ).join(
        TempHashCompare,
        TempHashCompare.hash == VehicleEvents.trip_stop_hash
    )

    db_events = db_manager.select_as_dataframe(get_db_events)

    # Join records from db with records from parquet db records should
    # contain 'index' column that does not exist in parquet dataset
    
    return pandas.concat([db_events, new_events]).sort_values(by=["trip_stop_hash", "pk_id"], na_position="last")


def merge_vehicle_position_events(
    new_events: pandas.DataFrame,
    db_manager: DatabaseManager,
) -> None:
    """
    merge new events from parquet file with exiting events found in database
    new events will be merged with existing events in a 5 minutes window
    surrounding the year/month/day/hour value found in path of parquet files
    """
    process_logger = ProcessLogger("vp_merge_events")
    process_logger.log_start()

    merge_events = get_event_overlap(new_events, db_manager)

    duplicate_event_mask = merge_events.duplicated(subset="trip_stop_hash", keep=False)

    # new events that will be inserted into db table
    new_to_insert_mask = (
        (~duplicate_event_mask)
        & (merge_events["pk_id"].isna())
    )

    update_mask = (
        (duplicate_event_mask)
        & (
            merge_events["trip_stop_hash"] == merge_events["trip_stop_hash"].shift(-1)
        )
        & (
            (
                (~merge_events["vp_move_timestamp"].shift(-1).isna())
                & (
                    (merge_events["vp_move_timestamp"].isna())
                    | (merge_events["vp_move_timestamp"] > merge_events["vp_move_timestamp"].shift(-1))
                )
                
            )
            | (
                (~merge_events["vp_stop_timestamp"].shift(-1).isna())
                & (
                    (merge_events["vp_stop_timestamp"].isna())
                    | (merge_events["vp_stop_timestamp"] > merge_events["vp_stop_timestamp"].shift(-1))
                )
                
            )
        )
    )

    update_records = merge_events[merge_events["trip_stop_hash"].isin(merge_events.loc[update_mask,"trip_stop_hash"])]
    
    update_records["vp_move_timestamp"] = numpy.where(
        (update_records["trip_stop_hash"] == update_records["trip_stop_hash"].shift(-1))
        & (
            (update_records["vp_move_timestamp"].isna())
            | (update_records["vp_move_timestamp"] > update_records["vp_move_timestamp"].shift(-1))
        ),
        update_records["vp_move_timestamp"].shift(-1),
        update_records["vp_move_timestamp"],
    )

    update_records["vp_stop_timestamp"] = numpy.where(
        (update_records["trip_stop_hash"] == update_records["trip_stop_hash"].shift(-1))
        & (
            (update_records["vp_stop_timestamp"].isna())
            | (update_records["vp_stop_timestamp"] > update_records["vp_stop_timestamp"].shift(-1))
        ),
        update_records["vp_stop_timestamp"].shift(-1),
        update_records["vp_stop_timestamp"],
    )

    update_records = update_records[~update_records["pk_id"].isna()]

    # add counts to process logger metadata
    process_logger.add_metadata(
        updated_count=update_records.shape[0],
        inserted_count=new_to_insert_mask.sum(),
    )

    # DB UPDATE operation
    if update_records.shape[0] > 0:
        update_db_events = sa.update(VehicleEvents.__table__).where(
            VehicleEvents.pk_id == sa.bindparam("b_pk_id")
        )
        db_manager.execute_with_data(
            update_db_events,
            update_records[["pk_id","vp_move_timestamp","vp_stop_timestamp"]].rename(columns={"pk_id": "b_pk_id"})
        )

    # DB INSERT operation
    if new_to_insert_mask.sum() > 0:
        insert_cols = list(set(merge_events.columns) - {"pk_id"})
        db_manager.execute_with_data(
            sa.insert(VehicleEvents.__table__),
            merge_events.loc[new_to_insert_mask, insert_cols],
        )

    process_logger.log_complete()


def process_vehicle_positions(db_manager: DatabaseManager) -> None:
    """
    process a bunch of vehicle position files
    create events for them
    merge those events with existing events
    """
    process_logger = ProcessLogger(
        "l0_tables_loader", table_type="rt_vehicle_position"
    )
    process_logger.log_start()

    # check metadata table for unprocessed parquet files
    paths_to_load = get_unprocessed_files(
        "RT_VEHICLE_POSITIONS", db_manager, file_limit=6
    )
    process_logger.add_metadata(count_of_paths=len(paths_to_load))

    for folder_data in paths_to_load:
        folder = str(pathlib.Path(folder_data["paths"][0]).parent)
        ids = folder_data["ids"]
        paths = folder_data["paths"]

        subprocess_logger = ProcessLogger(
            "l0_load_table",
            table_type="rt_vehicle_position",
            s3_path=folder,
            file_count=len(paths),
        )
        subprocess_logger.log_start()

        try:
            sizes = {}

            new_events = get_vp_dataframe(paths)
            sizes["parquet_events_count"] = new_events.shape[0]

            new_events = transform_vp_dtyes(new_events)

            new_events = join_gtfs_static(new_events, db_manager)

            new_events = transform_vp_timestamps(new_events)
            sizes["merge_events_count"] = new_events.shape[0]

            subprocess_logger.add_metadata(**sizes)

            merge_vehicle_position_events(new_events, db_manager)

            update_md_log = (
                sa.update(MetadataLog.__table__)
                .where(MetadataLog.pk_id.in_(ids))
                .values(processed=1)
            )
            db_manager.execute(update_md_log)

            subprocess_logger.log_complete()
        except Exception as exception:
            update_md_log = (
                sa.update(MetadataLog.__table__)
                .where(MetadataLog.pk_id.in_(ids))
                .values(processed=1, process_fail=1)
            )
            db_manager.execute(update_md_log)

            subprocess_logger.log_failure(exception)

    process_logger.log_complete()
