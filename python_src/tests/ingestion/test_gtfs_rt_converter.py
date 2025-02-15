import os
from datetime import datetime, timezone
from multiprocessing import Queue
from unittest.mock import patch

from pyarrow import fs

from lamp_py.ingestion.convert_gtfs_rt import GtfsRtConverter
from lamp_py.ingestion.converter import ConfigType

from ..test_resources import incoming_dir


def test_bad_conversion_local() -> None:
    """
    test that a bad filename will result in an empty table and the filename will
    be added to the error files list
    """
    # dummy config to avoid mypy errors
    converter = GtfsRtConverter(
        config_type=ConfigType.RT_ALERTS, metadata_queue=Queue()
    )
    converter.add_files(["badfile"])

    print("YOLO")

    # process the bad file and get the table out
    for _ in converter.process_files():
        assert False, "Generated Table for s3 badfile"

    # archive files should be empty, error files should have the bad file
    assert len(converter.archive_files) == 0
    assert converter.error_files == ["badfile"]


def test_bad_conversion_s3() -> None:
    """
    test that a bad s3 filename will result in an empty table and the filename
    will be added to the error files list
    """
    with patch("pyarrow.fs.S3FileSystem", return_value=fs.LocalFileSystem):
        converter = GtfsRtConverter(
            config_type=ConfigType.RT_ALERTS, metadata_queue=Queue()
        )
        converter.add_files(["s3://badfile"])

        # process the bad file and get the table out
        for _ in converter.process_files():
            assert False, "Generated Table for s3 badfile"

        # archive files should be empty, error files should have the bad file
        assert len(converter.archive_files) == 0
        assert converter.error_files == ["badfile"]


def test_empty_files() -> None:
    """test that empty files produce empty tables"""
    configs_to_test = (
        ConfigType.RT_VEHICLE_POSITIONS,
        ConfigType.RT_ALERTS,
        ConfigType.RT_TRIP_UPDATES,
    )

    for config_type in configs_to_test:
        converter = GtfsRtConverter(
            config_type=config_type, metadata_queue=Queue()
        )
        converter.thread_init()

        empty_file = os.path.join(incoming_dir, "empty.json.gz")
        _, filename, table = converter.gz_to_pyarrow(empty_file)
        assert filename == empty_file
        assert table.to_pandas().shape == (
            0,
            len(converter.detail.export_schema),
        )

        one_blank_file = os.path.join(incoming_dir, "one_blank_record.json.gz")
        _, filename, table = converter.gz_to_pyarrow(one_blank_file)
        assert filename == one_blank_file
        assert table.to_pandas().shape == (
            1,
            len(converter.detail.export_schema),
        )


def test_vehicle_positions_file_conversion() -> None:
    """
    TODO - convert a dummy json data to parquet and check that the new file
    matches expectations
    """
    gtfs_rt_file = os.path.join(
        incoming_dir,
        "2022-01-01T00:00:03Z_https_cdn.mbta.com_realtime_VehiclePositions_enhanced.json.gz",
    )
    config_type = ConfigType.from_filename(gtfs_rt_file)
    assert config_type == ConfigType.RT_VEHICLE_POSITIONS

    converter = GtfsRtConverter(config_type, metadata_queue=Queue())
    converter.thread_init()
    timestamp, filename, table = converter.gz_to_pyarrow(gtfs_rt_file)

    assert timestamp.month == 1
    assert timestamp.year == 2022
    assert timestamp.day == 1

    assert filename == gtfs_rt_file

    np_df = table.to_pandas()

    # tuple(na count, dtype, max, min)
    file_details = {
        "year": (0, "int16", "2022", "2022"),
        "month": (0, "int8", "1", "1"),
        "day": (0, "int8", "1", "1"),
        "hour": (0, "int8", "0", "0"),
        "feed_timestamp": (0, "int64", "1640995202", "1640995202"),
        "entity_id": (0, "object", "y4124", "1625"),
        "current_status": (0, "object", "STOPPED_AT", "INCOMING_AT"),
        "current_stop_sequence": (29, "float64", "640.0", "1.0"),
        "occupancy_percentage": (426, "float64", "nan", "nan"),
        "occupancy_status": (190, "object", "nan", "nan"),
        "stop_id": (29, "object", "nan", "nan"),
        "vehicle_timestamp": (0, "int64", "1640995198", "1640994366"),
        "bearing": (0, "int64", "360", "0"),
        "latitude": (0, "float64", "42.778499603271484", "41.82632064819336"),
        "longitude": (0, "float64", "-70.7895736694336", "-71.5481185913086"),
        "speed": (392, "float64", "29.9", "2.6"),
        "direction_id": (4, "float64", "1.0", "0.0"),
        "route_id": (0, "object", "Shuttle-Generic", "1"),
        "schedule_relationship": (0, "object", "UNSCHEDULED", "ADDED"),
        "start_date": (40, "object", "nan", "nan"),
        "start_time": (87, "object", "nan", "nan"),
        "trip_id": (0, "object", "nan", "nan"),
        "vehicle_id": (0, "object", "y4124", "1625"),
        "vehicle_label": (0, "object", "4124", "0420"),
        "vehicle_consist": (324, "object", "nan", "nan"),
    }

    # 426 records in 'entity' for 2022-01-01T00:00:03Z_https_cdn.mbta.com_realtime_VehiclePositions_enhanced.json.gz
    assert np_df.shape == (426, len(converter.detail.export_schema))

    all_expected_paths = set(file_details.keys())

    # ensure all of the expected paths were found and there aren't any
    # additional ones
    assert all_expected_paths == set(np_df.columns)

    # check file details
    for col, (na_count, d_type, upper, lower) in file_details.items():
        print(f"checking: {col}")
        assert na_count == np_df[col].isna().sum()
        assert d_type == np_df[col].dtype
        if upper != "nan":
            assert upper == str(np_df[col].max())
        if lower != "nan":
            assert lower == str(np_df[col].min())


def test_rt_alert_file_conversion() -> None:
    """
    TODO - convert a dummy json data to parquet and check that the new file
    matches expectations
    """
    gtfs_rt_file = os.path.join(
        incoming_dir,
        "2022-05-04T15:59:48Z_https_cdn.mbta.com_realtime_Alerts_enhanced.json.gz",
    )

    config_type = ConfigType.from_filename(gtfs_rt_file)
    assert config_type == ConfigType.RT_ALERTS

    converter = GtfsRtConverter(config_type, metadata_queue=Queue())
    converter.thread_init()
    timestamp, filename, table = converter.gz_to_pyarrow(gtfs_rt_file)

    assert timestamp.month == 5
    assert timestamp.year == 2022
    assert timestamp.day == 4

    assert filename == gtfs_rt_file

    np_df = table.to_pandas()

    # tuple(na count, dtype, max, min)
    file_details = {
        "year": (0, "int16", "2022", "2022"),
        "month": (0, "int8", "5", "5"),
        "day": (0, "int8", "4", "4"),
        "hour": (0, "int8", "15", "15"),
        "feed_timestamp": (0, "int64", "1651679986", "1651679986"),
        "entity_id": (0, "object", "442146", "293631"),
        "effect": (0, "object", "UNKNOWN_EFFECT", "DETOUR"),
        "effect_detail": (0, "object", "TRACK_CHANGE", "BIKE_ISSUE"),
        "cause": (0, "object", "UNKNOWN_CAUSE", "CONSTRUCTION"),
        "cause_detail": (0, "object", "UNKNOWN_CAUSE", "CONSTRUCTION"),
        "severity": (0, "int64", "10", "0"),
        "severity_level": (0, "object", "WARNING", "INFO"),
        "created_timestamp": (0, "int64", "1651679836", "1549051333"),
        "last_modified_timestamp": (0, "int64", "1651679848", "1549051333"),
        "alert_lifecycle": (0, "object", "UPCOMING_ONGOING", "NEW"),
        "duration_certainty": (0, "object", "UNKNOWN", "ESTIMATED"),
        "last_push_notification": (144, "float64", "nan", "nan"),
        "active_period": (2, "object", "nan", "nan"),
        "reminder_times": (132, "object", "nan", "nan"),
        "closed_timestamp": (142, "float64", "1651679848.0", "1651679682.0"),
        "short_header_text_translation": (0, "object", "nan", "nan"),
        "header_text_translation": (0, "object", "nan", "nan"),
        "description_text_translation": (0, "object", "nan", "nan"),
        "service_effect_text_translation": (0, "object", "nan", "nan"),
        "timeframe_text_translation": (44, "object", "nan", "nan"),
        "url_translation": (138, "object", "nan", "nan"),
        "recurrence_text_translation": (139, "object", "nan", "nan"),
        "informed_entity": (0, "object", "nan", "nan"),
    }

    # 144 records in 'entity' for 2022-05-04T15:59:48Z_https_cdn.mbta.com_realtime_Alerts_enhanced.json.gz
    assert np_df.shape == (144, len(converter.detail.export_schema))

    all_expected_paths = set(file_details.keys())

    # ensure all of the expected paths were found and there aren't any
    # additional ones
    assert all_expected_paths == set(np_df.columns)

    # check file details
    for col, (na_count, d_type, upper, lower) in file_details.items():
        print(f"checking: {col}")
        assert na_count == np_df[col].isna().sum()
        assert d_type == np_df[col].dtype
        if upper != "nan":
            assert upper == str(np_df[col].max())
        if lower != "nan":
            assert lower == str(np_df[col].min())


def test_rt_trip_file_conversion() -> None:
    """
    TODO - convert a dummy json data to parquet and check that the new file
    matches expectations
    """
    gtfs_rt_file = os.path.join(
        incoming_dir,
        "2022-05-08T06:04:57Z_https_cdn.mbta.com_realtime_TripUpdates_enhanced.json.gz",
    )

    config_type = ConfigType.from_filename(gtfs_rt_file)
    assert config_type == ConfigType.RT_TRIP_UPDATES

    converter = GtfsRtConverter(config_type, metadata_queue=Queue())
    converter.thread_init()
    timestamp, filename, table = converter.gz_to_pyarrow(gtfs_rt_file)

    assert timestamp.month == 5
    assert timestamp.year == 2022
    assert timestamp.day == 8

    assert filename == gtfs_rt_file

    np_df = table.to_pandas()

    # tuple(na count, dtype, max, min)
    file_details = {
        "year": (0, "int16", "2022", "2022"),
        "month": (0, "int8", "5", "5"),
        "day": (0, "int8", "8", "8"),
        "hour": (0, "int8", "6", "6"),
        "feed_timestamp": (0, "int64", "1651989896", "1651989896"),
        "entity_id": (0, "object", "CR-532710-1518", "50922039"),
        "timestamp": (51, "float64", "1651989886.0", "1651989816.0"),
        "stop_time_update": (0, "object", "nan", "nan"),
        "direction_id": (0, "int64", "1", "0"),
        "route_id": (0, "object", "SL1", "1"),
        "start_date": (50, "object", "nan", "nan"),
        "start_time": (51, "object", "nan", "nan"),
        "trip_id": (0, "object", "CR-532710-1518", "50922039"),
        "route_pattern_id": (79, "object", "nan", "nan"),
        "schedule_relationship": (79, "object", "nan", "nan"),
        "vehicle_id": (39, "object", "nan", "nan"),
        "vehicle_label": (55, "object", "nan", "nan"),
    }

    # 79 records in 'entity' for
    # 2022-05-08T06:04:57Z_https_cdn.mbta.com_realtime_TripUpdates_enhanced.json.gz
    assert np_df.shape == (79, len(converter.detail.export_schema))

    all_expected_paths = set(file_details.keys())

    # ensure all of the expected paths were found and there aren't any
    # additional ones
    assert all_expected_paths == set(np_df.columns)

    # check file details
    for col, (na_count, d_type, upper, lower) in file_details.items():
        print(f"checking: {col}")
        assert na_count == np_df[col].isna().sum()
        assert d_type == np_df[col].dtype
        if upper != "nan":
            assert upper == str(np_df[col].max())
        if lower != "nan":
            assert lower == str(np_df[col].min())


def test_bus_vehicle_positions_file_conversion() -> None:
    """
    TODO - convert a dummy json data to parquet and check that the new file
    matches expectations
    """
    gtfs_rt_file = os.path.join(
        incoming_dir,
        "2022-05-05T16_00_15Z_https_mbta_busloc_s3.s3.amazonaws.com_prod_VehiclePositions_enhanced.json.gz",
    )

    config_type = ConfigType.from_filename(gtfs_rt_file)
    assert config_type == ConfigType.BUS_VEHICLE_POSITIONS

    converter = GtfsRtConverter(config_type, metadata_queue=Queue())
    converter.thread_init()
    timestamp, filename, table = converter.gz_to_pyarrow(gtfs_rt_file)

    assert timestamp.month == 5
    assert timestamp.year == 2022
    assert timestamp.day == 5

    assert filename == gtfs_rt_file

    np_df = table.to_pandas()

    # tuple(na count, dtype, max, min)
    file_details = {
        "year": (0, "int16", "2022", "2022"),
        "month": (0, "int8", "5", "5"),
        "day": (0, "int8", "5", "5"),
        "hour": (0, "int8", "16", "16"),
        "feed_timestamp": (0, "int64", "1651766414", "1651766414"),
        "entity_id": (0, "object", "1651766413_1740", "1651764730_1426"),
        "block_id": (484, "object", "nan", "nan"),
        "capacity": (12, "float64", "57.0", "36.0"),
        "current_stop_sequence": (844, "float64", "nan", "nan"),
        "load": (569, "float64", "42.0", "0.0"),
        "location_source": (0, "object", "transitmaster", "samsara"),
        "occupancy_percentage": (569, "float64", "80.0", "0.0"),
        "occupancy_status": (569, "object", "nan", "nan"),
        "revenue": (0, "bool", "True", "False"),
        "run_id": (484, "object", "nan", "nan"),
        "stop_id": (844, "object", "nan", "nan"),
        "vehicle_timestamp": (0, "int64", "1651766413", "1651764730"),
        "bearing": (0, "int64", "356", "0"),
        "latitude": (0, "float64", "42.65629769", "42.1069972"),
        "longitude": (0, "float64", "-70.62653102", "-71.272446339"),
        "speed": (163, "float64", "25.1189", "0.0"),
        "overload_id": (844, "float64", "nan", "nan"),
        "overload_offset": (844, "float64", "nan", "nan"),
        "route_id": (529, "object", "nan", "nan"),
        "schedule_relationship": (529, "object", "nan", "nan"),
        "start_date": (779, "object", "nan", "nan"),
        "trip_id": (529, "object", "nan", "nan"),
        "vehicle_id": (0, "object", "y3159", "y0408"),
        "vehicle_label": (0, "object", "3159", "0408"),
        "assignment_status": (358, "object", "nan", "nan"),
        "operator_id": (844, "object", "nan", "nan"),
        "logon_time": (484, "float64", "1651766401.0", "1651740838.0"),
        "name": (844, "object", "nan", "nan"),
        "first_name": (844, "object", "nan", "nan"),
        "last_name": (844, "object", "nan", "nan"),
        "entity_is_deleted": (0, "bool", "False", "False"),
    }

    # 844 records in 'entity' for 2022-05-05T16_00_15Z_https_mbta_busloc_s3.s3.amazonaws.com_prod_VehiclePositions_enhanced.json.gz
    assert np_df.shape == (844, len(converter.detail.export_schema))

    all_expected_paths = set(file_details.keys())

    # ensure all of the expected paths were found and there aren't any
    # additional ones
    assert all_expected_paths == set(np_df.columns)

    # check file details
    for col, (na_count, d_type, upper, lower) in file_details.items():
        print(f"checking: {col}")
        assert na_count == np_df[col].isna().sum()
        assert d_type == np_df[col].dtype
        if upper != "nan":
            assert upper == str(np_df[col].max())
        if lower != "nan":
            assert lower == str(np_df[col].min())


def test_bus_trip_updates_file_conversion() -> None:
    """
    TODO - convert a dummy json data to parquet and check that the new file
    matches expectations
    """
    gtfs_rt_file = os.path.join(
        incoming_dir,
        "2022-06-28T10_03_18Z_https_mbta_busloc_s3.s3.amazonaws.com_prod_TripUpdates_enhanced.json.gz",
    )

    config_type = ConfigType.from_filename(gtfs_rt_file)
    assert config_type == ConfigType.BUS_TRIP_UPDATES

    converter = GtfsRtConverter(config_type, metadata_queue=Queue())
    converter.thread_init()
    timestamp, filename, table = converter.gz_to_pyarrow(gtfs_rt_file)

    assert timestamp.month == 6
    assert timestamp.year == 2022
    assert timestamp.day == 28

    assert filename == gtfs_rt_file

    np_df = table.to_pandas()

    # tuple(na count, dtype, max, min)
    file_details = {
        "year": (0, "int16", "2022", "2022"),
        "month": (0, "int8", "6", "6"),
        "day": (0, "int8", "28", "28"),
        "hour": (0, "int8", "10", "10"),
        "feed_timestamp": (0, "int64", "1656410597", "1656410597"),
        "entity_id": (
            0,
            "object",
            "T86-136:86:52101511:495418",
            "A19-16:19:52110036:495423",
        ),
        "stop_time_update": (0, "object", "nan", "nan"),
        "direction_id": (157, "float64", "nan", "nan"),
        "route_id": (0, "object", "93", "111"),
        "route_pattern_id": (157, "object", "nan", "nan"),
        "schedule_relationship": (0, "object", "SCHEDULED", "SCHEDULED"),
        "start_date": (157, "object", "nan", "nan"),
        "start_time": (157, "object", "nan", "nan"),
        "trip_id": (0, "object", "52271793", "52017959"),
        "vehicle_id": (145, "object", "nan", "nan"),
        "vehicle_label": (145, "object", "nan", "nan"),
    }

    # 157 records in 'entity' for 2022-06-28T10_03_18Z_https_mbta_busloc_s3.s3.amazonaws.com_prod_TripUpdates_enhanced.json.gz
    assert np_df.shape == (157, len(converter.detail.export_schema))

    all_expected_paths = set(file_details.keys())

    # ensure all of the expected paths were found and there aren't any
    # additional ones
    assert all_expected_paths == set(np_df.columns)

    # check file details
    for col, (na_count, d_type, upper, lower) in file_details.items():
        print(f"checking: {col}")
        assert na_count == np_df[col].isna().sum()
        assert d_type == np_df[col].dtype
        if upper != "nan":
            assert upper == str(np_df[col].max())
        if lower != "nan":
            assert lower == str(np_df[col].min())


def test_start_of_hour() -> None:
    """
    test that the start of hour member is set correctly and will filter out gtfs
    rt files when converting
    """
    gtfs_rt_file_1 = os.path.join(
        incoming_dir,
        "2022-01-01T00:00:03Z_https_cdn.mbta.com_realtime_VehiclePositions_enhanced.json.gz",
    )

    gtfs_rt_file_2 = os.path.join(
        incoming_dir,
        "2022-07-05T12:35:16Z_https_cdn.mbta.com_realtime_VehiclePositions_enhanced.json.gz",
    )

    config_type = ConfigType.from_filename(gtfs_rt_file_1)
    converter = GtfsRtConverter(config_type, metadata_queue=Queue())
    converter.add_files([gtfs_rt_file_1, gtfs_rt_file_2])

    # check that start of hour has not minutes, seconmds, or microseconds
    assert converter.start_of_hour.minute == 0
    assert converter.start_of_hour.second == 0
    assert converter.start_of_hour.microsecond == 0
    assert converter.start_of_hour.tzinfo == timezone.utc

    # set start of hour to a time between 2022-07-05 and 2022-01-01
    converter.start_of_hour = datetime(
        year=2022, month=4, day=1, tzinfo=timezone.utc
    )

    table = next(converter.process_files())

    # check that only the first file was archived and error files are empty
    assert converter.archive_files == [gtfs_rt_file_1]
    assert len(converter.error_files) == 0

    month_column = table.column("month")
    for month in month_column:
        assert month.as_py() == 1

    day_column = table.column("day")
    for day in day_column:
        assert day.as_py() == 1
