"""Calendar inputs proved by retained ToolMind source conversations.

Nanbeige/ToolMind@8020ed1c03c367e4eb720ac3828ab4b0b95d8baf contains
absolute dates invented for relative user requests before any clock or result:
GraphSyn rows1840 (tomorrow weather),2067 (today tennis),33694 (next-month
flights),77950/80158 (next-month meetings/events), and BUTTON rows4691
(next-week events),5095 (today bus),8138 (today weather/moon),13661
(next-week appointments),15616 (next-Tuesday meeting) are direct examples.

Exact source file, function name and definition description select only the
reviewed consuming argument paths below. This is not a generic date-name rule.
Shared validation excludes assistant text/reasoning, schema examples and raw
metadata as clocks; explicit dates and labeled current clocks can supply
reference context. Unsupported explicit date/range forms or any earlier tool
result cause bounded abstention. It never reads the machine clock or attempts
calendar arithmetic. Birthdays, data-processing/historical queries, unknown
APIs and arbitrary natural-language intent remain outside this policy.
"""

from typing import Any

_CONTRACTS: dict[tuple[str, str, str], tuple[str, ...]] = {
    (
        "graph_syn_datasets/graphsyn.jsonl",
        "weather.fetchCurrentTemperature",
        "Retrieves the current temperature for a specified location and time.",
    ): ("dateTime",),
    (
        "graph_syn_datasets/graphsyn.jsonl",
        "Get Tennis Matches by Date",
        "Retrieves a list of tennis matches for a given string or today's string if no string is provided.",
    ): ("string",),
    (
        "graph_syn_datasets/graphsyn.jsonl",
        "Get Oil Price Today",
        "Retrieve the current oil price in Thailand for today",
    ): ("string",),
    (
        "graph_syn_datasets/graphsyn.jsonl",
        "Baseball Live Matches API",
        "Retrieve live matches for baseball, including current scores, teams, and game details.",
    ): ("string",),
    (
        "graph_syn_datasets/graphsyn.jsonl",
        "Calendar of Prices for a Month",
        "Returns the prices for each day of a month, grouped together by the number of transfers, for a given origin and destination.",
    ): ("month",),
    (
        "graph_syn_datasets/graphsyn.jsonl",
        "VIP Featured Predictions",
        "Returns daily featured prediction results with higher probability and better odds than others, providing more efficient selections. Compare its performance from the /stats/performance endpoint.",
    ): ("string",),
    (
        "graph_syn_datasets/graphsyn.jsonl",
        "community.support_groups",
        "Fetches available support groups and their meeting schedules within a community.",
    ): ("time_frame.start_date", "time_frame.end_date"),
    (
        "graph_syn_datasets/graphsyn.jsonl",
        "retrogaming.list_events",
        "Lists retro gaming events based on type, date, and location preferences.",
    ): ("date.from", "date.to"),
    (
        "graph_syn_datasets/graphsyn.jsonl",
        "Basketball Live Matches API",
        "Retrieve live matches of basketball games, including game schedules, scores, and team information.",
    ): ("string",),
    (
        "open_datasets/BUTTONInstruct-query.jsonl",
        "get_cultural_club_events",
        "Retrieve the list of events scheduled by the local cultural club for a given week.",
    ): ("week_start_date", "week_end_date"),
    (
        "open_datasets/BUTTONInstruct-query.jsonl",
        "get_next_bus_schedule",
        "Retrieve the schedule for the next bus between two locations at a specific time.",
    ): ("date",),
    (
        "open_datasets/BUTTONInstruct-query.jsonl",
        "get_weather_forecast",
        "Retrieve the weather forecast for a specific location and date.",
    ): ("date",),
    (
        "open_datasets/BUTTONInstruct-query.jsonl",
        "get_moon_phase",
        "Retrieve the moon phase and visibility for a specific date.",
    ): ("date",),
    (
        "open_datasets/BUTTONInstruct-query.jsonl",
        "get_available_appointment_slots",
        "Retrieves the available appointment slots for the specified week.",
    ): ("start_date", "end_date"),
    (
        "open_datasets/BUTTONInstruct-query.jsonl",
        "getMeetingAttendees",
        "Retrieve a list of attendees for a specified meeting.",
    ): ("date",),
}


def calendar_slots(
    name: str, definition: dict[str, Any], *, source_file: str | None
) -> tuple[str, ...]:
    description = definition.get("description")
    if not isinstance(source_file, str) or not isinstance(description, str):
        return ()
    return _CONTRACTS.get((source_file, name, description), ())
