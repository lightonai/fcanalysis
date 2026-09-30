"""Audited missing-clock calendar inputs in the TxT360 agent release.

LLM360/TxT360-3efforts@bfc4a082d11967cd7810fe0b773be87bf54fb32e,
agent/low: global rows124438,152445,177917,291404,502108,587006,661367
request Hebrew dates, a ride, petrochemical/manufacturing data, flights or
sports for today/tomorrow/next Monday/next week. The first call supplies an
absolute date with no preceding user/system clock or explicit date. Exact
effort, name and released description select the consuming slots below.

Shared calendar validation excludes assistant prose/native reasoning, schema
examples and raw metadata as clocks. It abstains after any tool result or for
unsupported explicit dates/ranges, including month/year. It does not read the
machine clock, perform date arithmetic, or infer contracts for other APIs.
The high release has 213 independently reviewed consuming slot contracts. All
1,401,471 raw rows were scanned for the bounded condition, and every one of
1,380 provisional distinct rejected contexts/calls was reviewed. Mixed
contracts with partial dates, explicit years/quarters, static defaults or
indirect planning bounds remain outside the gate. High selectors require the
exact function name/description and full consuming parameter schema, using
exact JSON comparison. This is a context census, not reconstruction or
retention proof. Medium has no independently selected calendar contract.
"""

from typing import Any

from .normalization import json_bytes

_CONTRACTS: dict[tuple[str, str, str], tuple[str, ...]] = {
    (
        "low",
        "Get Hebrew Month and Date",
        "Get Hebrew month, string, and holydays from a supplied string string or today's string",
    ): ("stringstring",),
    (
        "low",
        "Get Holyday Information",
        "Get the Holyday, corresponding Scriptures and if this Holyday is also a Sabbath from the supplied string string or today's string.",
    ): ("stringstring",),
    (
        "low",
        "request_uber_ride",
        "Request an Uber ride to be scheduled for a specific time and location.",
    ): ("pickup_time",),
    (
        "low",
        "Get Petrochemical Index",
        "Retrieve the current petrochemical index from the commerce domain.",
    ): ("string",),
    (
        "low",
        "United States Scheduled Flights Level API",
        "Retrieve weekly United States scheduled departing flights data at a detailed level, including flight information, departure and arrival airports, and flight schedules.",
    ): ("start_string", "end_string"),
    (
        "low",
        "retrieve_manufacturing_data",
        "Retrieves manufacturing data including production metrics, inventory levels, and quality control information for specified product lines.",
    ): ("date",),
    (
        "low",
        "Basketball Live Matches API",
        "Retrieve live matches of basketball games, including game schedules, scores, and team information.",
    ): ("string",),
    (
        "low",
        "Tomorrow Sure VIP Over 2.5 Goals API",
        "This API delivers tomorrow's sure and precise over 2.5 goal forecasts for football matches. It returns a list of predictions with their corresponding probabilities and odds.",
    ): ("string",),
}


_HIGH_CONTRACTS: dict[tuple[str, str], tuple[tuple[str, dict[str, Any]], ...]] = {
    (
        "african_hotels_and_safaries",
        "Retrieves detailed information about hotels and safari experiences in Africa, including reviews, availability, and amenities. Use this function to help travelers discover accommodations and safari packages tailored to their preferences.",
    ): (
        (
            "check_in_date",
            {
                "type": "string",
                "description": "The "
                "desired "
                "check-in "
                "date "
                "for "
                "hotel "
                "stays "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD). "
                "Defaults "
                "to "
                "today's "
                "date "
                "if "
                "unspecified.",
                "default": "today",
            },
        ),
        (
            "check_out_date",
            {
                "type": "string",
                "description": "The "
                "desired "
                "check-out "
                "date "
                "for "
                "hotel "
                "stays "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD). "
                "Defaults "
                "to "
                "tomorrow "
                "if "
                "unspecified.",
                "default": "tomorrow",
            },
        ),
    ),
    (
        "aggregate",
        "Aggregates wildfire-related news broadcasts into spatial hexagonal bins for analysis. Returns geospatial features in the specified format. Supports date range from '2015-03-01' to current date. Useful for mapping wildfire trends and incident clustering.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "ISO "
                "8601 "
                "date "
                "string "
                "for "
                "data "
                "aggregation "
                "(e.g., "
                "'2023-09-15'). "
                "Must "
                "be "
                "between "
                "'2015-03-01' "
                "and "
                "the "
                "current "
                "date.",
            },
        ),
    ),
    (
        "alerts_parameters",
        "Fetches a list of weather alerts from the National Weather Service API, with optional filtering parameters.",
    ): (
        ("end", {"type": "string", "description": "End time in ISO8601 format."}),
        ("start", {"type": "string", "description": "Start time in ISO8601 format."}),
    ),
    (
        "american_football_livescores",
        "Retrieves live scores, game status updates, and match statistics for ongoing American football games at professional (NFL) and college (NCAA) levels. Use this function to get real-time sports data including current scores, quarter/time progress, and game highlights.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Filter "
                "matches "
                "by "
                "date "
                "(format: "
                "YYYY-MM-DD). "
                "If "
                "not "
                "provided, "
                "defaults "
                "to "
                "current "
                "date.",
                "default": None,
            },
        ),
    ),
    ("analyze_stock_portfolio", "Analyze the performance of a stock portfolio"): (
        (
            "end_date",
            {"type": "string", "description": "End date of portfolio analysis"},
        ),
    ),
    (
        "api",
        "Provides access to advertising campaign management capabilities including campaign creation, performance tracking, and analytics. Supports operations for managing ad groups, targeting parameters, and budget allocation.",
    ): (
        (
            "date_range.end_date",
            {
                "type": "string",
                "description": "End date for the operation window in YYYY-MM-DD format",
            },
        ),
        (
            "date_range.start_date",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "the "
                "operation "
                "window "
                "in "
                "YYYY-MM-DD "
                "format",
            },
        ),
    ),
    (
        "app_review_metrics",
        "Retrieves daily aggregated metrics from app reviews, including ratings, sentiment analysis, and key themes. Provides insights into app performance and user satisfaction over a specified time period.",
    ): (
        (
            "datefrom",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "the "
                "date "
                "range "
                "filter "
                "(inclusive) "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD)",
            },
        ),
        (
            "dateto",
            {
                "type": "string",
                "description": "End "
                "date "
                "for "
                "the "
                "date "
                "range "
                "filter "
                "(inclusive) "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD)",
            },
        ),
    ),
    (
        "app_reviews",
        "Retrieves and sorts app reviews by timestamp within a specified date range. Returns reviews filtered by app ID and time boundaries.",
    ): (
        (
            "dateto",
            {
                "type": "string",
                "description": "The "
                "ending "
                "date/time "
                "for "
                "the "
                "review "
                "search "
                "period "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2023-12-31T23:59:59Z'). "
                "Must "
                "be "
                "later "
                "than "
                "'datefrom'.",
            },
        ),
        (
            "datefrom",
            {
                "type": "string",
                "description": "The "
                "starting "
                "date/time "
                "for "
                "the "
                "review "
                "search "
                "period "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2023-01-01T00:00:00Z'). "
                "Must "
                "be "
                "earlier "
                "than "
                "'dateto'.",
            },
        ),
    ),
    (
        "arrivals",
        "Retrieves flight arrival information for Madrid Barajas Airport (IATA: MAD), including flight numbers, scheduled/actual arrival times, terminal information, and status updates. Returns arrivals for the current day by default, with optional filtering capabilities.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Filter "
                "arrivals "
                "by "
                "specific "
                "date "
                "(YYYY-MM-DD "
                "format). "
                "Defaults "
                "to "
                "current "
                "date "
                "when "
                "omitted. "
                "Accepts "
                "dates "
                "up "
                "to "
                "two "
                "days "
                "ahead "
                "of "
                "current "
                "date.",
                "default": "current_date",
            },
        ),
    ),
    (
        "articles_list",
        "Retrieves a filtered list of articles with options for sorting, pagination, and content-based filtering. Useful for analyzing market sentiment through news articles related to specific tickers, domains, or date ranges.",
    ): (
        (
            "date_to",
            {
                "type": "string",
                "description": "Only "
                "return "
                "articles "
                "published "
                "before "
                "this "
                "date. "
                "Format: "
                "%yyyy-%mm-%dd.",
            },
        ),
    ),
    (
        "available",
        "Retrieves available time slots for a schedule in SuperSaaS online booking system. Returns free time periods considering schedule constraints, appointment duration, resources, and availability settings.",
    ): (
        (
            "is_from",
            {
                "type": "string",
                "description": "Start "
                "date/time "
                "for "
                "availability "
                "check "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2024-02-20T15:00:00Z'). "
                "Defaults "
                "to "
                "current "
                "time "
                "if "
                "not "
                "provided",
                "default": "",
            },
        ),
    ),
    (
        "baseball_live_matches",
        "Retrieves current live baseball matches with real-time scores, game statistics, and betting odds data. Provides up-to-date information for wagering opportunities and match tracking. Returns matches across major leagues with configurable filters.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Filter "
                "matches "
                "by "
                "date "
                "(YYYY-MM-DD "
                "format). "
                "Defaults "
                "to "
                "current "
                "date "
                "if "
                "not "
                "provided.",
                "default": "",
            },
        ),
    ),
    (
        "baseball_livescores",
        "Retrieves current live scores, game status, and statistics for ongoing baseball matches worldwide. Returns real-time updates including current inning, runs, hits, and player performance data.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "to "
                "fetch "
                "games "
                "for "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Defaults "
                "to "
                "current "
                "date "
                "when "
                "unspecified.",
                "default": "current_date",
            },
        ),
    ),
    (
        "baseball_predictions_by_day",
        "Retrieves comprehensive baseball match schedules and predictive analytics for a specified date. Includes game outcome probabilities, score forecasts, and player performance predictions across multiple betting markets. Ideal for sports betting analysis and game preparation.",
    ): (
        (
            "day",
            {
                "type": "string",
                "description": "Date "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD) "
                "specifying "
                "the "
                "day "
                "to "
                "retrieve "
                "baseball "
                "match "
                "predictions "
                "for. "
                "Predictions "
                "are "
                "only "
                "available "
                "for "
                "upcoming "
                "dates.",
            },
        ),
    ),
    (
        "basketball_predictions_by_day",
        "Retrieves basketball match schedules and predictive analytics for a specified date. Provides game predictions including score forecasts, win probabilities, and betting market insights. Ideal for sports analysts and betting applications requiring structured sports data.",
    ): (
        (
            "day",
            {
                "type": "string",
                "description": "The "
                "date "
                "in "
                "YYYY-MM-DD "
                "format "
                "to "
                "retrieve "
                "match "
                "predictions "
                "for. "
                "Must "
                "be "
                "a "
                "valid "
                "calendar "
                "date.",
            },
        ),
    ),
    (
        "between_checker",
        "Checks if a given date is between two specified dates using the DateClock API.",
    ): (
        (
            "is_from",
            {
                "type": "string",
                "description": "The start date in the format 'YYYY-MM-DD'.",
                "default": "1980-06-06 00:00:00",
            },
        ),
    ),
    (
        "billboard_global_excl_us",
        "Fetches the Billboard Global Excl. US chart for a specified date using the RapidAPI service.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "for "
                "which "
                "to "
                "retrieve "
                "the "
                "Billboard "
                "Global "
                "Excl. "
                "US "
                "chart, "
                "formatted "
                "as "
                "'YYYY-MM-DD'. "
                "Available "
                "data "
                "starts "
                "from "
                "September "
                "19, "
                "2020.",
                "default": "2020-09-19",
            },
        ),
    ),
    ("book_appointment", "Book an appointment for a service"): (
        ("date", {"type": "string", "description": "Date of the appointment"}),
    ),
    ("book_table", "Book a table at a restaurant"): (
        (
            "date",
            {
                "type": "string",
                "description": "The date of the reservation in YYYY-MM-DD format",
            },
        ),
    ),
    (
        "bookretreats_com",
        "Submits a wellness retreat inquiry to bookretreats.com. This function facilitates contacting the platform with specific retreat preferences, dates, and contact details to request booking information or assistance.",
    ): (
        (
            "start_date",
            {
                "type": "string",
                "description": "Preferred retreat start date in YYYY-MM-DD format",
            },
        ),
    ),
    (
        "btts_predictions_by_country_by_date_range",
        "Retrieves BTTS (Both Teams To Score) predictions for football matches played in a specified country within a defined date range. Provides statistical insights including match probabilities, odds, and historical performance metrics for sports betting analysis.",
    ): (
        (
            "dateto",
            {
                "type": "string",
                "description": "End "
                "date "
                "of "
                "the "
                "prediction "
                "period "
                "(inclusive). "
                "Format: "
                "YYYY-MM-DD",
            },
        ),
        (
            "datefrom",
            {
                "type": "string",
                "description": "Start "
                "date "
                "of "
                "the "
                "prediction "
                "period "
                "(inclusive). "
                "Format: "
                "YYYY-MM-DD",
            },
        ),
    ),
    (
        "calculate_route_duration",
        "Calculate the duration of a route based on traffic",
    ): (
        (
            "departure_time",
            {
                "type": "string",
                "description": "The "
                "departure "
                "time "
                "for "
                "the "
                "route "
                "in "
                "YYYY-MM-DD "
                "HH:MM "
                "format",
            },
        ),
    ),
    (
        "calls",
        "Retrieves call records (incoming, outgoing, and missed) for the organization, ordered from most recent to oldest. Provides details about call direction, participants, timing, duration, and type. Use filters to narrow results by relayer, phone number, timestamp, duration, or call type.",
    ): (
        (
            "created_on",
            {
                "type": "string",
                "description": "Filter "
                "calls "
                "by "
                "creation "
                "time "
                "using "
                "comparison "
                "operators. "
                "Format "
                "as "
                "'operator:datetime' "
                "where "
                "operator "
                "is "
                "'before' "
                "or "
                "'after', "
                "and "
                "datetime "
                "is "
                "ISO "
                "8601 "
                "(e.g., "
                "'after:2023-09-20T14:30:00Z').",
            },
        ),
    ),
    ("check_bus_schedule", "Check the schedule of a bus route"): (
        ("date", {"type": "string", "description": "The date to check the schedule"}),
    ),
    ("check_movie_schedule", "Check the schedule of movies in a specific theater"): (
        (
            "date",
            {
                "type": "string",
                "description": "The date for which to check the schedule",
            },
        ),
    ),
    ("check_movie_timing", "Check the timing of a movie in a theater"): (
        (
            "date",
            {
                "type": "string",
                "format": "date",
                "description": "The date to check for movie timing",
            },
        ),
    ),
    ("check_train_schedule", "Check the schedule of a train"): (
        ("date", {"type": "string", "description": "The date of the train schedule"}),
    ),
    (
        "clickbank",
        "Manages email marketing campaigns and subscriber data through ClickBank University's API. Enables campaign creation, scheduling, analytics tracking, and list management for affiliate marketing workflows.",
    ): (
        (
            "send_time",
            {
                "type": "string",
                "description": "ISO "
                "8601 "
                "timestamp "
                "for "
                "scheduled "
                "delivery. "
                "Defaults "
                "to "
                "immediate "
                "sending "
                "if "
                "not "
                "specified.",
                "default": "now",
            },
        ),
    ),
    (
        "companies_id_jobs",
        "Retrieves the latest job postings for a given company identifier since a specified timestamp.",
    ): (
        (
            "since",
            {
                "type": "string",
                "description": "A "
                "timestamp "
                "to "
                "filter "
                "job "
                "postings. "
                "Defaults "
                "to "
                "'2017-01-01'.",
                "default": "2017-01-01",
            },
        ),
    ),
    (
        "copy_of_endpoint_site_abcr",
        "Interacts with the ABCR social media platform API to perform content creation or data retrieval operations. Supports customizable parameters for content formatting, visibility settings, and media integration.",
    ): (
        (
            "schedule_time",
            {
                "type": "string",
                "description": "ISO "
                "8601 "
                "timestamp "
                "for "
                "scheduling "
                "posts. "
                "If "
                "not "
                "specified, "
                "post "
                "will "
                "publish "
                "immediately",
                "default": None,
            },
        ),
    ),
    (
        "create_ad_campaign",
        "Creates and configures a new advertising campaign with specified parameters. Returns campaign ID and configuration details. Use this function to programmatically set up targeted advertising campaigns with customizable budgets, audiences, and content.",
    ): (
        (
            "end_date",
            {
                "type": "string",
                "description": "Campaign "
                "end "
                "date "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DDTHH:MM:SSZ)",
            },
        ),
        (
            "start_date",
            {
                "type": "string",
                "description": "Campaign "
                "start "
                "date "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DDTHH:MM:SSZ)",
            },
        ),
    ),
    ("create_event_reminder", "Create a reminder for an upcoming event"): (
        ("event_date", {"type": "string", "description": "The date of the event"}),
    ),
    ("create_new_event", "Create a new event in the calendar"): (
        (
            "start_time",
            {"type": "string", "description": "The start time of the event"},
        ),
        ("end_time", {"type": "string", "description": "The end time of the event"}),
    ),
    ("create_todo_task", "Create a new todo task"): (
        ("due_date", {"type": "string", "description": "The due date of the task"}),
    ),
    (
        "daily_3",
        "Retrieves historical Daily 3 lottery results from the California Lottery, including drawn numbers and associated dates. Provides data for tracking patterns, verifying past results, or financial record-keeping purposes.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "for "
                "which "
                "lottery "
                "results "
                "should "
                "be "
                "retrieved, "
                "in "
                "YYYY-MM-DD "
                "format. "
                "If "
                "omitted, "
                "returns "
                "the "
                "most "
                "recent "
                "available "
                "results.",
                "default": "latest",
            },
        ),
    ),
    (
        "daily_gold_rates",
        "Retrieves the latest gold rates in India for a specific date, including prices for different carat weights (e.g., 24K, 22K). Useful for jewelry transactions, investment tracking, and market analysis. Rates are provided in Indian Rupees (INR) per gram.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "to "
                "retrieve "
                "gold "
                "rates "
                "for, "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Must "
                "be "
                "a "
                "past "
                "or "
                "current "
                "date "
                "(cannot "
                "be "
                "in "
                "the "
                "future). "
                "Example: "
                "'2023-10-25'",
            },
        ),
    ),
    (
        "daily_predictions",
        "Fetches daily football predictions using various filters and pagination support.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "for "
                "filtering "
                "predictions "
                "in "
                "the "
                "format "
                "'YYYY-MM-DD'. "
                "Example: "
                "'2022-08-13'.",
            },
        ),
    ),
    (
        "daily_sentiment",
        "Gets the daily sentiment score for a given asset symbol and date using the SentiTrade API.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "for "
                "which "
                "to "
                "retrieve "
                "the "
                "sentiment "
                "score "
                "in "
                "'YYYY-MM-DD' "
                "format.",
                "default": "2023-01-01",
            },
        ),
    ),
    (
        "datedif",
        "Calculates the difference between two dates in years, months, days, hours, minutes, seconds, and microseconds. Returns the time interval components between start_date and end_date, supporting precise temporal calculations for financial and temporal analysis.",
    ): (
        (
            "end_date",
            {
                "type": "string",
                "description": "End "
                "date/time "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2024-03-20', "
                "'2024-03-20T15:30:00', "
                "or "
                "'2024-03-20T15:30:00Z'). "
                "Must "
                "be "
                "equal "
                "to "
                "or "
                "later "
                "than "
                "start_date.",
            },
        ),
    ),
    (
        "departures",
        "Retrieves flight departure information from Madrid-Barajas Airport (MAD) for the current day and next business day. Returns structured data including flight numbers, departure times, destinations, terminals, and status updates.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "for "
                "departure "
                "search "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Defaults "
                "to "
                "current "
                "date "
                "when "
                "omitted.",
                "default": "today",
            },
        ),
    ),
    (
        "earthquakes_by_date",
        "Retrieves earthquake data within a specified date range, filtered by optional criteria such as magnitude, intensity, or geographic location. Returns results sorted by recency, with support for pagination and location-based queries.",
    ): (
        (
            "enddate",
            {
                "type": "string",
                "description": "End "
                "date "
                "for "
                "the "
                "search "
                "period "
                "in "
                "YYYY-MM-DD "
                "format "
                "(UTC "
                "midnight). "
                "Must "
                "be "
                "greater "
                "than "
                "or "
                "equal "
                "to "
                "startdate.",
            },
        ),
        (
            "startdate",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "the "
                "search "
                "period "
                "in "
                "YYYY-MM-DD "
                "format "
                "(UTC "
                "midnight). "
                "Required "
                "for "
                "all "
                "queries.",
            },
        ),
    ),
    (
        "edate",
        "Calculates a date that is a specified number of months before or after a given start date, maintaining the same day of the month. This function replicates Excel's EDATE function behavior, making it ideal for financial calculations, subscription management, and date-based scheduling tasks.",
    ): (
        (
            "start_date",
            {
                "type": "string",
                "description": "Initial "
                "date "
                "value "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD "
                "or "
                "YYYY-MM-DDTHH:MM:SS). "
                "The "
                "calculation "
                "preserves "
                "the "
                "day "
                "of "
                "the "
                "month "
                "and "
                "time "
                "components "
                "when "
                "possible. "
                "If "
                "the "
                "target "
                "month "
                "has "
                "fewer "
                "days "
                "than "
                "the "
                "start "
                "date's "
                "day, "
                "the "
                "result "
                "will "
                "use "
                "the "
                "last "
                "day "
                "of "
                "the "
                "target "
                "month.",
            },
        ),
    ),
    (
        "endpoint_a",
        "Executes a PUT request to Advertising API endpoint A for managing ad campaigns. Use this function to update campaign configurations, adjust budgets, or modify targeting parameters in the advertising system.",
    ): (
        (
            "update_time",
            {
                "type": "string",
                "description": "Scheduled "
                "update "
                "time "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2024-03-20T14:30:00Z')",
                "default": "immediate",
            },
        ),
    ),
    (
        "eomonth",
        "Computes and returns the date of the last day of a given month based on a specified start date and the number of months to adjust.",
    ): (
        (
            "start_date",
            {
                "type": "string",
                "description": "The "
                "start "
                "date "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD), "
                "with "
                "or "
                "without "
                "time "
                "information.",
                "default": "2021-09-21",
            },
        ),
    ),
    (
        "equity_daily",
        "Retrieves end-of-day (daily) time series data for a specified equity symbol, including date, open/high/low/close prices, and trading volume. Provides adjusted prices when requested and supports date range filtering.",
    ): (
        (
            "to",
            {
                "type": "string",
                "description": "End "
                "date "
                "for "
                "the "
                "query "
                "in "
                "YYYY-MM-DD "
                "format "
                "(inclusive). "
                "Must "
                "be "
                "later "
                "than "
                "or "
                "equal "
                "to "
                "'is_from' "
                "date.",
            },
        ),
        (
            "is_from",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "the "
                "query "
                "in "
                "YYYY-MM-DD "
                "format "
                "(inclusive). "
                "Must "
                "be "
                "earlier "
                "than "
                "or "
                "equal "
                "to "
                "'to' "
                "date.",
            },
        ),
    ),
    (
        "equity_intraday",
        "Retrieve intraday time series data (Date, Open, High, Low, Close, Volume) for a specific symbol based on given parameters.",
    ): (
        (
            "is_from",
            {
                "type": "string",
                "description": "The "
                "start "
                "date "
                "and "
                "time "
                "of "
                "the "
                "query "
                "in "
                "formats "
                "like "
                "**YYYY-mm-dd "
                "HH:MM** "
                "or "
                "**YYYY-mm-dd**.",
                "default": "2020-04-21 10:00",
            },
        ),
        (
            "to",
            {
                "type": "string",
                "description": "The "
                "end "
                "date "
                "and "
                "time "
                "of "
                "the "
                "query "
                "in "
                "formats "
                "like "
                "**YYYY-mm-dd "
                "HH:MM** "
                "or "
                "**YYYY-mm-dd**.",
                "default": "2020-04-21 10:30",
            },
        ),
    ),
    (
        "events",
        "Retrieves cryptocurrency-related events filtered by coins, categories, date ranges, and other criteria. Events cannot be retrieved before November 25, 2017. The response includes a 'can_occur_before' field indicating potential earlier occurrences of recurring events. This function supports pagination and sorting.",
    ): (
        (
            "daterangeend",
            {
                "type": "string",
                "description": "End "
                "date "
                "for "
                "event "
                "filtering "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Default "
                "is "
                "the "
                "date "
                "of "
                "the "
                "furthest "
                "event "
                "available.",
                "default": "furthest_event_date",
            },
        ),
        (
            "daterangestart",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "event "
                "filtering "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Default "
                "is "
                "today's "
                "date.",
                "default": "today",
            },
        ),
    ),
    (
        "events_games",
        "Retrieves sports events and associated betting data for a specific sport and date. Returns event details including current odds, markets, scores, and team information from specified sportsbooks. Supports timezone offsets for date grouping and expanded market data retrieval.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "ISO "
                "8601 "
                "date "
                "string "
                "(YYYY-MM-DD) "
                "for "
                "which "
                "to "
                "retrieve "
                "events. "
                "If "
                "no "
                "offset "
                "is "
                "provided, "
                "date "
                "is "
                "interpreted "
                "as "
                "UTC.",
            },
        ),
    ),
    (
        "events_list",
        "Retrieves upcoming sports events filtered by sport type and date range, adjusted for the specified time zone. Returns event details including teams, schedules, and location information.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Filter "
                "events "
                "starting "
                "from "
                "this "
                "date/time "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2024-03-20T15:00:00Z')",
            },
        ),
    ),
    (
        "fantasy_5",
        "Retrieves historical results for the Fantasy 5 lottery, including draw dates, winning numbers, prize amounts, and jackpot information. Useful for analyzing lottery patterns or verifying historical outcomes.",
    ): (
        (
            "start_date",
            {
                "type": "string",
                "description": "Earliest "
                "date "
                "for "
                "lottery "
                "results "
                "(inclusive). "
                "Format: "
                "YYYY-MM-DD. "
                "If "
                "omitted, "
                "defaults "
                "to "
                "30 "
                "days "
                "before "
                "the "
                "end_date.",
                "default": None,
            },
        ),
        (
            "end_date",
            {
                "type": "string",
                "description": "Latest "
                "date "
                "for "
                "lottery "
                "results "
                "(inclusive). "
                "Format: "
                "YYYY-MM-DD. "
                "If "
                "omitted, "
                "defaults "
                "to "
                "the "
                "current "
                "date.",
                "default": None,
            },
        ),
    ),
    ("find_events", "Find events happening near a specified location"): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "desired "
                "date "
                "of "
                "the "
                "events "
                "in "
                "the "
                "format "
                "'YYYY-MM-DD'",
            },
        ),
    ),
    (
        "fixtures",
        "Retrieves sports fixtures data with multiple filtering options. Supports filtering by date ranges, league IDs, team IDs, match status, and more. Update frequency: Every 15 seconds. Recommended usage: 1 call per minute for active leagues/teams/fixtures, otherwise 1 call per day.",
    ): (
        (
            "is_from",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "fixtures "
                "query "
                "(inclusive). "
                "Format: "
                '"YYYY-MM-DD"',
            },
        ),
        (
            "to",
            {
                "type": "string",
                "description": "End "
                "date "
                "for "
                "fixtures "
                "query "
                "(inclusive). "
                "Format: "
                '"YYYY-MM-DD"',
            },
        ),
    ),
    (
        "fixtures_by_date_country",
        "Retrieves sports fixtures for a specific country and date. Provides match schedules, team information, and event details for the requested country and date combination.",
    ): (
        (
            "t",
            {
                "type": "string",
                "description": "Date "
                "for "
                "which "
                "to "
                "retrieve "
                "fixtures "
                "in "
                "YYYY-MM-DD "
                "format. "
                "If "
                "omitted, "
                "defaults "
                "to "
                "current "
                "date.",
                "default": "",
            },
        ),
    ),
    (
        "flight_search_v2",
        "Searches for available flights based on origin, destination, date, and passenger details. Returns flight options with pricing and availability. Use this function for real-time flight fare queries and booking preparation.",
    ): (
        (
            "date",
            {"type": "string", "description": "Departure date in YYYY-MM-DD format"},
        ),
    ),
    (
        "flights_to_country",
        "Searches for flights from a specified airport to a destination country, returning estimated lowest prices found in the last 8 days. Ideal for price tracking and itinerary planning.",
    ): (
        (
            "departuredate",
            {"type": "string", "description": "Departure date in YYYY-MM-DD format"},
        ),
        (
            "returndate",
            {
                "type": "string",
                "description": "Return "
                "date "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Omit "
                "for "
                "one-way "
                "trips",
            },
        ),
    ),
    (
        "football_predictions_by_day",
        "Retrieves football match schedules and predictive analytics for a specified date. Provides comprehensive market predictions including match outcomes, goal probabilities, and betting insights.",
    ): (
        (
            "day",
            {
                "type": "string",
                "description": "Date "
                "string "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD) "
                "specifying "
                "the "
                "day "
                "for "
                "which "
                "match "
                "data "
                "and "
                "predictions "
                "should "
                "be "
                "retrieved",
            },
        ),
    ),
    (
        "forex_intraday",
        "Fetches intraday time series data (Date, Open, High, Low, Close, Volume) for a given currency pair.",
    ): (
        (
            "to",
            {
                "type": "string",
                "description": "The "
                "query "
                "end "
                "date "
                "and "
                "time "
                "in "
                "the "
                "format "
                "`YYYY-mm-dd "
                "HH:MM` "
                "or "
                "just "
                "`YYYY-mm-dd`.",
                "default": "2020-04-21 10:30",
            },
        ),
    ),
    (
        "fx",
        "Converts a specified `amount` of currency from one type to another using the ForexGo API, with an optional historical date.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "for "
                "historical "
                "conversion "
                "rates "
                "in "
                "ISO "
                "format "
                "(YYYY-MM-DDTHH:mm:ss.sssZ). "
                "Defaults "
                "to "
                "None "
                "for "
                "real-time "
                "rates.",
            },
        ),
    ),
    (
        "games",
        "Retrieves sports game data with optional filtering by league, team, date, and game status. Provides comprehensive game information including odds, scores, and schedule details.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Filter "
                "games "
                "by "
                "one "
                "or "
                "two "
                "comma-separated "
                "dates "
                "in "
                "YYYY-MM-DD "
                "or "
                "ISO "
                "8601 "
                "format. "
                "Single "
                "date "
                "matches "
                "exact "
                "date; "
                "two "
                "dates "
                "define "
                "a "
                "range "
                "(inclusive). "
                "Defaults "
                "to "
                "current "
                "date "
                "when "
                "empty.",
                "default": "today",
            },
        ),
    ),
    (
        "get_3_fluctuation_endpoint",
        "Fetches the percentage change in exchange rates for specified currencies over a given time period.",
    ): (
        (
            "start_date",
            {
                "type": "string",
                "description": "The "
                "start "
                "date "
                "for "
                "the "
                "time "
                "period "
                "of "
                "the "
                "fluctuation "
                "data.",
                "default": "2023-01-01",
            },
        ),
        (
            "end_date",
            {
                "type": "string",
                "description": "The "
                "end "
                "date "
                "for "
                "the "
                "time "
                "period "
                "of "
                "the "
                "fluctuation "
                "data.",
                "default": "2023-02-28",
            },
        ),
    ),
    (
        "get_all_events_for_given_day",
        "Retrieves all scheduled events for a specified day in the project management system. Returns event details including title, time, participants, and status. Useful for calendar management and project planning.",
    ): (
        (
            "startdate",
            {
                "type": "string",
                "description": "The "
                "date "
                "to "
                "retrieve "
                "events "
                "for, "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD). "
                "If "
                "not "
                "provided, "
                "defaults "
                "to "
                "the "
                "current "
                "date.",
                "default": "current date (YYYY-MM-DD)",
            },
        ),
    ),
    (
        "get_all_events_in_given_window",
        "Retrieves events occurring within a specified date range window. This function is useful for filtering events when building calendar views, project timelines, or analyzing activity within specific timeframes.",
    ): (
        (
            "enddate",
            {
                "type": "string",
                "description": "The "
                "end "
                "of "
                "the "
                "date "
                "range "
                "window "
                "(exclusive). "
                "Must "
                "be "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DDTHH:MM:SSZ). "
                "If "
                "not "
                "provided, "
                "defaults "
                "to "
                "7 "
                "days "
                "after "
                "the "
                "start "
                "date.",
                "default": "2023-10-08T00:00:00Z",
            },
        ),
        (
            "startdate",
            {
                "type": "string",
                "description": "The "
                "start "
                "of "
                "the "
                "date "
                "range "
                "window "
                "(inclusive). "
                "Must "
                "be "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DDTHH:MM:SSZ). "
                "If "
                "not "
                "provided, "
                "defaults "
                "to "
                "the "
                "current "
                "date "
                "and "
                "time.",
                "default": "2023-10-01T00:00:00Z",
            },
        ),
    ),
    (
        "get_all_sports_predictions",
        "Retrieves sports betting predictions filtered by sport, date, competition, country, or market. Returns predictive analytics for upcoming matches or events. Use Get Sports endpoint to obtain valid sport IDs before calling this function.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "to "
                "filter "
                "predictions "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Required "
                "parameter "
                "for "
                "all "
                "requests.",
            },
        ),
    ),
    (
        "get_all_the_hotels",
        "Retrieves a comprehensive list of available hotels with their details including location, pricing, amenities, and availability. Useful for travel planning, hotel comparison, and accommodation booking.",
    ): (
        (
            "check_in_date",
            {
                "type": "string",
                "description": "Date "
                "to "
                "check "
                "hotel "
                "availability "
                "from, "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD). "
                "Default "
                "is "
                "today's "
                "date.",
            },
        ),
        (
            "check_out_date",
            {
                "type": "string",
                "description": "Date "
                "to "
                "check "
                "hotel "
                "availability "
                "until, "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD). "
                "Default "
                "is "
                "one "
                "day "
                "after "
                "check-in.",
            },
        ),
    ),
    (
        "get_api_sentiments",
        "Retrieves aggregated bullish and bearish sentiment values for a specified stock symbol or market-wide sentiment (using 'MARKET' as the symbol) over a defined date range and aggregation period. Returns raw sentiment scores that reflect market psychology derived from social media and news sources. Useful for financial analysis of market trends and investor behavior.",
    ): (
        (
            "to",
            {
                "type": "string",
                "description": "End "
                "date/time "
                "for "
                "sentiment "
                "analysis "
                "range. "
                "Format "
                "as "
                "'YYYY-MM-DD' "
                "or "
                "'YYYY-MM-DD "
                "HH:MM:SS'. "
                "If "
                "omitted, "
                "defaults "
                "to "
                "current "
                "date/time "
                "or "
                "latest "
                "available "
                "data "
                "based "
                "on "
                "API "
                "access "
                "level.",
            },
        ),
        (
            "is_from",
            {
                "type": "string",
                "description": "Start "
                "date/time "
                "for "
                "sentiment "
                "analysis "
                "range. "
                "Format "
                "as "
                "'YYYY-MM-DD' "
                "or "
                "'YYYY-MM-DD "
                "HH:MM:SS'. "
                "If "
                "omitted, "
                "defaults "
                "to "
                "earliest "
                "available "
                "data "
                "based "
                "on "
                "API "
                "access "
                "level.",
            },
        ),
    ),
    ("get_calendar_events", "Get a list of upcoming calendar events"): (
        (
            "end_date",
            {
                "type": "string",
                "description": "The end date of the events",
                "format": "date",
            },
        ),
        (
            "start_date",
            {
                "type": "string",
                "description": "The start date of the events",
                "format": "date",
            },
        ),
    ),
    (
        "get_calls",
        "Retrieves call data for a specified account with support for pagination, filtering, and date-range queries. This function enables monitoring of incoming calls and historical analysis of call records including caller/called numbers, timestamps, and source information.",
    ): (
        (
            "start_date",
            {
                "type": "string",
                "description": "ISO "
                "8601 "
                "start "
                "date/time "
                "(YYYY-MM-DDTHH:MM:SSZ) "
                "for "
                "filtering "
                "calls. "
                "Returns "
                "calls "
                "occurring "
                "at "
                "or "
                "after "
                "this "
                "timestamp.",
                "default": "",
            },
        ),
        (
            "end_date",
            {
                "type": "string",
                "description": "ISO "
                "8601 "
                "end "
                "date/time "
                "(YYYY-MM-DDTHH:MM:SSZ) "
                "for "
                "filtering "
                "calls. "
                "Returns "
                "calls "
                "occurring "
                "at "
                "or "
                "before "
                "this "
                "timestamp.",
                "default": "",
            },
        ),
    ),
    ("get_concert_info", "Get information about upcoming concerts"): (
        ("date", {"type": "string", "description": "The date of the concert"}),
    ),
    (
        "get_draw_result",
        "Retrieve the draw result for a specified game in a given region and on a specific date.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "of "
                "the "
                "draw "
                "result "
                "to "
                "retrieve, "
                "formatted "
                "as "
                "'YYYY-MM-DD'.",
                "default": "2023-01-01",
            },
        ),
    ),
    (
        "get_ecoindex_analysis_list_version_ecoindexes_get",
        "Fetches a list of ecoindex analysis based on the specified version and query filters. The results are ordered by ascending date.",
    ): (
        (
            "date_to",
            {
                "type": "string",
                "description": "End date for filtering results (format: 'YYYY-MM-DD').",
            },
        ),
        (
            "date_from",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "filtering "
                "results "
                "(format: "
                "'YYYY-MM-DD').",
            },
        ),
    ),
    (
        "get_economic_calendar_news_over_a_period_of_time",
        "Fetches economic calendar news within a specified time period.",
    ): (
        (
            "time_start",
            {
                "type": "string",
                "description": "The "
                "start "
                "date "
                "and "
                "time "
                "for "
                "the "
                "news "
                "fetch "
                "period "
                "in "
                "ISO "
                "8601 "
                "format. "
                "Default "
                "is "
                "'2022-12-20 "
                "17:34:58+00:00'.",
                "default": "2022-12-20 17:34:58+00:00",
            },
        ),
        (
            "time_finish",
            {
                "type": "string",
                "description": "The "
                "end "
                "date "
                "and "
                "time "
                "for "
                "the "
                "news "
                "fetch "
                "period "
                "in "
                "ISO "
                "8601 "
                "format. "
                "Default "
                "is "
                "'2023-02-13 "
                "19:34:58+00:00'.",
                "default": "2023-02-13 19:34:58+00:00",
            },
        ),
    ),
    (
        "get_fancy_re_settle_market",
        "Retrieves markets that experienced resettlement within a specified time window. Resettlement occurs when odds or betting lines are recalculated for sporting events, typically due to changing conditions like match progress or weather.",
    ): (
        (
            "settle_dt_start",
            {
                "type": "string",
                "description": "Start "
                "of "
                "the "
                "time "
                "window "
                "(inclusive) "
                "for "
                "filtering "
                "resettled "
                "markets, "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2024-03-17T14:30:00Z'). "
                "Defaults "
                "to "
                "72 "
                "hours "
                "before "
                "current "
                "time "
                "if "
                "not "
                "provided.",
                "default": "current_time - 72h",
            },
        ),
        (
            "settle_dt_end",
            {
                "type": "string",
                "description": "End "
                "of "
                "the "
                "time "
                "window "
                "(exclusive) "
                "for "
                "filtering "
                "resettled "
                "markets, "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2024-03-20T14:30:00Z'). "
                "Defaults "
                "to "
                "current "
                "time "
                "if "
                "not "
                "provided.",
                "default": "current_time",
            },
        ),
    ),
    (
        "get_futured_playlists",
        "Fetch featured playlists for a specific country and timestamp from Spotify.",
    ): (
        (
            "timestamp",
            {
                "type": "string",
                "description": "Date "
                "of "
                "the "
                "featured "
                "playlists "
                "in "
                "the "
                "format "
                "'yyyy-mm-dd'.",
                "default": "2022-03-23",
            },
        ),
    ),
    ("get_horoscope", "Get the horoscope for a specific zodiac sign"): (
        (
            "date",
            {
                "type": "string",
                "description": "The date for which horoscope is required",
            },
        ),
    ),
    (
        "get_list_of_users",
        "Retrieves a paginated list of users with optional filtering by creation date. Allows clients to specify date ranges and control result pagination for efficient data retrieval.",
    ): (
        (
            "created_at_lt",
            {
                "type": "string",
                "description": "Filter "
                "users "
                "created "
                "before "
                "this "
                "ISO "
                "8601 "
                "date "
                "string "
                "(e.g., "
                "'2024-01-15T09:30:00Z')",
            },
        ),
        (
            "created_at_gt",
            {
                "type": "string",
                "description": "Filter "
                "users "
                "created "
                "after "
                "this "
                "ISO "
                "8601 "
                "date "
                "string "
                "(e.g., "
                "'2024-01-15T09:30:00Z')",
            },
        ),
    ),
    (
        "get_live_cockfight_streams",
        "Provides access to live cockfighting broadcasts and related information from tructiepdaga.tv. Retrieves current events, schedules, regional matches, and poultry farming knowledge. Useful for users seeking real-time streaming data or cultural insights about cockfighting traditions in Southeast Asia.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "filter "
                "for "
                "scheduled "
                "matches "
                "(YYYY-MM-DD "
                "format). "
                "Defaults "
                "to "
                "current "
                "day.",
                "default": "today",
            },
        ),
    ),
    ("get_local_events", "Get local events happening in the user's area"): (
        (
            "date_range.end_date",
            {"type": "string", "description": "The end date of the event range"},
        ),
        (
            "date_range.start_date",
            {"type": "string", "description": "The start date of the event range"},
        ),
    ),
    (
        "get_lottery_result_by_date",
        "Fetches the lottery results for the given date using the RapidAPI service.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "for "
                "which "
                "to "
                "retrieve "
                "the "
                "lottery "
                "results, "
                "formatted "
                "as "
                "'YYYY-MM-DD'.",
                "default": "2022-10-16",
            },
        ),
    ),
    (
        "get_market_news",
        "Retrieves financial market news articles with summaries, sources, and relevance scores. Returns data filtered by date range, market category, and sorted by relevance. Useful for tracking market trends, sentiment analysis, and investment research.",
    ): (
        (
            "endday",
            {
                "type": "string",
                "description": "End "
                "date "
                "for "
                "filtering "
                "news "
                "articles, "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Only "
                "articles "
                "published "
                "on "
                "or "
                "before "
                "this "
                "date "
                "will "
                "be "
                "included.",
            },
        ),
        (
            "startday",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "filtering "
                "news "
                "articles, "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Only "
                "articles "
                "published "
                "on "
                "or "
                "after "
                "this "
                "date "
                "will "
                "be "
                "included.",
            },
        ),
    ),
    (
        "get_market_turnover",
        "Retrieves closing market turnover data from the National Stock Exchange (NSE) for a specified date. Contains total trading volume and value metrics across equity segments.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "to "
                "retrieve "
                "market "
                "turnover "
                "data "
                "for, "
                "formatted "
                "as "
                "YYYY-MM-DD. "
                "Must "
                "be "
                "a "
                "valid "
                "trading "
                "day "
                "with "
                "available "
                "market "
                "data.",
            },
        ),
    ),
    (
        "get_matches_in_play",
        "Retrieves real-time data for football matches currently in progress, including match status, current scores, time elapsed, and participating teams. Ideal for live score updates, sports betting applications, or real-time analytics.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Filter "
                "matches "
                "by "
                "date "
                "in "
                "'YYYY-MM-DD' "
                "format. "
                "Defaults "
                "to "
                "current "
                "date "
                "if "
                "not "
                "specified.",
                "default": "current_date",
            },
        ),
    ),
    (
        "get_next_win_draw_double_chance_predictions",
        "Retrieves sports match predictions for upcoming events occurring on a specified date. This function provides win, draw, and double chance predictions generated through advanced statistical analysis of team performance, player statistics, historical data, and other relevant factors. Use this tool to obtain actionable insights for sports betting or match outcome analysis.",
    ): (
        (
            "event_date",
            {
                "type": "string",
                "description": "Date "
                "of "
                "the "
                "event "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Predictions "
                "are "
                "generated "
                "for "
                "matches "
                "scheduled "
                "to "
                "occur "
                "on "
                "this "
                "date.",
            },
        ),
    ),
    (
        "get_positions_for_body",
        "Retrieves astronomical position data for a specified celestial body over a date range, relative to an observer's location. Returns tabular data including coordinates, distance, and visibility information. Ideal for astronomical research and celestial event planning.",
    ): (
        (
            "from_date",
            {
                "type": "string",
                "description": "Start "
                "date "
                "of "
                "the "
                "observation "
                "period "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD)",
            },
        ),
        (
            "to_date",
            {
                "type": "string",
                "description": "End "
                "date "
                "of "
                "the "
                "observation "
                "period "
                "(inclusive) "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD). "
                "Must "
                "be "
                "later "
                "than "
                "or "
                "equal "
                "to "
                "from_date.",
            },
        ),
    ),
    (
        "get_rsi_14_period_above_30",
        "Analyzes stock data to determine if the 14-period Relative Strength Index (RSI) is above 30 for the specified symbol and timeframe. Useful for identifying potential bullish momentum as RSI crossing above 30 often indicates emerging positive trends.",
    ): (
        (
            "period1",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "the "
                "analysis "
                "period "
                "in "
                "YYYY-MM-DD "
                "format. "
                "The "
                "date "
                "range "
                "must "
                "include "
                "at "
                "least "
                "100 "
                "data "
                "points "
                "based "
                "on "
                "the "
                "specified "
                "interval "
                "to "
                "ensure "
                "accurate "
                "RSI "
                "calculation.",
            },
        ),
        (
            "period2",
            {
                "type": "string",
                "description": "End "
                "date "
                "for "
                "the "
                "analysis "
                "period "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Must "
                "be "
                "after "
                "period1 "
                "and "
                "provide "
                "sufficient "
                "interval "
                "spacing "
                "to "
                "include "
                "100+ "
                "data "
                "points.",
            },
        ),
    ),
    (
        "get_scheduled_games_by_country",
        "Retrieves scheduled sports games for a specific country on a given date. Returns details about upcoming matches including teams, times, and venues when available.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD) "
                "to "
                "check "
                "for "
                "scheduled "
                "games. "
                "Must "
                "be "
                "a "
                "future "
                "date.",
            },
        ),
    ),
    (
        "get_scores_for_given_date",
        "Fetches the list of football match scores for a given date using the specified API.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "for "
                "which "
                "to "
                "fetch "
                "the "
                "football "
                "scores, "
                "in "
                "the "
                "format "
                "%Y-%m-%d "
                "(e.g., "
                "'2022-12-01').",
                "default": "2022-12-04",
            },
        ),
    ),
    (
        "get_soccer_prediction",
        "Retrieves soccer match predictions with optional filters. Returns data such as match outcomes, odds, and statistical insights for soccer matches across various leagues and markets.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "for "
                "which "
                "to "
                "retrieve "
                "predictions, "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Predictions "
                "are "
                "returned "
                "for "
                "matches "
                "occurring "
                "on "
                "this "
                "date.",
            },
        ),
    ),
    ("get_sports_scores", "Get the scores of recent sports matches"): (
        (
            "date",
            {
                "type": "string",
                "format": "date",
                "description": "The date for which scores are required",
            },
        ),
    ),
    (
        "get_sun_rise_and_sun_set_time",
        "Retrieves sunrise and sunset times for a specified date and geographical location. Includes timezone-aware times when a timezone is provided.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "for "
                "which "
                "to "
                "retrieve "
                "sunrise "
                "and "
                "sunset "
                "times, "
                "in "
                "YYYY-MM-DD "
                "format",
            },
        ),
    ),
    (
        "get_today_s_goals_predictions",
        "Retrieves goal predictions for sports events occurring on a specified date. This function provides actionable insights for match outcomes, enabling data-driven decision-making for sports analytics and betting scenarios.",
    ): (
        (
            "event_date",
            {
                "type": "string",
                "description": "Date "
                "of "
                "the "
                "events "
                "to "
                "retrieve "
                "predictions "
                "for, "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Must "
                "be "
                "provided "
                "to "
                "fetch "
                "predictions "
                "for "
                "a "
                "specific "
                "day.",
            },
        ),
    ),
    (
        "get_today_s_win_draw_double_chance_predictions",
        "Retrieves win, draw, and double chance predictions for sports events occurring on the specified date. Predictions are generated using advanced algorithms analyzing team performance metrics, player statistics, historical match data, and other relevant factors. Intended for current-day event forecasting.",
    ): (
        (
            "event_date",
            {
                "type": "string",
                "description": "Date "
                "for "
                "which "
                "predictions "
                "should "
                "be "
                "retrieved, "
                "formatted "
                "as "
                "YYYY-MM-DD. "
                "Must "
                "match "
                "the "
                "current "
                "date "
                "(UTC) "
                "as "
                "the "
                "function "
                "only "
                "supports "
                "same-day "
                "predictions.",
            },
        ),
    ),
    (
        "get_trend_keyword",
        "Retrieve trending keywords for a specific date and geographic location using the given RapidAPI key.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "for "
                "which "
                "to "
                "retrieve "
                "trending "
                "keywords. "
                "Defaults "
                "to "
                "'2023-05-18'.",
                "default": "2023-05-18",
            },
        ),
    ),
    (
        "getallcoupon",
        "Retrieves a list of available advertising coupons with optional filtering and sorting capabilities. This function enables users to query coupon inventory based on category, validity period, discount type, and other attributes.",
    ): (
        (
            "valid_before",
            {
                "type": "string",
                "format": "date",
                "description": "Only "
                "return "
                "coupons "
                "valid "
                "until "
                "this "
                "date "
                "(YYYY-MM-DD "
                "format). "
                "Must "
                "be "
                "after "
                "valid_after.",
            },
        ),
        (
            "valid_after",
            {
                "type": "string",
                "format": "date",
                "description": "Only "
                "return "
                "coupons "
                "valid "
                "from "
                "this "
                "date "
                "(YYYY-MM-DD "
                "format). "
                "Defaults "
                "to "
                "current "
                "date.",
                "default": "current_date",
            },
        ),
    ),
    (
        "getcompetitions",
        "Retrieves schedules of televised football matches in Brazil, including match times, participating teams, and broadcast network information. Ideal for users seeking to track live or upcoming televised games.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Filter "
                "matches "
                "by "
                "date "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Defaults "
                "to "
                "current "
                "day "
                "if "
                "not "
                "specified.",
                "default": "today",
            },
        ),
    ),
    (
        "getevents",
        "Retrieves information about televised football matches in Brazil, including match details, broadcast channels, and scheduling information. Useful for checking upcoming games, TV coverage, and competition schedules.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Filter "
                "matches "
                "by "
                "a "
                "specific "
                "date "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Defaults "
                "to "
                "current "
                "date "
                "if "
                "unspecified.",
                "default": "current_date",
            },
        ),
    ),
    (
        "getevents",
        "Retrieves sports events data filtered by league group, market type, and time range. Returns event details including teams, schedules, and associated odds information.",
    ): (
        (
            "enddate",
            {
                "type": "string",
                "description": "End "
                "date/time "
                "for "
                "event "
                "search "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2023-12-07T22:00:00Z')",
            },
        ),
        (
            "startdate",
            {
                "type": "string",
                "description": "Start "
                "date/time "
                "for "
                "event "
                "search "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2023-12-01T14:30:00Z')",
            },
        ),
    ),
    (
        "gethotels",
        "Searches for hotel listings based on location, dates, and guest preferences. Returns hotel details including pricing, availability, amenities, and ratings. Ideal for travel planning and accommodation booking scenarios.",
    ): (
        (
            "check_in_date",
            {
                "type": "string",
                "format": "date",
                "description": "Check-in "
                "date "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD). "
                "Defaults "
                "to "
                "current "
                "date "
                "if "
                "not "
                "specified",
                "default": "2023-10-10",
            },
        ),
        (
            "check_out_date",
            {
                "type": "string",
                "format": "date",
                "description": "Check-out "
                "date "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD). "
                "Defaults "
                "to "
                "next "
                "day "
                "if "
                "not "
                "specified",
                "default": "2023-10-11",
            },
        ),
    ),
    (
        "getting_historical_exchange_rate_s",
        "Retrieves historical exchange rates for specified currencies on a given date. Returns exchange rates for one or more target currencies (ISO 4217 format) relative to a base currency. If no target currencies are specified, returns rates for all available currencies. Ideal for financial analysis, historical comparisons, or international transaction planning.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "historical "
                "date "
                "to "
                "query "
                "exchange "
                "rates "
                "for, "
                "in "
                "YYYY-MM-DD "
                "format "
                "(e.g., "
                "'2023-12-31'). "
                "Must "
                "be "
                "a "
                "valid "
                "past "
                "date.",
            },
        ),
    ),
    (
        "hot_trending_songs_powered_by_twitter",
        "Fetch the HOT TRENDING SONGS POWERED BY TWITTER chart information for a specific date and range using the Toolbench RapidAPI key.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "for "
                "which "
                "to "
                "fetch "
                "the "
                "chart "
                "information, "
                "formatted "
                "as "
                "'YYYY-MM-DD'.",
                "default": "2021-11-06",
            },
        ),
    ),
    (
        "hotels_dynamic",
        "Retrieves hotel availability, pricing, and details with dynamic search parameters. Use this function to search for hotel options based on location, dates, occupancy, and regional settings. Returns real-time data including rates, amenities, and booking constraints.",
    ): (
        (
            "checkindatetime",
            {
                "type": "string",
                "description": "Check-in "
                "date/time "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2024-03-20T15:00:00'). "
                "Must "
                "be "
                "in "
                "the "
                "future.",
            },
        ),
        (
            "checkoutdatetime",
            {
                "type": "string",
                "description": "Check-out "
                "date/time "
                "in "
                "ISO "
                "8601 "
                "format. "
                "Must "
                "be "
                "after "
                "checkindatetime.",
            },
        ),
    ),
    (
        "hotels_search",
        "Searches for available hotels based on the provided filters and parameters.",
    ): (
        (
            "checkout_date",
            {
                "type": "string",
                "description": "Check-out date in the format YYYY-MM-DD.",
                "default": "2023-09-28",
            },
        ),
        (
            "checkin_date",
            {
                "type": "string",
                "description": "Check-in date in the format YYYY-MM-DD.",
                "default": "2023-09-27",
            },
        ),
    ),
    (
        "indicator",
        "Retrieves technical analysis indicators for financial instruments. This function provides quantitative metrics used in trading analysis, such as moving averages, momentum indicators, or volatility measures for specified assets and timeframes.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Analysis "
                "date "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD). "
                "Must "
                "be "
                "a "
                "valid "
                "trading "
                "day "
                "for "
                "the "
                "specified "
                "instrument.",
            },
        ),
    ),
    (
        "is_hotel_available",
        "Checks the availability of a hotel for a given date range.",
    ): (
        (
            "checkin",
            {
                "type": "string",
                "description": 'The check-in date in the format "YYYY-MM-DD".',
            },
        ),
        (
            "checkout",
            {
                "type": "string",
                "description": 'The check-out date in the format "YYYY-MM-DD".',
            },
        ),
    ),
    (
        "lia",
        "Provides travel recommendations and itinerary planning assistance based on destination preferences, travel dates, and group size. Returns curated travel options, local attractions, and practical tips for the specified location.",
    ): (
        (
            "travel_date",
            {
                "type": "string",
                "description": "Planned "
                "travel "
                "date "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Defaults "
                "to "
                "current "
                "date "
                "if "
                "unspecified",
                "default": "current_date",
                "format": "date",
            },
        ),
    ),
    (
        "list_pools_v1_dex_day_poolpairs_get",
        "Retrieves historical liquidity pool data from DeFiChain's decentralized exchange (DEX) for a specified day. Returns metrics such as total value locked (TVL), trading volume, and liquidity composition for all available pools.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "in "
                "ISO "
                "format "
                "(YYYY-MM-DD) "
                "to "
                "query "
                "historical "
                "data "
                "for. "
                "Defaults "
                "to "
                "the "
                "most "
                "recent "
                "available "
                "day "
                "if "
                "not "
                "specified.",
                "default": "latest",
            },
        ),
    ),
    ("make_appointment", "Schedule an appointment"): (
        ("date", {"type": "string", "description": "The date for the appointment"}),
    ),
    ("make_calendar_event", "Create a new event in the user's calendar"): (
        (
            "start_time",
            {
                "type": "string",
                "description": "The start time of the event in ISO 8601 format",
            },
        ),
        (
            "end_time",
            {
                "type": "string",
                "description": "The end time of the event in ISO 8601 format",
            },
        ),
    ),
    (
        "match_list_alt",
        "Retrieve a list of matches for a specific date with optional filtering for live matches. Returns match details including teams, scores, and current status (Played, Playing, Fixture, Cancelled).",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "to "
                "filter "
                "matches, "
                "formatted "
                "as "
                "YYYY-MM-DD. "
                "Required "
                "parameter "
                "for "
                "all "
                "requests.",
            },
        ),
    ),
    (
        "matches",
        "Retrieves latest sports match data from FDJ, including match details, scores, schedules, and results across various sports disciplines. Ideal for real-time sports updates and historical match information.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "in "
                "YYYY-MM-DD "
                "format "
                "to "
                "filter "
                "matches "
                "occurring "
                "on "
                "a "
                "specific "
                "day. "
                "Defaults "
                "to "
                "current "
                "date "
                "when "
                "unspecified.",
                "default": "current_date",
            },
        ),
    ),
    (
        "matches",
        "Retrieves the latest sports match data from Cbet, including match details, scores, schedules, and real-time updates. Use this function to get current sports event information across various leagues and competitions.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Match "
                "date "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Default "
                "is "
                "current "
                "date.",
                "default": "today",
            },
        ),
    ),
    (
        "matches",
        "Retrieves the latest sports matches data from Chillybets, including match details such as teams, scores, timestamps, and outcomes. Ideal for accessing up-to-date sports event information across multiple leagues and sports.",
    ): (
        (
            "match_date",
            {
                "type": "string",
                "description": "Filter "
                "matches "
                "by "
                "date "
                "in "
                "YYYY-MM-DD "
                "format, "
                "or "
                "use "
                "'today' "
                "for "
                "current "
                "day "
                "matches. "
                "Defaults "
                "to "
                "returning "
                "the "
                "most "
                "recent "
                "matches "
                "available.",
                "default": "latest",
            },
        ),
    ),
    (
        "matches",
        "Retrieves the latest sports matches data from Happybet, including match details like teams, scores, schedules, and betting options. Useful for real-time sports betting applications or match tracking systems.",
    ): (
        (
            "match_date",
            {
                "type": "string",
                "description": "Filter "
                "matches "
                "by "
                "date "
                "(YYYY-MM-DD "
                "format). "
                "Default "
                "returns "
                "matches "
                "from "
                "the "
                "next "
                "7 "
                "days",
                "default": "next_7_days",
            },
        ),
    ),
    (
        "mega_millions",
        "Retrieves official Mega Millions lottery results, including current jackpot amounts, winning numbers, and historical drawing data for Oregon Lottery. Use this function to check the latest results, verify winning numbers, or access historical lottery statistics.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Specific "
                "date "
                "to "
                "retrieve "
                "results "
                "for "
                "(format: "
                "YYYY-MM-DD). "
                "If "
                "omitted, "
                "returns "
                "the "
                "most "
                "recent "
                "drawing.",
            },
        ),
    ),
    (
        "monthly",
        "Retrieves monthly historical price, volume, and market data for stocks and ETFs. Returns adjusted closing prices, trading volume, and other financial metrics for analysis, trend identification, and portfolio backtesting.",
    ): (
        (
            "dateend",
            {
                "type": "string",
                "description": "End "
                "date "
                "for "
                "historical "
                "data "
                "retrieval "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Must "
                "be "
                "later "
                "than "
                "datestart.",
            },
        ),
        (
            "datestart",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "historical "
                "data "
                "retrieval "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Must "
                "be "
                "earlier "
                "than "
                "dateend.",
            },
        ),
    ),
    (
        "moon_horizontal_position_position_on_the_sky",
        "Calculates and returns the Moon's horizontal position in the sky as azimuth and altitude angles. Azimuth represents the compass direction (0° = North), and altitude represents the angle above the horizon. This function is useful for astronomy applications, celestial navigation, and observational planning.",
    ): (
        (
            "date_time",
            {
                "type": "string",
                "description": "Specific "
                "date "
                "and "
                "time "
                "for "
                "the "
                "calculation "
                "in "
                "'YYYY-MM-DD "
                "HH-MM-SS' "
                "format. "
                "If "
                "omitted, "
                "uses "
                "the "
                "current "
                "time "
                "at "
                "the "
                "moment "
                "of "
                "the "
                "request.",
                "default": "current time",
            },
        ),
    ),
    (
        "mutual_funds",
        "Calculates returns for Systematic Investment Plan (SIP) investments in mutual funds. Returns key metrics including total investment, current value, and annualized returns based on specified parameters.",
    ): (
        (
            "enddt",
            {
                "type": "string",
                "description": "End date of SIP investment in YYYY-MM-DD format",
                "default": "2024-01-01",
            },
        ),
        (
            "startdt",
            {
                "type": "string",
                "description": "Start date of SIP investment in YYYY-MM-DD format",
                "default": "2023-01-01",
            },
        ),
    ),
    (
        "natural_milk",
        "Manages natural milk product information and ordering capabilities for an e-commerce platform. Enables retrieval of product details, inventory status, and facilitates milk product purchases with customizable delivery options.",
    ): (
        (
            "preferred_delivery_date",
            {
                "type": "string",
                "description": "Requested "
                "delivery "
                "date "
                "in "
                "ISO "
                "format "
                "(YYYY-MM-DD). "
                "If "
                "not "
                "specified, "
                "uses "
                "earliest "
                "available "
                "date",
            },
        ),
    ),
    (
        "new",
        "Creates a new advertising campaign with specified configuration parameters. Used for initializing campaign details including budget allocation, scheduling, and audience targeting.",
    ): (
        (
            "start_date",
            {
                "type": "string",
                "description": "Scheduled "
                "start "
                "date/time "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2024-03-20T09:00:00Z'). "
                "Must "
                "be "
                "in "
                "the "
                "future.",
            },
        ),
    ),
    (
        "newlyregistereddomains",
        "Fetch a list of newly registered domains from the WhoIs Lookup API, applying optional filters to the search.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "registration "
                "date "
                "of "
                "the "
                "domains "
                "to "
                "be "
                "searched "
                "in "
                "'YYYY-MM-DD' "
                "format. "
                "Default "
                "is "
                "'2023-06-12'.",
                "default": "2023-06-12",
            },
        ),
    ),
    (
        "newlyregistereddomains",
        "Retrieves information about newly registered domains with optional filtering and pagination capabilities. This function enables users to search domains by registration date, filter by inclusion/exclusion of specific keywords, and navigate through results using pagination.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Registration "
                "date "
                "of "
                "domains "
                "to "
                "search, "
                "formatted "
                "as "
                "YYYY-MM-DD. "
                "If "
                "not "
                "specified, "
                "defaults "
                "to "
                "the "
                "current "
                "date.",
                "default": "current date (YYYY-MM-DD)",
            },
        ),
    ),
    (
        "news_sentiment",
        "Analyzes news sentiment and volume for a specified stock ticker over a defined date range. Returns quantitative sentiment scores (positive/negative indicators) and news volume metrics to support financial decision-making.",
    ): (
        (
            "enddate",
            {
                "type": "string",
                "description": "End "
                "date "
                "for "
                "news "
                "analysis "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Must "
                "be "
                "a "
                "valid "
                "calendar "
                "date "
                "and "
                "not "
                "in "
                "the "
                "future.",
            },
        ),
        (
            "startdate",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "news "
                "analysis "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Must "
                "be "
                "a "
                "valid "
                "calendar "
                "date "
                "and "
                "on "
                "or "
                "before "
                "the "
                "end "
                "date.",
            },
        ),
    ),
    (
        "odds_soccer",
        "Retrieves the latest soccer betting odds from Admiralbet, including match outcomes, over/under, and other market types. Use this function to get real-time betting data for specific matches, leagues, or teams.",
    ): (
        (
            "date",
            {
                "type": "string",
                "format": "date",
                "description": "Filter "
                "matches "
                "occurring "
                "on "
                "a "
                "specific "
                "date "
                "(YYYY-MM-DD). "
                "Defaults "
                "to "
                "current "
                "date "
                "if "
                "not "
                "provided.",
                "default": "current_date",
            },
        ),
    ),
    (
        "orders",
        "Retrieve order history from an e-commerce store with optional filtering and sorting capabilities. Useful for order management, inventory tracking, and sales analysis.",
    ): (
        (
            "end_date",
            {
                "type": "string",
                "description": "Filter "
                "orders "
                "created "
                "on "
                "or "
                "before "
                "this "
                "ISO "
                "8601 "
                "date "
                "(e.g., "
                "'2024-12-31T23:59:59Z')",
            },
        ),
        (
            "start_date",
            {
                "type": "string",
                "description": "Filter "
                "orders "
                "created "
                "on "
                "or "
                "after "
                "this "
                "ISO "
                "8601 "
                "date "
                "(e.g., "
                "'2024-01-01T00:00:00Z')",
            },
        ),
    ),
    (
        "properties_search",
        "Search for rental properties with advanced filtering options. Supports location-based queries with filters for property type, amenities, pricing, availability, and guest requirements. Returns listings matching specified criteria.",
    ): (
        (
            "checkin",
            {"type": "string", "description": "Checkin date in YYYY-MM-DD format"},
        ),
        (
            "checkout",
            {"type": "string", "description": "Checkout date in YYYY-MM-DD format"},
        ),
    ),
    (
        "query",
        "Retrieves locations affected by natural disasters within a specified date range. Returns common disaster locations with associated metadata. Date range must be between 2023-05-24 and yesterday's date (inclusive).",
    ): (
        (
            "is_from",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "disaster "
                "query "
                "in "
                "ISO "
                "8601 "
                "format "
                "(YYYY-MM-DD). "
                "Must "
                "be "
                "on "
                "or "
                "after "
                "2023-05-24 "
                "and "
                "before "
                "the "
                "'to' "
                "date.",
            },
        ),
    ),
    (
        "query_races",
        "Retrieves horse racing data with customizable filters and sorting. Supports searching by race name, course, date ranges, class levels, and distance parameters. Returns paginated results sorted by date.",
    ): (
        (
            "date_from",
            {
                "type": "string",
                "description": "Lower date boundary (inclusive) in YYYY-MM-DD format",
            },
        ),
        (
            "date_to",
            {
                "type": "string",
                "description": "Upper date boundary (inclusive) in YYYY-MM-DD format",
            },
        ),
    ),
    (
        "racecards",
        "Retrieves a comprehensive list of horse racing events for a specified date. Use this function to obtain race details including participants, schedules, and event-specific information for betting or tracking purposes.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "to "
                "filter "
                "races "
                "by, "
                "in "
                "YYYY-MM-DD "
                "format. "
                "If "
                "no "
                "date "
                "is "
                "specified, "
                "defaults "
                "to "
                "the "
                "current "
                "date.",
                "default": "current_date",
            },
        ),
    ),
    (
        "results",
        "Retrieves historical soccer match results for a specified date from a comprehensive sports database. Use this function to access detailed match records, scores, and team performance data for sports analysis or historical research.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "for "
                "which "
                "to "
                "retrieve "
                "soccer "
                "match "
                "results, "
                "formatted "
                "as "
                "YYYY-MM-DD. "
                "Required "
                "parameter.",
            },
        ),
    ),
    (
        "results",
        "Retrieves horse racing results for US-based events by date. Returns race outcomes, winning horses, jockeys, and event statistics. Useful for historical analysis or post-event review.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "to "
                "retrieve "
                "results "
                "for, "
                "in "
                "YYYY-MM-DD "
                "format. "
                "When "
                "not "
                "specified, "
                "defaults "
                "to "
                "current "
                "date "
                "to "
                "fetch "
                "latest "
                "results.",
                "default": "2023-10-15",
            },
        ),
    ),
    (
        "risk_free_rate",
        "Fetches the risk-free rate for a specific date, duration, and geography using the Toolbench RapidAPI.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Specific "
                "date "
                "for "
                "which "
                "the "
                "risk-free "
                "rate "
                "is "
                "to "
                "be "
                "fetched "
                "in "
                "'YYYY-MM-DD' "
                "format. "
                "Default "
                "is "
                "'2023-05-10'.",
                "default": "2023-05-10",
            },
        ),
    ),
    (
        "sample_predictions",
        "Retrieves a representative sample of sports match predictions from the previous day across multiple betting markets. Returns structured data including match details, prediction models, and market-specific insights for analysis and validation purposes.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "to "
                "retrieve "
                "predictions "
                "for "
                "(ISO "
                "format: "
                "YYYY-MM-DD). "
                "Defaults "
                "to "
                "previous "
                "day "
                "when "
                "unspecified.",
                "format": "date",
            },
        ),
    ),
    (
        "schedule",
        "Retrieves baseball game schedules with details including dates, times, teams, locations, and game statuses. Returns structured data for planning, tracking, or analysis of upcoming or historical games.",
    ): (
        (
            "start_date",
            {
                "type": "string",
                "format": "date",
                "description": "Earliest "
                "game "
                "date "
                "to "
                "include "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Defaults "
                "to "
                "current "
                "date "
                "if "
                "unspecified.",
                "default": "today",
            },
        ),
        (
            "end_date",
            {
                "type": "string",
                "format": "date",
                "description": "Latest "
                "game "
                "date "
                "to "
                "include "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Defaults "
                "to "
                "7 "
                "days "
                "after "
                "start_date "
                "if "
                "unspecified.",
                "default": "start_date + 7 days",
            },
        ),
    ),
    ("schedule_appointment", "Schedule an appointment with a specific date and time"): (
        ("date", {"type": "string", "description": "The appointment date"}),
    ),
    (
        "schedule_by_date",
        "Retrieves event schedules for a specific sport on a specified date. This function is useful for obtaining organized sports event data for applications like sports calendars, live score updates, or event planning tools.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "for "
                "which "
                "the "
                "schedule "
                "is "
                "required, "
                "formatted "
                "as "
                "'YYYY-MM-DD'. "
                "This "
                "must "
                "be "
                "a "
                "valid "
                "calendar "
                "date "
                "in "
                "the "
                "future "
                "or "
                "present.",
            },
        ),
    ),
    (
        "schedule_by_date",
        "Retrieves sports event schedules for a specified date and sport. Returns a list of upcoming events matching the specified sport and date.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "to "
                "query "
                "in "
                "YYYY-MM-DD "
                "format "
                "(e.g., "
                "'2024-03-20')",
            },
        ),
    ),
    (
        "schedule_date",
        "Fetches the baseball game schedule for a given date using the specified RapidAPI key.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "in "
                "the "
                "format "
                "'YYYY-MM-DD' "
                "for "
                "which "
                "to "
                "fetch "
                "the "
                "baseball "
                "schedule. "
                "Defaults "
                "to "
                "'2021-07-30'.",
                "default": "2021-07-30",
            },
        ),
    ),
    (
        "schedule_date",
        "Retrieves baseball schedule information for a specified date. Returns game details including teams, times, and locations. If no date is provided, defaults to the current date.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "to "
                "check "
                "baseball "
                "schedules "
                "for, "
                "formatted "
                "as "
                "YYYY-MM-DD. "
                "If "
                "not "
                "provided, "
                "defaults "
                "to "
                "the "
                "current "
                "date.",
                "default": "",
            },
        ),
    ),
    ("schedule_maintenance", "Schedule maintenance for a specific equipment"): (
        (
            "maintenance_date",
            {
                "type": "string",
                "format": "date",
                "description": "The date for the maintenance to be scheduled",
            },
        ),
    ),
    ("schedule_meeting", "Schedule a meeting with participants"): (
        ("date", {"type": "string", "description": "The date of the meeting"}),
    ),
    ("schedule_social_media_post", "Schedule a post on social media"): (
        ("date", {"type": "string", "description": "The date to schedule the post"}),
    ),
    ("schedule_task", "Schedule a task to be executed at a specific time"): (
        (
            "execution_time",
            {
                "type": "string",
                "description": "The time at which the task should be executed",
            },
        ),
    ),
    (
        "schedules",
        "Retrieves sports schedules for a specified sport ID with optional filtering by start date and result limits. Schedules are returned in chronological order (ascending by event date) and support pagination via date cursors. Use this function to fetch upcoming or historical events for a specific sport.",
    ): (
        (
            "is_from",
            {
                "type": "string",
                "description": "Starting "
                "date "
                "for "
                "the "
                "schedule "
                "query "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Defaults "
                "to "
                "today's "
                "date "
                "if "
                "not "
                "specified.",
                "default": "2023-10-10",
            },
        ),
    ),
    (
        "search_arrivals_by_flight",
        "Retrieves real-time arrival information for a specific flight using its identifier and scheduled arrival details. Provides current status, terminal, gate, and estimated arrival time data for operational transparency.",
    ): (
        (
            "flightnumber_scheduledarrivaldate_scheduledarrivaldate",
            {
                "type": "string",
                "description": "Composite "
                "key "
                "combining "
                "flight "
                "number "
                "and "
                "scheduled "
                "arrival "
                "date "
                "in "
                "the "
                "format "
                "FL123_YYYY-MM-DD. "
                "Used "
                "to "
                "uniquely "
                "identify "
                "the "
                "flight's "
                "scheduled "
                "arrival "
                "record.",
            },
        ),
    ),
    (
        "search_arrivals_by_route",
        "Retrieves flight arrival information for a specified route and scheduled arrival date. Useful for checking arrival times and flight details between specific locations.",
    ): (
        (
            "departurelocation_arrivallocation_arrivallocation_scheduledarrivaldate_scheduledarrivaldate",
            {
                "type": "string",
                "description": "Composite "
                "identifier "
                "combining "
                "departure "
                "location, "
                "arrival "
                "location, "
                "and "
                "scheduled "
                "arrival "
                "date "
                "in "
                "a "
                "format "
                "used "
                "internally "
                "for "
                "route "
                "matching",
            },
        ),
    ),
    (
        "search_departures_by_route",
        "Retrieves flight departure schedules for a specific route between two locations on a specified date. Returns flight numbers, departure times, and operational status including potential delays or cancellations.",
    ): (
        (
            "departurelocation_arrivallocation_arrivallocation_scheduleddeparturedate_scheduleddeparturedate",
            {
                "type": "string",
                "description": "Composite "
                "identifier "
                "combining "
                "departure "
                "location, "
                "arrival "
                "location, "
                "and "
                "scheduled "
                "departure "
                "date "
                "in "
                "a "
                "standardized "
                "format "
                "(e.g., "
                "'LHR_JFK_2023-12-25'). "
                "This "
                "field "
                "must "
                "match "
                "the "
                "format "
                "exactly "
                "to "
                "ensure "
                "accurate "
                "route "
                "identification.",
            },
        ),
    ),
    ("search_flights", "Search for flights based on given criteria"): (
        ("departure_date", {"type": "string", "description": "The departure date"}),
        (
            "return_date",
            {"type": "string", "description": "The return date for round-trip flights"},
        ),
    ),
    (
        "search_flights",
        "Searches for flights with options to filter by price, duration, stops, and dates. Supports one-way and round-trip bookings with customizable sorting and passenger counts. Ideal for finding optimal flight options based on user preferences.",
    ): (
        (
            "date_departure",
            {"type": "string", "description": "Departure date in YYYY-MM-DD format"},
        ),
        (
            "date_departure_return",
            {
                "type": "string",
                "description": "Return "
                "departure "
                "date "
                "in "
                "YYYY-MM-DD "
                "format "
                "(required "
                "for "
                "round-trip "
                "searches)",
            },
        ),
    ),
    ("search_hotel", "Search for a hotel by location and check-in/check-out dates"): (
        (
            "check_out_date",
            {
                "type": "string",
                "description": "The check-out date in format 'YYYY-MM-DD'",
            },
        ),
        (
            "check_in_date",
            {
                "type": "string",
                "description": "The check-in date in format 'YYYY-MM-DD'",
            },
        ),
    ),
    ("search_movie_showtimes", "Search for movie showtimes in a specific location"): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "date "
                "for "
                "which "
                "showtimes "
                "are "
                "to "
                "be "
                "searched "
                "in "
                "YYYY-MM-DD "
                "format",
            },
        ),
    ),
    (
        "search_today_s_arrivals_by_time",
        "Searches for flight arrivals at a specified location occurring on the current day within a defined time window. Returns detailed information about matching flights, including airline, flight number, status, and estimated arrival time.",
    ): (
        (
            "starttime",
            {
                "type": "string",
                "description": "Start "
                "of "
                "the "
                "search "
                "time "
                "window "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2023-10-05T08:00:00Z'). "
                "Search "
                "will "
                "include "
                "flights "
                "arriving "
                "on "
                "or "
                "after "
                "this "
                "time.",
            },
        ),
        (
            "arrivallocation_starttime_starttime_endtime_endtime",
            {
                "type": "string",
                "description": "Combined "
                "search "
                "criteria "
                "containing "
                "arrival "
                "location, "
                "start "
                "time, "
                "and "
                "end "
                "time "
                "in "
                "a "
                "single "
                "string. "
                "Format: "
                "'LOCATION|START_TIME|END_TIME' "
                "where "
                "LOCATION "
                "is "
                "an "
                "IATA "
                "airport "
                "code "
                "(e.g., "
                "'LHR'), "
                "and "
                "START_TIME/END_TIME "
                "are "
                "in "
                "ISO "
                "8601 "
                "format "
                "(e.g., "
                "'2023-10-05T08:00:00Z').",
            },
        ),
    ),
    (
        "short_volume_specific_date",
        "Retrieves short volume data for a specified stock ticker on a specific calendar date. Provides insights into market short selling activity for the given security.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "The "
                "calendar "
                "date "
                "to "
                "retrieve "
                "short "
                "volume "
                "data "
                "for, "
                "formatted "
                "as "
                "YYYY-MM-DD",
            },
        ),
    ),
    (
        "sports",
        "Retrieves live sports scores, schedules, and statistics. Supports filtering by sport, league, team, player, and event date. Returns comprehensive sports-related information based on specified criteria.",
    ): (
        (
            "date",
            {
                "type": "string",
                "description": "Date "
                "to "
                "retrieve "
                "events "
                "for, "
                "formatted "
                "as "
                "YYYY-MM-DD. "
                "If "
                "not "
                "specified, "
                "defaults "
                "to "
                "current "
                "date.",
                "format": "date",
            },
        ),
    ),
    (
        "submit_business_inquiry",
        "Submits a formal business inquiry to the Brisbane Agency for services, consultations, or information requests. Use this function to schedule meetings, request documentation, or initiate business processes with the agency.",
    ): (
        (
            "preferred_date",
            {
                "type": "string",
                "description": "Preferred "
                "date "
                "and "
                "time "
                "for "
                "service "
                "delivery "
                "or "
                "response "
                "(ISO "
                "8601 "
                "format), "
                "e.g., "
                "'2024-03-20T14:00:00+10:00'. "
                "Defaults "
                "to "
                "immediate "
                "processing "
                "if "
                "omitted.",
                "default": None,
            },
        ),
    ),
    (
        "test",
        "Retrieves sports match data for 90-minute games. Provides information on upcoming, live, or completed matches with options to filter by team, league, or date.",
    ): (
        (
            "match_date",
            {
                "type": "string",
                "format": "date",
                "description": "Date "
                "to "
                "filter "
                "matches, "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Returns "
                "matches "
                "occurring "
                "on "
                "the "
                "specified "
                "date.",
            },
        ),
    ),
    (
        "tickerdata",
        "Fetches historical ticker data for a given period and date range from the RapidAPI service.",
    ): (
        (
            "startdate",
            {
                "type": "string",
                "description": "The start date for the data in YYYY-MM-DD format.",
                "default": "2010-04-12T14:30",
            },
        ),
        (
            "enddate",
            {
                "type": "string",
                "description": "The end date for the data in YYYY-MM-DD format.",
                "default": "2010-04-20T12:30",
            },
        ),
    ),
    (
        "time_series_endpoint",
        "Retrieves historical exchange rate data between two specified dates. Returns currency conversion rates from a source currency to one or more target currencies. Maximum time range allowed is 365 days.",
    ): (
        (
            "start_date",
            {
                "type": "string",
                "description": "Start "
                "date "
                "for "
                "historical "
                "exchange "
                "rate "
                "data "
                "in "
                "YYYY-MM-DD "
                "format",
            },
        ),
    ),
    (
        "time_series_yield_curve",
        "Retrieves historical yield curve data for a specified date range. Returns risk-free rate information across different maturities for Eurozone instruments.",
    ): (
        (
            "enddate",
            {
                "type": "string",
                "description": "End "
                "date "
                "of "
                "the "
                "requested "
                "date "
                "range "
                "(inclusive). "
                "Must "
                "be "
                "in "
                "ISO "
                "8601 "
                "format: "
                "YYYY-MM-DD",
            },
        ),
    ),
    (
        "travelopro",
        "Provides comprehensive travel search capabilities for flights, hotels, and car rentals. Enables users to search for travel options with customizable parameters including destination, dates, and pricing filters.",
    ): (
        (
            "return_date",
            {
                "type": "string",
                "description": "Return "
                "date "
                "for "
                "round-trip "
                "bookings "
                "in "
                "YYYY-MM-DD "
                "format",
                "format": "date",
            },
        ),
        (
            "departure_date",
            {
                "type": "string",
                "description": "Travel "
                "departure "
                "date "
                "in "
                "YYYY-MM-DD "
                "format. "
                "Required "
                "for "
                "flight "
                "and "
                "car "
                "rental "
                "searches",
                "format": "date",
            },
        ),
    ),
    (
        "upcoming_matches_api",
        "Retrieves comprehensive information about upcoming cricket matches globally, including teams, dates, venues, and match formats. Ideal for sports analytics, fantasy cricket applications, or real-time sports tracking services.",
    ): (
        (
            "date_range",
            {
                "type": "string",
                "description": "Filter "
                "matches "
                "within "
                "a "
                "specific "
                "date "
                "range "
                "(e.g., "
                "'2023-11-01 "
                "to "
                "2023-11-15'). "
                "Default: "
                "next "
                "7 "
                "days "
                "from "
                "current "
                "date",
                "default": "next_7_days",
            },
        ),
    ),
    ("us_de", "Fetches current or historical gas price data for Delaware."): (
        (
            "date",
            {
                "type": "string",
                "description": "A "
                "specific "
                "date "
                "for "
                "querying "
                "historical "
                "gas "
                "price "
                "data. "
                "If "
                "not "
                "provided, "
                "the "
                "current "
                "gas "
                "price "
                "data "
                "is "
                "returned.",
            },
        ),
    ),
    (
        "v1_search",
        "Performs a customizable search through a news database with various filtering and sorting options.",
    ): (
        (
            "is_from",
            {"type": "string", "description": "Date from which to start the search."},
        ),
        (
            "to",
            {
                "type": "string",
                "description": "Date until which to search for articles.",
            },
        ),
    ),
    (
        "year_fractions",
        "Calculates the year fraction between two dates based on the specified day count convention using the RapidAPI service.",
    ): (
        (
            "start_date",
            {
                "type": "string",
                "description": "The start date of the period in YYYY-MM-DD format.",
                "default": "2021-03-31",
            },
        ),
        (
            "end_date",
            {
                "type": "string",
                "description": "The end date of the period in YYYY-MM-DD format.",
                "default": "2021-04-30",
            },
        ),
    ),
}


def _parameter_schema(definition: dict[str, Any], path: str) -> Any:
    """Follow only the audited object-property/array-item schema path."""
    schema: Any = definition.get("parameters")
    for part in path.split("."):
        if not isinstance(schema, dict):
            return None
        if part == "*":
            schema = schema.get("items")
        else:
            properties = schema.get("properties")
            schema = properties.get(part) if isinstance(properties, dict) else None
    return schema


def calendar_slots(
    name: str, definition: dict[str, Any], *, split: str | None
) -> tuple[str, ...]:
    description = definition.get("description")
    if not isinstance(split, str) or not isinstance(description, str):
        return ()
    if split == "high":
        return tuple(
            path
            for path, expected in _HIGH_CONTRACTS.get((name, description), ())
            if json_bytes(_parameter_schema(definition, path)) == json_bytes(expected)
        )
    return _CONTRACTS.get((split, name, description), ())
