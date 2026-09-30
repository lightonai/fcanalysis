"""Bounded calendar-reference checks for reviewed Nemotron consuming slots.

The pinned releases contain real booking/scheduling calls that translate a
user's tonight/tomorrow/next-weekday request to a full calendar date despite
having no visible clock: v1 hotel availability, v2 facility tours, conference
rooms, API-Bank conflict checks and the TOUCAN-derived 12306 search are examples.
The mappings below identify reviewed consuming calendar arguments, not arbitrary
date-shaped prose, historical queries, birthdays, or generated sample records.
Relative expressions alone never cause deletion: a mapped call must also emit
an absolute calendar date that lacks eligible prior evidence.

Prior explicit full dates can ground the corresponding requested date, while
an explicitly labeled current date provides a reference for relative
conversion. Historical dates do not become clocks. This gate checks the
demonstrated omission before any tool results; later unclassified backend
results can establish temporal facts, so they cause bounded abstention.
Assistant prose/reasoning, definitions/examples, raw metadata and future
results are excluded. The comparison uses shared calendar parsing and never
rewrites the source, guesses a missing year, or reads the machine clock. This
is a necessary-evidence gate, not general arithmetic or prose certification.
User date ranges, numeric slash/dot dates and partial month/day or month/year
expressions cause abstention: this parser cannot prove their omission.
Unknown temporal APIs and implicit natural-language clock assertions remain
outside this bounded policy. The full gate runs again after system overrides.
"""

from typing import Any

from .context import validate_calendar_evidence
from .pipeline import RowState

# Exact reviewed source function/argument pairs. Nested array positions use *.
# Free-form summaries, IDs, generated document contents and historical-data
# query endpoints are intentionally absent even when they contain date text.
_CALENDAR_INPUTS = {
    "get_room_availability": "check_in_date check_out_date",
    "create_reservation": "check_in_date check_out_date date_time start_date end_date",
    "modify_reservation": "new_check_in_date new_check_out_date new_date_time new_dates.start_date new_dates.end_date",
    "get_available_rooms": "date",
    "checktimeconflict": "time",
    "schedule_facility_tour": "tour_datetime",
    "find_hotel_deals": "check_in_date check_out_date",
    "search_for_hotels": "check_in_date check_out_date",
    "find_hotels": "check_in_date check_out_date",
    "find_hotel": "check_in_date check_out_date",
    "search_hotel": "check_in_date check_out_date",
    "search_hotels": "check_in_date check_out_date check_in check_out date_checkin date_checkout",
    "find_nearby_hotels": "check_in_date check_out_date",
    "find_hotel_availability": "check_in_date check_out_date",
    "find_hotel_rooms": "check_in_date check_out_date",
    "book_hotel_room": "check_in_date check_out_date",
    "book_hotel": "check_in_date check_out_date",
    "reserve_hotel_room": "checkin_date checkout_date",
    "hotels_search": "checkin_date checkout_date",
    "gethotels": "check_in_date check_out_date",
    "is_hotel_available": "checkin checkout",
    "properties_search": "checkin checkout",
    "check_hotspot_availability": "check_in_date check_out_date",
    "book_table": "date",
    "reserve_table": "date",
    "make_reservation": "date",
    "book_appointment": "date datetime",
    "make_appointment": "date",
    "schedule_appointment": "date preferred_date preferred_time time_slot slot",
    "reschedule_appointment": "new_date",
    "modify_appointment": "new_date new_time",
    "schedule_meeting": "date",
    "schedule_social_media_post": "date",
    "create_event_reminder": "event_date",
    "set_reminder": "date_time",
    "create_todo_item": "due_date",
    "create_todo_task": "due_date",
    "create_todo": "due_date",
    "add_task": "due_date",
    "create_task": "due_date",
    "create_new_task": "due_date",
    "add_todo": "due_date",
    "add_event": "date",
    "get_available_flights": "date",
    "search_flights": "departure_date return_date date date_departure date_departure_return",
    "book_flight": "date",
    "flight_api": "departure_date return_date",
    "airline_travel": "departure_date return_date",
    "ensure_flight": "departure_date return_date",
    "lista": "departure_date return_date",
    "check_bus_schedule": "date",
    "check_train_schedule": "date",
    "check_flight_status": "date",
    "get_flight_status": "departure_date",
    "get_local_events": "date_range.start_date date_range.end_date start_date end_date",
    "get_calendar_events": "start_date end_date",
    "schedule_maintenance": "maintenance_date preferred_date",
    "schedule_service": "appointment_date preferred_date preferred_dates.* requested_date",
    "schedule_service_appointment": "preferred_date",
    "schedule_field_service": "preferred_date",
    "schedule_consultation": "preferred_datetime",
    "schedule_technician_visit": "preferred_datetime",
    "schedule_inspection": "requested_date",
    "schedule_yard_appointment": "preferred_date",
    "schedule_bulky_item": "service_date",
    "schedule_collection": "service_date",
    "schedule_hazardous_dropoff": "preferred_date",
    "schedule_delivery": "delivery_date delivery_time",
    "reschedule_delivery": "new_delivery_date",
    "schedule_pickup": "pickup_date",
    "schedule_medical_appointment": "date",
    "schedule_tour": "date",
    "book_design_consultation": "preferred_date",
    "book_showroom_appointment": "date",
    "book_lesson": "time_slot",
    "book_service": "date_time",
    "create_booking": "service_date move_date",
    "check_tech_availability": "appointment_time date_range",
    "get_service_availability": "date",
    "check_technician_availability": "date",
    "check_appointment_availability": "date date_range",
    "check_reservation_availability": "date",
    "check_facility_availability": "date_range",
    "check_service_availability": "date_range",
    "check_showroom_capacity": "date",
    "check_ticket_availability": "date",
    "check_tasting_availability": "date",
    "check_venue_capacity": "date",
    "check_site_availability": "start_date end_date",
    "check_designer_availability": "start_date end_date",
    "check_equipment_availability": "start_date end_date",
    "check_rental_availability": "start_date end_date",
    "get_available_time_slots": "date",
    "get_available_installation_dates": "start_date",
    "search_tee_time_availability": "date",
    "searchAvailableSlips": "startDate endDate",
    "createReservation": "startDate endDate",
    "check_slip_availability": "start_date end_date",
    "get_tutor_availability": "date_range",
    "get_class_availability": "date_range",
    "get_class_schedule": "date_range",
    "get_therapy_schedule": "start_date",
    "purchase_tickets": "date",
    "modify_ticket": "new_date",
    "modify_booking_details": "new_date",
    "modify_booking": "new_date_range.start_date new_date_range.end_date",
    "inquire_private_event": "event_date",
    "manage_facility_reservation": "event_date",
    "reserve_facility": "date",
    "forecast_weather_api": "dt",
    "get_weather_by_datetime_range": "start_date end_date",
    "get_weather_forecast": "time",
    "clock_alarm_set": "date",
    "clock_alarm_cancel": "date",
}
_CALENDAR_PATHS = {
    name: frozenset(paths.split()) for name, paths in _CALENDAR_INPUTS.items()
}


def _calendar_slots(name: str, definition: dict[str, Any]) -> frozenset[str]:
    if name == "search" and definition.get("description") == "查询12306火车票":
        schema = definition.get("parameters")
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if isinstance(properties, dict) and (
            properties.get("date")
            == {
                "type": "string",
                "format": "date",
                "description": "出发日期 格式：YYYY-MM-DD",
            }
            and properties.get("fromCity")
            == {"type": "string", "description": "出发城市"}
            and properties.get("toCity")
            == {"type": "string", "description": "到达城市"}
        ):
            return frozenset({"date"})
    return _CALENDAR_PATHS.get(name, frozenset())


def validate_temporal_inputs(state: RowState) -> None:
    validate_calendar_evidence(state, slots=_calendar_slots)
