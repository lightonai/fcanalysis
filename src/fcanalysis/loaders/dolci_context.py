"""Bounded credential, existing-ID and chart-date checks for pinned Dolci SimFC.

These exact tool/parameter contracts were inspected in all 200,000 pinned
SimFC rows; ``dolci.py`` documents the source and interpretation limits.
A source schema describes each as
an existing authentication credential, never an invented password, password
analysis input, cryptocurrency token, or pagination cursor. Matching depends
on exact function names and exact parameter descriptions, not keyword search.

This stage follows structural validation and complete call/result linkage.
Only earlier user/system literal tokens or string values in earlier linked
JSON results establish literal evidence. Assistant assertions, definition
examples/defaults, future results and substrings do not establish credentials.
Unresolved values are quarantined, including a requested deliberately wrong
password without a specified test value. This conservative evidence test is
not a certificate of authentication, authorization, or general semantic truth.
The exact Bolivia Songs chart contract additionally requires a supplied date
from earlier user/system text or a linked result, permitting only ISO dates and
unambiguous English month/day/year normalization. It never uses a host clock or
interprets relative-time words as clock evidence. Other dates, IDs, task-specific
state, and unreviewed contracts remain unresolved.

Exact Instagram follower/media contracts and temporary-upload URL contracts
also require supplied existing user/video/account IDs from earlier literal
evidence. A TikTok handle does not create an Instagram ID, and an email does
not create a video/account ID. Literal evidence alone cannot establish that
the value belongs to the intended entity, is authorized, or denotes today.
"""

from datetime import date
import re

from .context import collect_string_values, contains_literal_token
from .normalization import Reject, parse_json
from .pipeline import RowState

SIMFC_SOURCE = "allenai/olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated"

_CHART_NAMES = frozenset(
    (
        "bolivia_songs",
        "music.bolivia_songs",
        "bolivia.bolivia_songs",
        "module.bolivia_songs",
        "songs.bolivia_songs",
    )
)
_CHART_DESCRIPTION = (
    "Fetches the Bolivia Songs chart information for a given range and date."
)
_CHART_DATE_PARAMETER = {
    "type": "string",
    "description": "The date for which the chart information is required, in the format YYYY-MM-DD.",
    "default": "2022-05-07",
}
_IDENTIFIER_CONTRACTS = (
    (
        "Retrieves the list of followers for a given Instagram user.",
        frozenset(
            ("followers", "instagram.followers", "module.followers", "user.followers")
        ),
        {
            "user_id": {
                "type": "string",
                "default": "25025320",
                "description": "The ID of the Instagram user whose followers are to be retrieved.",
            }
        },
    ),
    (
        "Retrieves media posts from a specified Instagram user using the Instagram RapidAPI.",
        frozenset(
            (
                "instagram.medias",
                "medias",
                "medias.retriever",
                "medias.medias",
                "medias.api",
                "medias.get_medias",
                "Instagram.medias",
                "media.medias",
            )
        ),
        {
            "user_id": {
                "type": "string",
                "default": "25025320",
                "description": "The ID of the Instagram user whose media posts are to be retrieved.",
            }
        },
    ),
    (
        "Generates temporary upload URLs for a given video on a specific account using the provided source name.",
        frozenset(
            (
                "upload.temp_upload_urls",
                "temp_upload_urls",
                "module.temp_upload_urls",
                "video.temp_upload_urls",
            )
        ),
        {
            "video_id": {
                "type": "string",
                "default": "",
                "description": "The unique identifier for the video.",
            },
            "account_id": {
                "type": "string",
                "default": "",
                "description": "The unique identifier for the account.",
            },
        },
    ),
)
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_MONTH_NUMBER: dict[str, int] = {
    name.lower(): number
    for number, full in enumerate(_MONTHS, 1)
    for name in (full, full[:3])
}
_MONTH = (
    "(?:"
    + "|".join(sorted(_MONTH_NUMBER, key=lambda value: len(value), reverse=True))
    + ")"
)
_DAY = r"(?P<day>\d{1,2})(?P<ordinal>st|nd|rd|th)?"
_NATURAL_DATES = (
    re.compile(
        r"(?<!\w)(?P<month>" + _MONTH + r")\s+" + _DAY + r",?\s+(?P<year>\d{4})(?!\w)",
        re.I,
    ),
    re.compile(
        r"(?<!\w)" + _DAY + r"\s+(?P<month>" + _MONTH + r"),?\s+(?P<year>\d{4})(?!\w)",
        re.I,
    ),
)


def _calendar_dates(text: str) -> set[str]:
    """Only explicit English month/day/year forms; never relative dates."""
    dates = set()
    for pattern in _NATURAL_DATES:
        for match in pattern.finditer(text):
            day = int(match["day"])
            ordinal = (
                "th"
                if 10 <= day % 100 <= 20
                else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            )
            if match["ordinal"] and match["ordinal"].lower() != ordinal:
                continue
            try:
                dates.add(
                    date(
                        int(match["year"]), _MONTH_NUMBER[match["month"].lower()], day
                    ).isoformat()
                )
            except ValueError:
                continue
    return dates


# Exact audited policies, grouped only to avoid duplicating identical contract
# text for source-qualified names. No name suffix or description keyword match.
_CREDENTIAL_CONTRACTS = (
    (
        "password",
        "The password for logging in.",
        frozenset(
            (
                "PetStoreAPI.loginuser",
                "auth.loginuser",
                "authentication.loginuser",
                "loginuser",
                "module.loginuser",
                "moduleName.loginuser",
                "module_name.loginuser",
                "petstore.loginuser",
                "user.login",
                "users.loginuser",
            )
        ),
    ),
    (
        "password",
        "The password for login in plain text.",
        frozenset(
            (
                "auth.loginuser",
                "authentication.loginuser",
                "loginuser",
                "module.loginuser",
                "module_name.loginuser",
                "user.loginuser",
                "user_auth.loginuser",
                "user_management.loginuser",
                "user_module.loginuser",
            )
        ),
    ),
    (
        "secret",
        "The secret key for authenticating the API request.",
        frozenset(
            (
                "countries.getthelistofcountriesregistered",
                "ecombr.getthelistofcountriesregistered",
                "ecommerce.getthelistofcountriesregistered",
                "getthelistofcountriesregistered",
                "module.getthelistofcountriesregistered",
            )
        ),
    ),
    (
        "token",
        "The token for validating the API request.",
        frozenset(
            (
                "countries.getthelistofcountriesregistered",
                "ecombr.getthelistofcountriesregistered",
                "ecommerce.getthelistofcountriesregistered",
                "getthelistofcountriesregistered",
                "module.getthelistofcountriesregistered",
            )
        ),
    ),
    (
        "password",
        "The password for login in clear text.",
        frozenset(
            (
                "AuthService.loginuser",
                "Toolbench.loginuser",
                "ToolbenchAPI.loginuser",
                "auth.loginuser",
                "auth_module.loginuser",
                "auth_service.loginuser",
                "authentication.loginuser",
                "login.user",
                "loginuser",
                "module.loginuser",
                "module_name.loginuser",
                "toolbench.loginuser",
                "user.loginuser",
                "userAuth.loginuser",
                "user_auth.loginuser",
                "user_management.loginuser",
                "users.loginuser",
            )
        ),
    ),
    (
        "password",
        "The password for logging in, in clear text.",
        frozenset(
            (
                "api.loginuser",
                "auth.loginuser",
                "auth_api.loginuser",
                "auth_module.loginuser",
                "authentication.loginuser",
                "loginuser",
                "loginuser.loginuser",
                "module.loginuser",
                "numerology.loginuser",
                "user.loginuser",
                "user_management.loginuser",
            )
        ),
    ),
    (
        "password",
        "The user's password for login in clear text.",
        frozenset(
            (
                "Toolbench.loginuser",
                "UserAuth.loginuser",
                "api.loginuser",
                "auth.loginuser",
                "auth_module.loginuser",
                "authentication.loginuser",
                "login.loginuser",
                "loginuser",
                "module.loginuser",
                "module_name.loginuser",
                "toolbench.loginuser",
                "user.loginuser",
                "user_auth.loginuser",
            )
        ),
    ),
    ("password", "The password for login in clear text", frozenset(("loginUser",))),
    (
        "api_key",
        "The API key associated with the user account.",
        frozenset(
            (
                "api.dashboard",
                "dashboard",
                "dashboard.dashboard",
                "dashboard.fetch",
                "dashboard_module.dashboard",
                "dashboardmodule.dashboard",
                "module.dashboard",
                "survey.api",
                "survey.dashboard",
                "surveyModule.dashboard",
                "survey_api.dashboard",
                "survey_module.dashboard",
                "surveys.dashboard",
            )
        ),
    ),
    (
        "password",
        "Password associated with the username.",
        frozenset(
            (
                "auth.user_login",
                "authentication.user_login",
                "module.user_login",
                "module_name.user_login",
                "user.authentication.user_login",
                "user.login",
                "user.user_login",
                "user_auth.user_login",
                "user_login",
                "user_management.user_login",
            )
        ),
    ),
    (
        "token",
        "Authentication token for the Ecombr API.",
        frozenset(
            (
                "Ecombr.listoforders",
                "api.listoforders",
                "ecombr.listoforders",
                "ecommerce.listoforders",
                "listoforders",
                "orders.listoforders",
            )
        ),
    ),
    (
        "secret",
        "Secret key for additional authentication.",
        frozenset(
            (
                "Ecombr.listoforders",
                "api.listoforders",
                "ecombr.listoforders",
                "ecommerce.listoforders",
                "listoforders",
                "orders.listoforders",
            )
        ),
    ),
    (
        "password",
        "Your SensSMS API key.",
        frozenset(
            (
                "SensSMS.message_send",
                "message.send",
                "message_send",
                "message_send.message_send",
                "message_send.send",
                "messaging.message_send",
                "sensms.message_send",
                "senssms.message_send",
                "senssms_api.message_send",
                "sms.message_send",
            )
        ),
    ),
    (
        "password",
        "The password for the social media account",
        frozenset(("loginWithSocialMedia",)),
    ),
    (
        "client_secret",
        "The client's secret key. Defaults to None.",
        frozenset(
            (
                "Auth.token",
                "OAuth.token",
                "auth_module.token",
                "module_name.token",
                "oauth.token",
                "token",
            )
        ),
    ),
    (
        "authentication_token",
        "The authentication token of the user",
        frozenset(("validateAccess",)),
    ),
    (
        "session_token",
        "The session token of the user",
        frozenset(("logoutUser", "verifyUser")),
    ),
    (
        "apikey",
        "The API key associated with the account. You can obtain the API key at https://app.rivet.solutions/ApiDocument/ApiDocs once your account is created.",
        frozenset(("get_sender_id", "module.get_sender_id")),
    ),
    (
        "secret",
        "Secret key for authentication with the API.",
        frozenset(
            (
                "listoforders",
                "listoforders.api",
                "marketplace.listoforders",
                "module.listoforders",
                "orders.listoforders",
            )
        ),
    ),
    (
        "token",
        "Token for authentication with the API.",
        frozenset(
            (
                "listoforders",
                "listoforders.api",
                "marketplace.listoforders",
                "module.listoforders",
                "orders.listoforders",
            )
        ),
    ),
    (
        "password",
        "The password of the user attempting to access the database",
        frozenset(("checkDatabaseAccess",)),
    ),
    (
        "apikey",
        "The Rivet SMS API key. Obtain it from https://app.rivet.solutions/ApiDocument/ApiDocs#.",
        frozenset(("sendsms",)),
    ),
    (
        "password",
        "The password of the user for authentication.",
        frozenset(("databasePrivilegeFetcher.fetchPrivileges",)),
    ),
    (
        "password",
        "The password to access the database",
        frozenset(("createDatabase", "deleteDatabase")),
    ),
    (
        "password",
        "The password to authenticate the database",
        frozenset(("checkDatabaseStatus",)),
    ),
    ("password", "The password for login", frozenset(("checkLogin",))),
    (
        "access_token",
        "The access token for authentication",
        frozenset(
            (
                "deleteFromCloud",
                "downloadFromCloud",
                "shareFileInCloud",
                "uploadToCloud",
            )
        ),
    ),
    (
        "access_token",
        "The access token for the OAuth application",
        frozenset(("fetchOauthAuthorizedApplications",)),
    ),
    (
        "password",
        "The password to authenticate with the server",
        frozenset(("runCommand",)),
    ),
    (
        "token",
        "Token sent in the email to confirm registration",
        frozenset(("ConfirmRegistration",)),
    ),
)


def validate_simfc_credentials(state: RowState) -> None:
    """Validate audited credential/date contracts; keep the existing stage API."""
    if (state.sample.raw or {}).get("dataset_source") != SIMFC_SOURCE:
        return
    selected: dict[str, set[str]] = {}
    charts = set()
    identifiers: dict[str, set[str]] = {}
    for tool in state.sample.tools:
        function = tool["function"]
        name = function["name"]
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            continue
        for description, names, parameters in _IDENTIFIER_CONTRACTS:
            if (
                name in names
                and function.get("description") == description
                and all(
                    properties.get(arg) == parameter
                    for arg, parameter in parameters.items()
                )
            ):
                identifiers[name] = set(parameters)
        if (
            name in _CHART_NAMES
            and function.get("description") == _CHART_DESCRIPTION
            and properties.get("date") == _CHART_DATE_PARAMETER
        ):
            charts.add(name)
        for argument, description, names in _CREDENTIAL_CONTRACTS:
            if name not in names:
                continue
            parameter = properties.get(argument)
            if (
                isinstance(parameter, dict)
                and parameter.get("description") == description
            ):
                selected.setdefault(name, set()).add(argument)
    if not selected and not charts and not identifiers:
        return

    # Bounded to one complete trajectory. Only rows with protected calls parse
    # results here, once each; earlier resolved call arguments are not evidence.
    trusted_text: list[str] = []
    result_values: set[str] = set()
    explicit_dates: set[str] = set()
    linked_results = {
        index for batch in state.batches for index in batch.result_indices
    }
    for index, message in enumerate(state.sample.messages):
        for call_index, call in enumerate(message.get("tool_calls", [])):
            arguments = state.parsed_arguments[(index, call_index)]
            for argument in identifiers.get(call["function"]["name"], ()):
                if argument not in arguments:
                    continue
                value = arguments[argument]
                if (
                    not isinstance(value, str)
                    or not value
                    or (
                        value not in result_values
                        and not contains_literal_token(value, trusted_text)
                    )
                ):
                    raise Reject("dolci_unresolved_identifier_context")
            if call["function"]["name"] in charts and "date" in arguments:
                value = arguments["date"]
                if (
                    not isinstance(value, str)
                    or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None
                ):
                    raise Reject("dolci_unresolved_date_context")
                try:
                    date.fromisoformat(value)
                except ValueError as exc:
                    raise Reject("dolci_unresolved_date_context") from exc
                if (
                    value not in result_values
                    and value not in explicit_dates
                    and not contains_literal_token(value, trusted_text)
                ):
                    raise Reject("dolci_unresolved_date_context")
            for argument in selected.get(call["function"]["name"], ()):
                if argument not in arguments:
                    continue  # Presence is governed by the complete source schema.
                value = arguments[argument]
                # An explicit schema-valid null/empty value asserts no opaque
                # credential. This does not certify the result's auth claims.
                if value is None or value == "":
                    continue
                if not isinstance(value, str):
                    raise Reject("dolci_unresolved_credential_context")
                if value not in result_values and not contains_literal_token(
                    value, trusted_text
                ):
                    raise Reject("dolci_unresolved_credential_context")
        content = message.get("content")
        if isinstance(content, str):
            if message["role"] in ("user", "system"):
                trusted_text.append(content)
                if charts:
                    explicit_dates.update(_calendar_dates(content))
            elif index in linked_results:
                try:
                    parsed = parse_json(content)
                except Reject:
                    continue  # Unparsed prose is not an audited state grammar.
                if charts:
                    # Normalize each newly parsed string leaf once, not the
                    # accumulated result set on every observation.
                    leaves: set[str] = set()
                    collect_string_values(parsed, leaves)
                    result_values.update(leaves)
                    for value in leaves:
                        explicit_dates.update(_calendar_dates(value))
                else:
                    collect_string_values(parsed, result_values)
