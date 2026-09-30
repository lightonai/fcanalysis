"""Finite ToolMind credential predicates from all eight pinned source files.

Audited Nanbeige/ToolMind at 8020ed1c03c367e4eb720ac3828ab4b0b95d8baf.
These exact field/description contracts declare existing authentication state.
Ambiguous password descriptions also require the exact consuming function name;
missing property descriptions require an exact function description. The
Linkedin Contacts/key rule is supported by that released tool envelope's
input_description, which identifies the key as an API key (GraphSyn row 4562).

New account passwords, strength/hash/encryption operations, ordinary code/key
fields and crypto token identifiers remain separate. Description-absent audit
candidates can include undeclared arguments; names alone do not add semantics.
Only three explicitly released public access constants bypass prior evidence.
GraphSyn rows 54597/78367 additionally expose apiTokenInstance, whose released
description explicitly identifies an existing authentication token. Row 68142
calls http.post with an unresolved URL template that its parameter description
requires replacing with an actual API key. That exact source template is
quarantined; arbitrary URLs are not treated as credentials or rewritten.
"""

from typing import TYPE_CHECKING, Any

from .normalization import Reject

if TYPE_CHECKING:
    from .pipeline import RowState

_EXISTING_DESCRIPTIONS: dict[str, frozenset[str]] = {
    "apiTokenInstance": frozenset(
        {
            "API token used for authentication",
            "The API token for the WhatsApp instance.",
            "The API token for the WhatsApp account.",
            "The API token for the user's account",
            "The API token used to authenticate the request.",
        }
    ),
    "access_token": frozenset(
        {
            "Access token for authenticating the request.",
            "Authentication token for the user",
            "The access token for authentication",
            "The access token for the Instagram API.",
            "The access token for the OAuth application",
            "The access token obtained from the authenticate",
            "The access token of the authenticated user.",
            "The access token received during authentication.",
            "The access token to use for accessing the user's profile information.",
        }
    ),
    "apiKey": frozenset(
        {
            "API key for authenticating the request.",
            "API key for authenticating with the air quality API.",
            "API key for authenticating with the open weather API.",
            "API key for authentication",
            "Stormglass API key",
            "The API key for authentication",
            "The API key provided by ScraperAPI.",
            "The API key used to authenticate the request.",
        }
    ),
    "api_key": frozenset(
        {
            "2Factor account API Key",
            "API Key Obtained From 2Factor.in",
            "API Key obtained from 2Factor.in",
            "API Obtained From 2Factor.in",
            "API key for authentication",
            "API key for authentication.",
            "API key for authorized access",
            "API key obtained from 2Factor.in",
            "API key obtained from http://2Factor.in",
            "API key to authenticate the request.",
            "Optional API key",
            "Optional API key for authentication.",
            "The API key",
            "The API key associated with the user account.",
            "The API key for accessing Amazon product data.",
            "The API key for accessing Amazon's e-commerce platform.",
            "The API key for authentication",
            "The API key from ScraperAPI",
            "The API key obtained from 2Factor.in",
            "The API key of the Twitter account.",
            "The API key of the user making the API call",
            "The API key provided by the medical institution",
            "The API key required for accessing Amazon's search results",
            "The API key required for authentication",
            "The API key required for authentication.",
            "The API key required for scraping data from Amazon Turkey",
            "The API key required to access the Mars rover images API.",
            "The API key to authenticate the request.",
            "The API key used to authenticate with the weather service.",
            "The unique identifier for the API key.",
            "api key",
        }
    ),
    "apikey": frozenset(
        {
            "API key for authentication",
            "API key for authentication. Defaults to 'demo'.",
            "API key for the Financial Modeling Prep API.",
            "An API key for authentication and rate limiting.",
            "The API key",
            "The API key associated with the account. You can obtain the API key at https://app.rivet.solutions/ApiDocument/ApiDocs once your account is created.",
            "The API key for Alpha Vantage",
            "The API key for accessing Alpha Vantage API.",
            "The API key for accessing the Earning Call Transcript API.",
            "The API key for accessing the FMP Cloud API",
            "The API key for accessing the FMP Cloud API. Obtainable from https://fmpcloud.io/register.",
            "The API key for authentication",
            "The API key for authentication.",
            "The API key obtained after signing up on the IP World website",
            "The API key obtained from FMP Cloud",
            "The API key required for accessing the Financial Modeling Prep service.",
            "The API key required for authentication",
            "The Rivet SMS API key. Obtain it from https://app.rivet.solutions/ApiDocument/ApiDocs#.",
            "The cubiculus application key used for authentication",
            "Your API Key - Obtain the API key from your dashboard",
            "Your API key from https://fmpcloud.io/register.",
        }
    ),
    "auth_token": frozenset(
        {
            "The authentication token from the login process.",
            "The authentication token of the client.",
        }
    ),
    "authentication_token": frozenset(
        {
            "The authentication token of the user",
        }
    ),
    "authorization": frozenset(
        {
            "Authorization token for the API request",
            "Authorization token for the API.",
            "Authorization token for the admin user",
            "Authorization token required for the API request.",
            "The authorization token required for API access.",
            "The authorization token required for accessing the API.",
            "The authorization token required to access the API.",
        }
    ),
    "code": frozenset(
        {
            "The SMS code sent to the user's mobile number.",
        }
    ),
    "confirmation_code": frozenset(
        {
            "The code provided to confirm the delivery.",
        }
    ),
    "credentials": frozenset(
        {
            "The authentication credentials of the personnel.",
            "The credentials required to authenticate the connection",
            "The credentials to authenticate with the video streaming service. This is typically a username and password or API key.",
        }
    ),
    "key": frozenset(
        {
            "API key for authentication.",
            "Account API Key",
            "Gives access to private memories and customized API limits",
            "The API Key used to access the API functions.",
            "The API key generated by Infodb.com",
            "The API key obtained from https://geocoder.opencagedata.com/",
            "The API key obtained from registering at https://geocoder.opencagedata.com/.",
            "The API key required for authentication.",
            "The GitHub API key.",
            "The access token for Github API authentication.",
            "The account API key.",
            "The decryption key to be used for AES decryption.",
            "The decryption key.",
            "The key to be used for decoding the message.",
            "TrumpetBox Cloud API KEY",
            "Your API Key that you get from your account on our website API key",
            "Your API Key. Each user has a unique API Key that can be used to access the API functions.",
            "Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.",
            "Your API key for authentication",
        }
    ),
    "otp": frozenset(
        {
            "The OTP that was sent.",
            "The OTP to use for verification.",
            "The one-time password that was sent via SMS.",
            "The six-digit One Time Password (OTP) provided by the user.",
        }
    ),
    "password": frozenset(
        {
            "Password associated with the username.",
            "Password for authentication",
            "Password for authentication. ",
            "Password for the username, same as the one used during account creation",
            "Password of the Gmail account",
            "Password used for decryption.",
            "The password associated with the username",
            "The password for database authentication",
            "The password for database authentication.",
            "The password for logging in, in clear text.",
            "The password for logging in.",
            "The password for login",
            "The password for login in clear text",
            "The password for login in clear text.",
            "The password for login in plain text.",
            "The password for the MySQL database.",
            "The password for the WiFi network",
            "The password for the social media account",
            "The password for the specified provider.",
            "The password for the wireless network",
            "The password of the account.",
            "The password of the client.",
            "The password of the employee.",
            "The password of the user attempting to access the database",
            "The password of the user for authentication.",
            "The password of the user to authenticate.",
            "The password of the user to log out.",
            "The password of the user to verify",
            "The password of the user who is sending the message. A token can also be used.",
            "The password of the user.",
            "The password of the user. A token can also be used.",
            "The password of the user. It should be at least 8 characters long, contain at least one uppercase letter, one lowercase letter, one number, and one special character.",
            "The password provided by the user.",
            "The password to authenticate the database",
            "The password to authenticate the shortcode",
            "The password to log in with",
            "The password to use for authentication",
            "The password to use when connecting to the database.",
            "The user's password for login in clear text.",
            "Your SensSMS API key.",
            "Your Wavecell Password",
        }
    ),
    "session_token": frozenset(
        {
            "A token representing the user's session.",
            "The session token from the opened application.",
            "The session token of the authenticated user.",
            "The session token of the user",
        }
    ),
    "token": frozenset(
        {
            "A free token obtained by sending a WhatsApp message with the command 'get-token' to the number +34 631 428 039.",
            "A free token obtained by sending a WhatsApp message with the command `get-token`.",
            "A valid token obtained by sending a WhatsApp message with the command 'get-token' to the provided number.",
            "A valid token obtained by sending a WhatsApp message with the command `get-token` to the number +34 631 428 039.",
            "API key for authentication",
            "API key for authentication.",
            "Authentication token (account) to retrieve bookable items for",
            "Authentication token for the Ecombr API.",
            "Authentication token for the current user",
            "Authentication token for the user.",
            "Authorization token received from authentication",
            "The API token for authentication. Defaults to 'TokenDemoRapidapi'.",
            "The API token for which to fetch the expiry time.",
            "The API token that needs to be verified.",
            "The JWT token to be validated.",
            "The token containing the application ID",
            "The token for API authentication. Defaults to 'TokenDemoRapidapi'.",
            "The token of the user to retrieve information for.",
            "The token of the user who is sending the message. A token can be obtained through check-user, and is valid until reset.",
            "The token of the user. A token can be obtained through check-user, and is valid until reset.",
            "The token that needs to be invalidated.",
            "The token used to summarize the video.",
            "To get a free token, click here: https://wa.me/34631428039?text=get-token to send a WhatsApp with the command `get-token`.",
            "Token for authentication",
            "Token for authentication with the API.",
            "Token sent in the email to confirm registration",
            "Your Serpstat API token",
        }
    ),
    "verification_data": frozenset(
        {
            "The data used to verify control over the DNS records.",
        }
    ),
    "verification_token": frozenset(
        {
            "The verification token provided in the link.",
        }
    ),
}

_MIXED_DESCRIPTION_FUNCTIONS = frozenset(
    {
        ("key", "Use this key for testing.", "Linkedin Contacts"),
        ("password", "The password", "verify_user"),
        ("password", "The password of the user", "login"),
        ("password", "The password of the user", "validateCredentials"),
        ("password", "The password to access the database", "deleteDatabase"),
    }
)

_NO_DESCRIPTION_FUNCTIONS = frozenset(
    {
        (
            "Authorization",
            "disciplina_2",
            "Retrieves disciplinary information for a specific student using the given authorization token.",
        ),
        (
            "api_key",
            "game_odds_by_site_schedule",
            "Fetches the game schedule from a specified sports betting site using the provided API key.",
        ),
        (
            "api_key",
            "retrieve_file",
            "Retrieves a file from the server using the provided file path and API key.",
        ),
        (
            "api_key",
            "stock_get_fund_profile",
            "Fetch the fund profile information for a given stock using the provided ticker ID and API key.",
        ),
        (
            "api_key",
            "v1_convert",
            "Converts a sum of money from one currency to another using the specified conversion type and RapidAPI key.",
        ),
        (
            "apikey",
            "balansheet_financials",
            "Fetches and returns the financial data for a given company symbol using the specified RapidAPI key.",
        ),
        (
            "key",
            "Content Decrypt",
            "Decrypts the content of a URL field returned in the /market/get-reports endpoint.",
        ),
        (
            "key",
            "getprojects",
            "Fetches artificial intelligence projects from the specified page using the provided RapidAPI key.",
        ),
        (
            "key",
            "title_get_technical",
            "Fetches technical information for a movie or TV show using its tconst identifier and RapidAPI key.",
        ),
    }
)

_PUBLIC_ACCESS_CONSTANTS = frozenset(
    {
        ("apikey", "API key for authentication. Defaults to 'demo'.", "demo"),
        (
            "token",
            "The API token for authentication. Defaults to 'TokenDemoRapidapi'.",
            "TokenDemoRapidapi",
        ),
        (
            "token",
            "The token for API authentication. Defaults to 'TokenDemoRapidapi'.",
            "TokenDemoRapidapi",
        ),
    }
)


_HTTP_POST_URL_DESCRIPTION = (
    "For submitting information to create a new user in a database, modify this "
    "template URL with your actual API key where indicated: "
    "https://api.example.com/v1/users/create?apiKey={YOUR-API-KEY}"
)


def validate_source_placeholders(state: RowState) -> None:
    """Quarantine the exact released HTTP template when its key is unresolved.

    This is a source-template contract, not a credential projection for all
    URLs. A substituted URL remains untouched; a key provided separately by
    the user does not make an unmodified template executable.
    """
    for tool in state.sample.tools:
        function = tool["function"]
        if function["name"] != "http.post":
            continue
        parameters = function.get("parameters")
        properties = (
            parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        )
        schema = properties.get("url") if isinstance(properties, dict) else None
        if (
            not isinstance(schema, dict)
            or schema.get("description") != _HTTP_POST_URL_DESCRIPTION
        ):
            return
        for (mi, ci), arguments in state.parsed_arguments.items():
            call = state.sample.messages[mi]["tool_calls"][ci]
            value = arguments.get("url")
            if (
                call["function"]["name"] == "http.post"
                and isinstance(value, str)
                and "apiKey={YOUR-API-KEY}" in value
            ):
                raise Reject("unresolved_source_api_key_placeholder")
        return


def credential_requires_evidence(
    name: str, field: str, definition: dict[str, Any], value: Any
) -> bool:
    """Match only audited source semantics; keep literal field descriptions."""
    parameters = definition.get("parameters")
    properties = (
        parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    )
    schema = properties.get(field) if isinstance(properties, dict) else None
    description = schema.get("description") if isinstance(schema, dict) else None
    if isinstance(description, str):
        if (
            isinstance(value, str)
            and (field, description, value) in _PUBLIC_ACCESS_CONSTANTS
        ):
            return False
        if description in _EXISTING_DESCRIPTIONS.get(field, ()):
            return True
        if (field, description, name) in _MIXED_DESCRIPTION_FUNCTIONS:
            return True
    function_description = definition.get("description")
    if (
        description is None
        and isinstance(function_description, str)
        and (field, name, function_description) in _NO_DESCRIPTION_FUNCTIONS
    ):
        return True
    # Independently audited named source consumers, including variants whose
    # argument property description is absent in the released definition.
    return field in {
        "Get Amazon Product Details": {"api_key"},
        "Get Amazon Search Results": {"api_key"},
        "loginUser": {"password"},
        "login_user": {"password"},
        "verify_user": {"password"},
    }.get(name, ())
