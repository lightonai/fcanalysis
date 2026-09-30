"""Finite TxT360 credential predicates from the pinned agent release.

Audited LLM360/TxT360-3efforts at
bfc4a082d11967cd7810fe0b773be87bf54fb32e, all 34 agent parquet files.
These exact parameter descriptions declare existing authentication state. They
are source rules, not a generic keyword classifier. Missing descriptions need
an exact function/field/function-description contract. New account passwords,
custom OTP delivery, strength/hash checks, crypto tokens and ordinary keys or
codes are outside these predicates. Unclassified or ambiguous descriptions do
not acquire credential semantics from a field name alone. The expanded audit
checks actual called top-level fields by descriptor, including alternate API
key spellings, rather than limiting discovery to familiar credential names.
Mixed registration/login containers and computed signatures, key hashes or
authentication timestamps remain outside literal credential matching.

Only the explicitly released demo/test access constants below bypass prior-state
evidence. A static default alone does not establish public access. Exact
contracts supported only by such defaults remain outside this finite gate;
normalization still preserves their released definitions and argument values.
Invalid placeholders and examples do not establish authentication state.
"""

from typing import Any

_EXISTING_DESCRIPTIONS: dict[str, frozenset[str]] = {
    "ApiKey": frozenset(
        {
            "Use Rivet SMS API API key",
            "Your unique API key to authenticate the API request.",
        }
    ),
    "ClientId": frozenset(
        {
            "Your unique client ID to authenticate the API request.",
        }
    ),
    "access_token": frozenset(
        {
            "A valid Facebook OAuth access token with appropriate permissions (e.g., 'user_friends') to access the friends endpoint",
            "Authentication token for the user",
            "Authentication token issued by PropMix for API access. This token must be obtained through prior registration with PropMix sales team at sales@propmix.io",
            "Authentication token with appropriate permissions to access the file. Must be a valid Kloudless API access token",
            "Instagram Graph API access token for authenticated requests. Required for endpoints needing user permissions.",
            "The access token for the Instagram API.",
            "The access token received during authentication.",
        }
    ),
    "accesstoken": frozenset(
        {
            "Authentication token provided by PropMix during account registration. Contact sales@propmix.io for registration.",
            "Authentication token provided by PropMix upon registration. Required for API access - contact sales@propmix.io to obtain credentials.",
            "Authentication token provided by PropMix upon registration. This token must be included in all API requests to validate access permissions.",
        }
    ),
    "account": frozenset(
        {
            "Your Mopapp account identifier, chosen during account creation. Used for authentication and organization-specific data access.",
        }
    ),
    "account_id": frozenset(
        {
            "The cloud storage account ID to authenticate the request. If not specified, the primary account will be used.",
            "The unique identifier for the merchant's Cashtie™ account. This ID is used to authenticate and route payment verification requests to the correct account.",
        }
    ),
    "accountid": frozenset(
        {
            "Wavecell account ID for authentication and API access. This is your primary account identifier provided by Wavecell.",
        }
    ),
    "aid": frozenset(
        {
            "Account ID for authentication",
        }
    ),
    "api": frozenset(
        {
            "Client-specific license identifier or API key that requires validation against active licenses.",
            "The API key for accessing the Geokeo Forward Geocoding service.",
            "Your API key for accessing the Geokeo reverse geocoding service.",
            "Your API key obtained from ShortAdLink.",
            "Your Hajana One API key obtained from their official website. This authenticates your access to the SMS service.",
        }
    ),
    "apiKey": frozenset(
        {
            "The API key for authentication",
            "The API key used to authenticate the request.",
            "The write API key for the ThingSpeak channel.",
        }
    ),
    "apiTokenInstance": frozenset(
        {
            "API token used for authentication",
            "The API token for the user's account",
            "The API token used to authenticate the request.",
        }
    ),
    "api_key": frozenset(
        {
            "2Factor account API Key",
            "2Factor account API key with permissions to access service balance information",
            "API authentication key for Amazon services. If provided, enables authenticated requests with higher rate limits. If not specified, unauthenticated requests may be subject to API rate limits or data restrictions.",
            "API authentication key for accessing the Amazon data scraping service. Must be obtained from the service provider.",
            "API authentication key required for accessing the Amazon data scraping service. This key authenticates your requests and determines service access level.",
            "API authentication key required to access the Amazon scraper service. This key must be obtained through the service provider.",
            "API authentication key. Use 'test' for limited access (rate-limited) or obtain a premium key from https://ipdata.co/ for production use.",
            "API key for authenticating requests to Amazon's product data API. Must be obtained through Amazon's developer platform and maintained securely.",
            "API key for authenticating requests to the Amazon API. A valid key is required for successful operation.",
            "API key for authenticating requests to the Amazon India service. Required for successful API access. If not provided, an empty string is used (which may result in authentication errors).",
            "API key for authenticating requests to the Amazon Scraper service. Ensure this key has appropriate permissions and keep it secure.",
            "API key for authenticating requests to the Amazon data parser service. If not provided, uses the default placeholder value 'your_api_key_here' which is not valid for actual API access.",
            "API key for authenticating requests to the Amazon data scraper service. If not provided, a default empty string will be used, which may cause authentication failures. For production use, providing a valid API key is strongly recommended.",
            "API key for authenticating requests to the Amazon product data service. This key must be obtained from the service provider and maintained securely.",
            "API key for authenticating requests to the Amazon search service. Must be obtained from the service provider and maintained securely.",
            "API key for authenticating with the Amazon API service. Must be obtained from the service provider and maintained securely.",
            "API key for authenticating with the Amazon API service. Must be obtained from the service provider.",
            "API key for authenticating with the Amazon India data scraping service. If not provided, a default key may be used if available in the environment configuration.",
            "API key for authenticating with the Amazon data scraper service",
            "API key for authenticating with the Amazon data scraper service. Must be obtained through the service provider's dashboard",
            "API key for authenticating with the Amazon product data API. Required unless configured through environment variables or other system-wide settings.",
            "API key for authenticating with the Amazon scraper service. If not provided, requests may be subject to rate limiting or blocked.",
            "API key for authenticating with the Amazon web scraping service. A valid key is required for successful API access.",
            "API key for authentication",
            "API key for authentication.",
            "API key from ScraperAPI for authenticating requests. Register at [ScraperAPI](https://www.scraperapi.com) to obtain your API key. Format: string",
            "API key obtained from 2Factor.in",
            "API key obtained from ScraperAPI service (free keys available at https://www.scraperapi.com). This key authenticates and authorizes the API requests",
            "API key obtained from http://2Factor.in",
            "Amazon API authentication key for accessing product data. If not provided, requests may be subject to API rate limits or access restrictions",
            "Amazon API authentication key for authenticated requests. Required for production use, but optional for testing environments where anonymous access may be permitted.",
            "Amazon API authentication key with access to product review endpoints. This key must be provisioned through Amazon's developer portal with appropriate permissions",
            "Amazon API key for authenticating requests. Obtain from your AWS account. A valid API key is required for successful requests.",
            "Amazon Product API authentication key. This key is obtained from the Amazon Developer Console and grants access to marketplace data. Example: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "Authentication API key for accessing Amazon Product API. Must be kept secure and obtained through Amazon's developer portal",
            "Authentication API key for accessing Amazon product data services. Must be obtained from the service provider.",
            "Authentication API key for accessing Amazon product data. If the service requires authentication, provide a valid API key here. Leave empty if using an unauthenticated request or if credentials are managed through other means.",
            "Authentication API key for accessing Amazon's product database. Must be obtained from the service provider or API vendor prior to usage",
            "Authentication API key for accessing the Amazon Scraper service. If not provided, an empty string is used by default (note: actual API usage may require a valid API key obtained from the service provider).",
            "Authentication API key for accessing the Amazon Scraper service. Required for authorized access. If not provided, requests may be subject to rate limiting or denied.",
            "Authentication API key for accessing the Amazon data scraper API. Must be provided to ensure authorized access to the service.",
            "Authentication API key for accessing the Amazon data scraper service. This key must be obtained from the service provider and configured in the request headers.",
            "Authentication API key for accessing the Amazon product data service. If not provided, the function may use a predefined default key or require it to be set in the environment.",
            "Authentication API key for the NestStack Amazon Data Scraper service. Must be obtained from the service provider and maintained as a secret credential.",
            "Authentication API key obtained from 2Factor.in account",
            "Authentication API key obtained from 2Factor.in dashboard for service access",
            "Authentication API key obtained from 2Factor.in service. This key authorizes access to the voice OTP API and must be kept secure.",
            "Authentication API key obtained from CoinMarketCap's developer portal. Required for API access.",
            "Authentication API key obtained from your 2Factor.in account. This key is required to authorize the OTP request.",
            "Authentication API key required to access the Amazon Product Data Scraper API. Must be a valid string provided by the service provider.",
            "Authentication API key required to access the Amazon data scraping service. Must be kept secure and confidential.",
            "Authentication API key required to access the Amazon data scraping service. This key should be kept confidential and rotated periodically for security.",
            "Authentication API key required to access the Amazon product data API. This key must be obtained through the service provider's authentication system.",
            "Authentication API key required to access the Amazon product data service. This key must be obtained from the service provider and maintained securely.",
            "Authentication API key required to access the product data API. Contact the service provider for access credentials",
            "Authentication API key required to access the product search service. This key should be obtained from the service provider and maintained securely.",
            "Authentication API key with access permissions to the product reviews endpoint. The key must be provisioned through the platform's developer portal and maintain active status.",
            "Authentication key for API access and rate limiting control. If not provided, an empty string will be used as the default value.",
            "Authentication key for API access. Use 'test' (default) for limited access, or a personal API key from https://ipdata.co/ for production use",
            "Authentication key for ScraperAPI service. Free API keys are available at https://www.scraperapi.com. This key authenticates your requests and tracks usage limits.",
            "Authentication key for ScraperAPI service. Required for accessing Amazon product data. Register at [ScraperAPI](https://www.scraperapi.com) to obtain your API key. If not provided, the function will return an error.",
            "Authentication key for accessing Amazon India's API services. For production use, a valid API key is required. If not provided, defaults to 'default_api_key' (intended for testing purposes only).",
            "Authentication key for accessing Amazon Product API (provided by Amazon after registration)",
            "Authentication key for accessing Amazon Product API services. Must be kept secure and maintained with appropriate usage permissions.",
            "Authentication key for accessing Amazon Product API services. Must be obtained from the API provider.",
            "Authentication key for accessing Amazon Product API services. Must be obtained through Amazon Developer registration.",
            "Authentication key for accessing Amazon Product API. Should be a valid API key with appropriate permissions.",
            "Authentication key for accessing Amazon product API. Providing a valid API key ensures higher rate limits and access to premium product data. If not provided, a default placeholder value will be used, which may result in limited or restricted access.",
            "Authentication key for accessing Amazon product data API services. Must be obtained from the service provider's dashboard and kept confidential.",
            "Authentication key for accessing Amazon product data API. Must be obtained from the service provider",
            "Authentication key for accessing Amazon product data API. Must be obtained from the service provider.",
            "Authentication key for accessing Amazon product data API. This credential must be obtained from the API service provider.",
            "Authentication key for accessing Amazon product data APIs. If not provided, a placeholder key will be used, though providing your own is recommended for reliability and rate limit management.",
            "Authentication key for accessing Amazon product data APIs. Required for authorized access.",
            "Authentication key for accessing Amazon product data. Provided by the service provider. Must be kept secure and not exposed in client-side code.",
            "Authentication key for accessing Amazon product data. This string must be obtained from the Amazon API or service provider.",
            "Authentication key for accessing Amazon's API services. Must be obtained through Amazon's developer portal or authorized provider.",
            "Authentication key for accessing Amazon's API. Must be a valid API key with appropriate permissions.",
            "Authentication key for accessing Amazon's Product API. This must be a valid API key obtained from Amazon's developer portal. Ensure proper security handling to avoid exposure.",
            "Authentication key for accessing Amazon's product API. Must be obtained through Amazon's developer portal and have appropriate permissions for product data access.",
            "Authentication key for accessing Amazon's product API. Required for successful API calls.",
            "Authentication key for accessing Amazon's product API. This sensitive credential should be obtained through Amazon's developer portal and maintained securely.",
            "Authentication key for accessing Amazon's product API. This should be a valid API key obtained through Amazon's Associates Program or other authorized Amazon API access programs. The key must have appropriate permissions for product data retrieval.",
            "Authentication key for accessing Amazon's product API. This should be obtained from your Amazon Developer account or API service provider.",
            "Authentication key for accessing Amazon's product data API. Must be kept secure and should have appropriate permissions for product data access.",
            "Authentication key for accessing Amazon's product data API. Must be obtained from the service provider. While optional in the parameters, a valid API key is required for successful requests.",
            "Authentication key for accessing Amazon's product data API. Must be obtained through Amazon's developer portal or authorized reseller.",
            "Authentication key for accessing Amazon's product database. Providing an API key ensures authenticated access and may improve rate limiting allowances. If not provided, a default placeholder will be used.",
            "Authentication key for accessing Amazon's product search API. Must be a valid API key obtained through Amazon's developer portal or authorized provider.",
            "Authentication key for accessing Amazon's product search API. Must be a valid API key with appropriate permissions.",
            "Authentication key for accessing Amazon's product search API. This key must be obtained from the service provider or API documentation.",
            "Authentication key for accessing ScraperAPI. Required for service access. Free keys are available at ScraperAPI's official website.",
            "Authentication key for accessing the AIDs API. While technically optional, a valid API key is required for successful requests. Developers should replace the default value with their registered API key.",
            "Authentication key for accessing the API service. A valid API key must be provided in production environments.",
            "Authentication key for accessing the API service. This key is provided by the service provider to authorize requests.",
            "Authentication key for accessing the API. Format: string. Must be a valid API key with access to product reviews data.",
            "Authentication key for accessing the API. Must be a valid API key with appropriate permissions to retrieve product reviews.",
            "Authentication key for accessing the API. This must be provided by the service provider.",
            "Authentication key for accessing the Amazon API or scrapper service. If not provided, defaults to an empty string. Note: Some services may require a valid API key for successful requests.",
            "Authentication key for accessing the Amazon API service. Must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the Amazon API service. This key must be obtained from the service provider and kept confidential.",
            "Authentication key for accessing the Amazon API service. This must be a valid API key obtained from the service provider. Ensure this key is kept secure and not exposed in client-side code.",
            "Authentication key for accessing the Amazon API. Obtain this from your Amazon developer account or service provider.",
            "Authentication key for accessing the Amazon API. This should be obtained from your Amazon developer account or API provider.",
            "Authentication key for accessing the Amazon Big Data API. Must be a valid API key with appropriate permissions.",
            "Authentication key for accessing the Amazon DE data scraping API. Obtain this from your API provider or service administrator. This key authenticates your requests and determines API usage limits.",
            "Authentication key for accessing the Amazon Data Scraper API. Must be a valid API key with appropriate permissions.",
            "Authentication key for accessing the Amazon Data Scraper API. Must be obtained from the API provider.",
            "Authentication key for accessing the Amazon Data Scraper API. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon Data Scraper API. This key must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the Amazon Data Scraper API. This key must be obtained separately and maintained securely",
            "Authentication key for accessing the Amazon Data Scraper API. This key should be obtained from the service provider and maintained securely",
            "Authentication key for accessing the Amazon Data Scraper API. This should be a valid API key with appropriate access permissions.",
            "Authentication key for accessing the Amazon India data scraping API",
            "Authentication key for accessing the Amazon India web scraping API. Must be a valid API key with appropriate permissions.",
            "Authentication key for accessing the Amazon India web scraping API. This must be obtained from the service provider and maintained as a secret credential.",
            "Authentication key for accessing the Amazon Kindle Scraper API. Required for authorized access. If not provided, the system may use a predefined default key.",
            "Authentication key for accessing the Amazon Product API or associated data scraping service. Must have active permissions for search operations and sufficient rate limits",
            "Authentication key for accessing the Amazon Product API or scrapper service. Must be provided by the user.",
            "Authentication key for accessing the Amazon Product API. If not provided, a placeholder key will be used (not valid for actual requests).",
            "Authentication key for accessing the Amazon Product API. Must be obtained from the API provider and maintained securely.",
            "Authentication key for accessing the Amazon Product API. Must be obtained through Amazon's developer portal and configured for product data access.",
            "Authentication key for accessing the Amazon Product API. Must be obtained through Amazon's developer portal and maintained in a secure configuration.",
            "Authentication key for accessing the Amazon Product API. This key is provided by the service administrator and grants access to product data.",
            "Authentication key for accessing the Amazon Product API. This key must be obtained from the API service provider and should be kept confidential.",
            "Authentication key for accessing the Amazon Product API. This should be obtained from your API provider and must be a valid string.",
            "Authentication key for accessing the Amazon Product API. While a default key is provided for basic access, users are encouraged to obtain their own API key from Amazon's developer portal for production use and higher rate limits.",
            "Authentication key for accessing the Amazon Product Data API. This key must be obtained from the service provider and have appropriate permissions.",
            "Authentication key for accessing the Amazon Product Data Scraper API. Must be a valid, authorized API key string provided by the service.",
            "Authentication key for accessing the Amazon Product Data Scrapper API. This key must be obtained through proper authorization channels and should be kept secure. Never expose API keys in client-side code or public repositories.",
            "Authentication key for accessing the Amazon Product Scraper API. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon Product Scrapper API. Must be obtained through the service provider.",
            "Authentication key for accessing the Amazon Products API. This must be a valid API key with appropriate permissions.",
            "Authentication key for accessing the Amazon Scraper API service. If not provided, the request will be made without authentication, which may result in limited access or rate restrictions.",
            "Authentication key for accessing the Amazon Scraper API service. Must be obtained from the API provider.",
            "Authentication key for accessing the Amazon Scraper API. If not provided, defaults to an empty string. Requests without a valid API key may be denied or subject to strict rate limiting.",
            "Authentication key for accessing the Amazon Scraper API. Must be a valid API key obtained from the service provider.",
            "Authentication key for accessing the Amazon Scraper API. Must be obtained from the service provider and maintained securely",
            "Authentication key for accessing the Amazon Scraper API. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon Scraper API. Must be obtained separately from the service provider. Required for successful API requests.",
            "Authentication key for accessing the Amazon Scraper API. Must be obtained through the service provider.",
            "Authentication key for accessing the Amazon Scraper API. Obtain this from your service provider and keep it confidential to prevent unauthorized access.",
            "Authentication key for accessing the Amazon Scraper API. Obtain your API key from the service provider or dashboard",
            "Authentication key for accessing the Amazon Scraper API. This key identifies the user and grants access to the service. Please ensure it is kept secure.",
            "Authentication key for accessing the Amazon Scraper API. This key is required to authorize requests and must be obtained through the service provider.",
            "Authentication key for accessing the Amazon Scraper API. This key must be obtained through the service provider",
            "Authentication key for accessing the Amazon Scraper API. This key must be obtained through the service provider's registration process.",
            "Authentication key for accessing the Amazon Scraper API. This must be a valid API key obtained from the service provider.",
            "Authentication key for accessing the Amazon Scraper API. This secret key grants access to the service and must be kept confidential.",
            "Authentication key for accessing the Amazon Scraper API. This secret key must be obtained through Amazon's developer portal or authorized reseller.",
            "Authentication key for accessing the Amazon Scraper API. This sensitive credential must be kept confidential and rotated periodically",
            "Authentication key for accessing the Amazon Scraper API. While optional, providing a valid API key is recommended to ensure higher rate limits and access to premium features.",
            "Authentication key for accessing the Amazon Scrapper API service. Must be obtained from the API provider and maintained securely",
            "Authentication key for accessing the Amazon Scrapper API. If omitted, the default value 'default_api_key' will be used. Note that this default is a placeholder and should be replaced with a valid API key for production use.",
            "Authentication key for accessing the Amazon Scrapper API. Must be a valid API key with active permissions for product data access.",
            "Authentication key for accessing the Amazon Scrapper API. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon Scrapper API. This key must be obtained from the API provider and have appropriate permissions for product data access.",
            "Authentication key for accessing the Amazon Scrapper Pro API. Must be a valid API key string obtained from the service provider.",
            "Authentication key for accessing the Amazon Scrapper Pro API. This key should be obtained from the API provider and maintained securely.",
            "Authentication key for accessing the Amazon Store Scraper API. Required for authorized access; some features may be limited without a valid key.",
            "Authentication key for accessing the Amazon Store Scraper API. Some endpoints may require this for authorized access.",
            "Authentication key for accessing the Amazon data API. Contact the service administrator to obtain a valid API key.",
            "Authentication key for accessing the Amazon data API. If omitted, requests may use a default quota-limited key or fail if authentication is required.",
            "Authentication key for accessing the Amazon data API. Must be kept secure and should have appropriate permissions.",
            "Authentication key for accessing the Amazon data scraper API service. Must be obtained through the service provider.",
            "Authentication key for accessing the Amazon data scraper API. Contact the service provider to obtain a valid API key",
            "Authentication key for accessing the Amazon data scraper API. Ensure this key is kept secure and not exposed in client-side code. Contact the API provider for key acquisition and management instructions.",
            "Authentication key for accessing the Amazon data scraper API. If not provided, the default empty string may result in unauthorized errors.",
            "Authentication key for accessing the Amazon data scraper API. Must be a valid API key obtained from the service provider.",
            "Authentication key for accessing the Amazon data scraper API. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon data scraper API. This key is provided by the service provider and must be kept confidential.",
            "Authentication key for accessing the Amazon data scraper API. This key is provided by the service provider.",
            "Authentication key for accessing the Amazon data scraper API. This key is required to authenticate requests and must be obtained through the service provider.",
            "Authentication key for accessing the Amazon data scraper API. This key must be obtained from the service provider and must be passed with each request.",
            "Authentication key for accessing the Amazon data scraper API. This key must be obtained through the service provider and must be kept confidential. The API key grants access to perform search operations against Amazon's product database.",
            "Authentication key for accessing the Amazon data scraper API. This must be obtained from the service provider and has rate-limiting restrictions.",
            "Authentication key for accessing the Amazon data scraping API",
            "Authentication key for accessing the Amazon data scraping API service",
            "Authentication key for accessing the Amazon data scraping API service. Must be a valid API key with active subscription.",
            "Authentication key for accessing the Amazon data scraping API service. Must be obtained from the service provider",
            "Authentication key for accessing the Amazon data scraping API service. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon data scraping API service. Required for successful API calls.",
            "Authentication key for accessing the Amazon data scraping API service. This secret key is provided by the service provider and must be kept confidential.",
            "Authentication key for accessing the Amazon data scraping API. A valid API key is required for authorized access.",
            "Authentication key for accessing the Amazon data scraping API. A valid API key is required to authorize requests. If not provided, a default placeholder key will be used, which may have limited access or rate restrictions.",
            "Authentication key for accessing the Amazon data scraping API. A valid API key is required to use this service. This parameter is optional but must be explicitly set by the user as the default value is not functional.",
            "Authentication key for accessing the Amazon data scraping API. Contact the service provider to obtain a valid API key.",
            "Authentication key for accessing the Amazon data scraping API. If not provided, a default placeholder is used, though providing a valid API key is recommended for production use.",
            "Authentication key for accessing the Amazon data scraping API. If not provided, defaults to an empty string. Users should replace this with their valid API key obtained from the service provider.",
            "Authentication key for accessing the Amazon data scraping API. If not provided, requests may be subject to rate limits or require alternative authentication methods.",
            "Authentication key for accessing the Amazon data scraping API. Must be a valid API key obtained from the service provider. If not provided, defaults to an empty string.",
            "Authentication key for accessing the Amazon data scraping API. Must be a valid API key obtained from the service provider. This key should be kept secure and not exposed in client-side code.",
            "Authentication key for accessing the Amazon data scraping API. Must be a valid API key string obtained from the service provider.",
            "Authentication key for accessing the Amazon data scraping API. Must be a valid API key string provided by the service administrator.",
            "Authentication key for accessing the Amazon data scraping API. Must be a valid API key with active service subscription.",
            "Authentication key for accessing the Amazon data scraping API. Must be a valid API key with appropriate access permissions.",
            "Authentication key for accessing the Amazon data scraping API. Must be a valid API key with appropriate permissions.",
            "Authentication key for accessing the Amazon data scraping API. Must be a valid, active API key with appropriate permissions.",
            "Authentication key for accessing the Amazon data scraping API. Must be kept confidential and obtained through proper authorization channels.",
            "Authentication key for accessing the Amazon data scraping API. Must be obtained from the API provider",
            "Authentication key for accessing the Amazon data scraping API. Must be obtained from the service administrator or API provider",
            "Authentication key for accessing the Amazon data scraping API. Must be obtained from the service provider",
            "Authentication key for accessing the Amazon data scraping API. Must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the Amazon data scraping API. Must be obtained from the service provider and maintained securely. Required for API access.",
            "Authentication key for accessing the Amazon data scraping API. Must be obtained from the service provider or API vendor.",
            "Authentication key for accessing the Amazon data scraping API. Must be obtained from the service provider or Amazon Web Services.",
            "Authentication key for accessing the Amazon data scraping API. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon data scraping API. Must be obtained through authorized channels.",
            "Authentication key for accessing the Amazon data scraping API. Must be obtained through the service provider and maintained securely.",
            "Authentication key for accessing the Amazon data scraping API. Obtain from the service provider. If not provided, uses a default key with potential rate limitations.",
            "Authentication key for accessing the Amazon data scraping API. Obtain this from your service provider or API dashboard.",
            "Authentication key for accessing the Amazon data scraping API. Requests without a valid API key may be rejected. If not provided, defaults to an empty string.",
            "Authentication key for accessing the Amazon data scraping API. Required for authorized access to product search functionality.",
            "Authentication key for accessing the Amazon data scraping API. Required for authorized access. If not provided, a default value of an empty string will be used, which may result in limited functionality or authentication errors.",
            "Authentication key for accessing the Amazon data scraping API. Required for production use to ensure rate limit allowance and service access.",
            "Authentication key for accessing the Amazon data scraping API. Required if the API implementation enforces key-based authentication. If not provided, the function will attempt to use environment variables or default credentials when available.",
            "Authentication key for accessing the Amazon data scraping API. This credential must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the Amazon data scraping API. This key authenticates requests and tracks API usage for rate limiting purposes. Keep this key secure and do not share publicly.",
            "Authentication key for accessing the Amazon data scraping API. This key authorizes access to the product data retrieval service.",
            "Authentication key for accessing the Amazon data scraping API. This key is provided by the service provider and is required for API access.",
            "Authentication key for accessing the Amazon data scraping API. This key is provided by the service provider and must be kept confidential.",
            "Authentication key for accessing the Amazon data scraping API. This key is provided by the service provider and must be kept secure.",
            "Authentication key for accessing the Amazon data scraping API. This key must be obtained from the API provider and maintained securely.",
            "Authentication key for accessing the Amazon data scraping API. This key must be obtained through the service provider and must be kept confidential.",
            "Authentication key for accessing the Amazon data scraping API. This must be a valid API key with appropriate permissions for product data retrieval",
            "Authentication key for accessing the Amazon data scraping API. This should be a string value provided by the service administrator.",
            "Authentication key for accessing the Amazon data scraping API. This should be a valid string provided by the service administrator.",
            "Authentication key for accessing the Amazon data scraping API. This should be obtained from the service provider or platform administrator.",
            "Authentication key for accessing the Amazon data scraping API. This should be obtained from your API service provider.",
            "Authentication key for accessing the Amazon data scraping API. Users must provide their own valid API key.",
            "Authentication key for accessing the Amazon data scraping service. Obtain this from your service provider or account dashboard.",
            "Authentication key for accessing the Amazon data scraping service. This key is provided by the service administrator or API provider.",
            "Authentication key for accessing the Amazon data scrapper API. Must be a valid string and kept secure. Contact the API provider for credential acquisition.",
            "Authentication key for accessing the Amazon data scrapper API. Must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the Amazon data scrapper API. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon data scrapper API. Must be obtained through the service provider and maintained securely.",
            "Authentication key for accessing the Amazon data scrapper API. This key must be obtained through authorized means and properly configured in your API client before making requests.",
            "Authentication key for accessing the Amazon data scrapper API. This should be obtained from your service provider or API documentation.",
            "Authentication key for accessing the Amazon e-commerce data scrapper service. Must be a valid, active API key with appropriate permissions.",
            "Authentication key for accessing the Amazon eCommerce data scraping API. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon product API",
            "Authentication key for accessing the Amazon product API. A default key may be used if not provided, though providing your own is recommended for reliability and security.",
            "Authentication key for accessing the Amazon product API. If not provided, the default placeholder value will result in unauthorized requests. Must be replaced with a valid API key from the service provider.",
            "Authentication key for accessing the Amazon product API. If provided, enables higher rate limits and access to premium features. When omitted, the function uses anonymous access with potential restrictions.",
            "Authentication key for accessing the Amazon product API. Must be a valid API key string. Keep this confidential and do not expose in client-side code.",
            "Authentication key for accessing the Amazon product API. Must be a valid API key with appropriate access permissions.",
            "Authentication key for accessing the Amazon product API. Must be a valid string with sufficient permissions.",
            "Authentication key for accessing the Amazon product API. Must be kept confidential and should not be hardcoded in production environments.",
            "Authentication key for accessing the Amazon product API. Must be obtained from the service provider or API documentation.",
            "Authentication key for accessing the Amazon product API. Must be obtained through service provider registration",
            "Authentication key for accessing the Amazon product API. Required for authorized access. If not provided, a default key configured in the system may be used if available.",
            "Authentication key for accessing the Amazon product API. This key must be kept confidential and should be sourced from your API provider.",
            "Authentication key for accessing the Amazon product API. This key must be obtained from the API provider and should be kept confidential. Refer to the API documentation for instructions on obtaining and managing API keys.",
            "Authentication key for accessing the Amazon product API. This key should be obtained through the service provider's dashboard and must be kept confidential.",
            "Authentication key for accessing the Amazon product API. This must be a valid API key obtained from the service provider. Ensure proper security handling as this key grants access to product data.",
            "Authentication key for accessing the Amazon product API. This must be a valid API key with appropriate permissions for product data retrieval.",
            "Authentication key for accessing the Amazon product API. This secret key must be obtained from the service provider and kept confidential. Format: 32-character alphanumeric string",
            "Authentication key for accessing the Amazon product API. This should be obtained from your API provider or service administrator.",
            "Authentication key for accessing the Amazon product API. This should be obtained through the service provider's dashboard or API management portal.",
            "Authentication key for accessing the Amazon product data API. Contact the service provider for access credentials.",
            "Authentication key for accessing the Amazon product data API. If not provided, requests may be subject to rate limiting or restricted data availability.",
            "Authentication key for accessing the Amazon product data API. If not provided, uses 'default_api_key' as placeholder (replace with valid key for production use).",
            "Authentication key for accessing the Amazon product data API. Must be a valid API key string provided by the service.",
            "Authentication key for accessing the Amazon product data API. Must be a valid API key string provisioned through the service provider's dashboard.",
            "Authentication key for accessing the Amazon product data API. Must be a valid API key with appropriate permissions for product review access.",
            "Authentication key for accessing the Amazon product data API. Must be a valid API key with appropriate permissions.",
            "Authentication key for accessing the Amazon product data API. Must be a valid API key with proper permissions to query Amazon's product database.",
            "Authentication key for accessing the Amazon product data API. Must be a valid API key with sufficient permissions",
            "Authentication key for accessing the Amazon product data API. Must be kept secure and provided by the service provider.",
            "Authentication key for accessing the Amazon product data API. Must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the Amazon product data API. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon product data API. Must be obtained through the service provider",
            "Authentication key for accessing the Amazon product data API. Must be obtained through the service provider.",
            "Authentication key for accessing the Amazon product data API. Providing a valid API key is recommended for secure and reliable access.",
            "Authentication key for accessing the Amazon product data API. The key must have active permissions for the service and should be stored securely.",
            "Authentication key for accessing the Amazon product data API. This key is provided by the API service provider and must be kept confidential.",
            "Authentication key for accessing the Amazon product data API. This key is provided by the API service provider.",
            "Authentication key for accessing the Amazon product data API. This key is required to authorize access to the service.",
            "Authentication key for accessing the Amazon product data API. This key must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the Amazon product data API. This key must be obtained from the service provider.",
            "Authentication key for accessing the Amazon product data API. This key should be obtained from the API provider and have appropriate permissions configured.",
            "Authentication key for accessing the Amazon product data API. This key should be obtained from your service provider or API documentation.",
            "Authentication key for accessing the Amazon product data API. This key should be obtained through proper authorization channels and maintained securely.",
            "Authentication key for accessing the Amazon product data API. This must be a valid API key with appropriate permissions for product data access.",
            "Authentication key for accessing the Amazon product data API. This string should be obtained from your service provider or account dashboard.",
            "Authentication key for accessing the Amazon product data scraping API. This key must be obtained from the service provider.",
            "Authentication key for accessing the Amazon product reviews API. Obtain this from your service provider or API documentation. This key ensures authorized access to the review data.",
            "Authentication key for accessing the Amazon product scraper API service. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon product scraper API. Must be obtained from the service provider and kept secure",
            "Authentication key for accessing the Amazon product scraping API service. Must be obtained through the service provider",
            "Authentication key for accessing the Amazon product scraping API. Required for all requests and must be obtained through the service provider.",
            "Authentication key for accessing the Amazon product search API. Must be a valid API key obtained from the service provider.",
            "Authentication key for accessing the Amazon product search API. Must be a valid API key with appropriate permissions.",
            "Authentication key for accessing the Amazon product search API. Must be obtained from the service provider",
            "Authentication key for accessing the Amazon product search API. Must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the Amazon product search API. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon scraper API service. This key must be obtained from the API provider and maintained securely.",
            "Authentication key for accessing the Amazon scraper API service. This key should be obtained from the service provider and must be included in all requests for successful API authentication.",
            "Authentication key for accessing the Amazon scraper API. This key grants access to the product data extraction service.",
            "Authentication key for accessing the Amazon scraping API service. Must be obtained through the service provider's dashboard and maintained as a secure credential.",
            "Authentication key for accessing the Amazon scraping API. Leave empty if authentication is handled through other means or if using a pre-configured integration.",
            "Authentication key for accessing the Amazon scraping API. Must be a valid API key string provided by the service administrator.",
            "Authentication key for accessing the Amazon scraping API. Must be a valid string obtained from the service provider. Keep this value secure.",
            "Authentication key for accessing the Amazon scraping API. Must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the Amazon scraping API. Obtain this from your service provider or API documentation. This is a sensitive credential and should be handled securely.",
            "Authentication key for accessing the Amazon scraping API. This should be obtained from your service provider or API documentation.",
            "Authentication key for accessing the Amazon scrapper API. Must be obtained from the service provider and maintained securely",
            "Authentication key for accessing the Amazon scrapper API. This key must be obtained from the API provider and have appropriate permissions for product data access.",
            "Authentication key for accessing the Amazon search API. Contact the API provider to obtain a valid API key.",
            "Authentication key for accessing the Amazon search API. This key must be obtained from the service provider and has rate-limiting implications.",
            "Authentication key for accessing the Amazon web scraper API. This secret key must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the Amazon web scraping API. Contact the API provider for access credentials",
            "Authentication key for accessing the Amazon web scraping API. Must be a valid API key with active service subscription.",
            "Authentication key for accessing the Amazon web scraping API. Must be a valid API key with appropriate permissions for product data retrieval.",
            "Authentication key for accessing the Amazon.de data scraping API. If not provided, anonymous access will be used if permitted by the service.",
            "Authentication key for accessing the Demiurgos Amazon Scraper API service",
            "Authentication key for accessing the Junkzon auto parts marketplace API",
            "Authentication key for accessing the ScraperAPI service. Must be obtained by registering at https://www.scraperapi.com.",
            "Authentication key for accessing the ScraperAPI service. Obtain a free API key at https://www.scraperapi.com",
            "Authentication key for accessing the ScraperAPI service. Register at [https://www.scraperapi.com](https://www.scraperapi.com) to obtain an API key. Format: 32-character alphanumeric string.",
            "Authentication key for accessing the TripFro API services",
            "Authentication key for accessing the currency conversion API. Must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the currency exchange rate API",
            "Authentication key for accessing the e-commerce API. Must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the e-commerce API. This secret key should be kept confidential and is typically provided by the service provider",
            "Authentication key for accessing the e-commerce data API. If not provided, a placeholder value will be used (note: actual API access requires a valid key).",
            "Authentication key for accessing the e-commerce data scraping service. Must be kept confidential and provided as a string",
            "Authentication key for accessing the e-commerce data scrapper API",
            "Authentication key for accessing the e-commerce platform's API",
            "Authentication key for accessing the e-commerce platform's API, provided by the service administrator",
            "Authentication key for accessing the e-commerce platform's API. Must be a valid API key with review data permissions.",
            "Authentication key for accessing the e-commerce platform's API. Must be a valid, active API key with appropriate permissions.",
            "Authentication key for accessing the e-commerce platform's API. Must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the e-commerce platform's API. Must be obtained through platform registration and maintained securely.",
            "Authentication key for accessing the e-commerce platform's API. This key should be provided by the service administrator or obtained through the platform's developer portal.",
            "Authentication key for accessing the e-commerce product data API. Must be a valid API key with appropriate permissions to query product offers.",
            "Authentication key for accessing the eCommerce platform's API. A valid API key is required for successful requests. If not provided, a default placeholder value is used.",
            "Authentication key for accessing the financial data API",
            "Authentication key for accessing the metal price API service",
            "Authentication key for accessing the metalpriceapi service. Must be obtained from the service provider and included in all API requests.",
            "Authentication key for accessing the product data API. Must be kept confidential.",
            "Authentication key for accessing the product data scraping API. Must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the product database API. Should be kept secure and obtained through proper authorization channels.",
            "Authentication key for accessing the product database. Required for authorized access. If not provided, the function may use a default key or fail if authentication is mandatory.",
            "Authentication key for accessing the product reviews API service. If not provided, the function will use the default API key configured in the environment.",
            "Authentication key for accessing the product reviews API. Must be a valid API key string obtained from the service provider.",
            "Authentication key for accessing the product reviews API. Must be obtained from the service provider",
            "Authentication key for accessing the product reviews API. This key should be obtained through the platform's developer portal and maintained securely.",
            "Authentication key for accessing the review data API",
            "Authentication key for accessing the sports score API. This is required for all API operations.",
            "Authentication key for accessing the underlying API service. Some APIs may require this for full functionality or rate-limited access.",
            "Authentication key for accessing the underlying API. If not provided, the request may be subject to API rate limiting or restricted access.",
            "Authentication key for accessing the web scraping API. Must be obtained from the service provider and maintained in a secure environment.",
            "Authentication key for accessing the wkalidev-amazon-scraper API. Must be a valid API key obtained from the service provider.",
            "Authentication key for the API service. If not provided, defaults to an empty string.",
            "Authentication key for the Amazon Data Scraper API. This key verifies access permissions and tracks API usage. For security, avoid hardcoding in client-side code and use environment variables or secure credential management systems.",
            "Authentication key for the Amazon Scraper API. Users must obtain this API key from the service provider and maintain its confidentiality.",
            "Authentication key for the Amazon scraping API. A valid API key is required for successful requests. If not provided, an empty string will be used, which may result in authentication errors.",
            "Authentication key for the Rauf Amazon Scraper API service. Must be obtained from the API provider and maintained securely. Should not be hardcoded in production environments.",
            "Authentication key for the Scraper API service. Must be obtained from the Scraper API provider and maintained securely.",
            "Authentication key for the amazon_data_scraper_2 API. Must be obtained from the service provider or API documentation.",
            "Authentication key from ScraperAPI service for accessing the scraping API",
            "Authentication key granting access to the e-commerce API. Must be kept secure and confidential.",
            "Authentication key granting access to the product data API. Must be kept confidential and provided by the service administrator",
            "Authentication key granting access to the product database or marketplace API",
            "Authentication key obtained from 2Factor.in API service. Must be a valid API key with SMS sending permissions configured.",
            "Authentication key obtained from 2Factor.in dashboard for API access authorization.",
            "Authentication key obtained from 2Factor.in service. This API key must have permissions to modify blocklist settings.",
            "Authentication key provided by the API service provider for accessing Amazon's product database. Must be kept confidential and used in accordance with the service's terms of service.",
            "Authentication key required to access Amazon Product API services. This should be a valid API key with appropriate permissions for product data retrieval.",
            "Authentication key required to access the Amazon API. Must be a string provided by the service administrator.",
            "Authentication key required to access the Amazon API. Must be a valid API key with appropriate permissions for product data retrieval.",
            "Authentication key required to access the Amazon API. Users must obtain this key from the service provider or API documentation",
            "Authentication key required to access the Amazon India Web Scraper API. Must be obtained from the service provider and stored securely.",
            "Authentication key required to access the Amazon Product API. Must be obtained through Amazon's developer portal or authorized provider.",
            "Authentication key required to access the Amazon Product API. Must be obtained through Amazon's developer portal or authorized service provider. Ensure proper API usage rights for commercial applications.",
            "Authentication key required to access the Amazon Product API. This key is used for identifying and authorizing API requests.",
            "Authentication key required to access the Amazon Scraper API. Must be a valid API key obtained from the service provider. Ensure this key is kept secure and not exposed in client-side code",
            "Authentication key required to access the Amazon Scraper API. This key is obtained through your service provider and should be kept confidential",
            "Authentication key required to access the Amazon Scraper API. This secret key must be obtained from the API provider and included in all requests for authorization.",
            "Authentication key required to access the Amazon data retrieval API. Must be obtained through the service provider's dashboard or platform administrator.",
            "Authentication key required to access the Amazon data scraper API. Must be obtained from the service provider and included in all requests. This parameter is mandatory for successful API calls despite being marked as optional in the schema.",
            "Authentication key required to access the Amazon data scraper API. This sensitive credential should be kept secure and rotated periodically. Follow the service's security best practices for API key management",
            "Authentication key required to access the Amazon data scraping API. If not provided, the request will be made without authentication, which may result in limited access or rate restrictions.",
            "Authentication key required to access the Amazon data scraping API. Must be obtained from the service provider and kept secure.",
            "Authentication key required to access the Amazon data scraping API. This should be a string obtained from the service provider.",
            "Authentication key required to access the Amazon data scraping API. This should be a valid API key provided by the service administrator.",
            "Authentication key required to access the Amazon product API. Obtain this from your API provider or dashboard.",
            "Authentication key required to access the Amazon product API. This key must be obtained from the service administrator or API provider. When not provided, API access may be restricted.",
            "Authentication key required to access the Amazon product API. This key must be obtained through the service provider's developer portal and maintained securely.",
            "Authentication key required to access the Amazon product API. This should be obtained from the API provider or service administrator.",
            "Authentication key required to access the Amazon product data API. This credential must be obtained from the API service provider and maintained securely.",
            "Authentication key required to access the Amazon product data API. This key must be obtained from the API provider and included in all requests.",
            "Authentication key required to access the Amazon product data API. This key must be obtained from the service provider and should be provided as a string value.",
            "Authentication key required to access the Amazon product scraper API",
            "Authentication key required to access the Amazon product search API. Obtain this from your API provider.",
            "Authentication key required to access the Amazon product search API. This key must be obtained from the API service provider and maintained securely.",
            "Authentication key required to access the Amazon product search API. This should be a string provided by the service administrator or obtained through authorized channels.",
            "Authentication key required to access the Amazon scraping API. Users must obtain this key from the service provider.",
            "Authentication key required to access the commerce API. This should be obtained through the platform's developer portal or API management system",
            "Authentication key required to access the e-commerce platform's API. Must be obtained from the service provider.",
            "Authentication key required to access the eCommerce platform's API. This key should be kept confidential and rotated periodically for security.",
            "Authentication key required to access the product data API. Must be obtained from the service administrator.",
            "Authentication key required to access the product data API. This key should be obtained from your account dashboard and must be kept confidential.",
            "Authentication key required to access the product reviews API. Must be a string provided by the service administrator.",
            "Authentication key required to access the product reviews API. Must be obtained from the service provider and maintained as a secure credential.",
            "Authentication key to access the Amazon product API. This key grants access to product data and must be kept confidential.",
            "Authentication token for API access. Must be a valid key with appropriate permissions for product data retrieval.",
            "Authentication token for API access. Obtain your free API key from https://ar-code.com/",
            "Authentication token for Scraper API services. Users must obtain this key through the ScraperAPI platform to establish valid API connections and handle request authentication.",
            "Authentication token for accessing the Amazon DataScraper API. Must be a valid API key with active subscription.",
            "Authentication token for accessing the Amazon Kindle scraper API. If not provided, requests may be subject to API rate limits or restrictions.",
            "Authentication token for accessing the Amazon Scraper API. Must be obtained from the API provider and included in all requests.",
            "Authentication token for accessing the Amazon Scraper API. This key must be obtained from the service provider and must have active permissions for product review scraping",
            "Authentication token for accessing the Amazon Scrapper API. If the service requires authentication, provide a valid API key here.",
            "Authentication token for accessing the Amazon data scraping API. Contact the service provider for credentials. Leave empty for unauthenticated access (limited functionality).",
            "Authentication token for accessing the Amazon data scraping API. Must be obtained from the service provider.",
            "Authentication token for accessing the Amazon product data API. Contact the service provider to obtain your API key.",
            "Authentication token for accessing the Amazon web scraping API. Must be a valid API key with active service access",
            "Authentication token for accessing the Amazon.de data scraping API. Must be kept confidential and should not be exposed in client-side code.",
            "Authentication token for accessing the e-commerce API. If not provided, requests may be subject to rate limiting or restricted access.",
            "Authentication token for the Amazon Scraper API. If not provided, the function will attempt to use an API key configured in environment variables. This key grants access to the Amazon product database.",
            "Authentication token obtained from 2Factor.in dashboard. This API key must have 'VOICE_BLOCKLIST' permissions enabled for the operation to succeed.",
            "Authentication token obtained from 2Factor.in service. Must be kept confidential and passed as a bearer token for API requests.",
            "Authentication token required to access Amazon product data. Must be obtained through Amazon's developer portal and maintained securely.",
            "Authentication token required to access Amazon's product API or third-party scraping service. Must be obtained through official API registration or service subscription. Keep this key secure and do not expose it in client-side code.",
            "Authentication token required to access the API. This key should be obtained through your account settings or API provider",
            "Authentication token required to access the Amazon API service. This key must be obtained through proper authorization channels and maintained securely.",
            "Authentication token required to access the Amazon Data Scraper API service. Must be kept confidential.",
            "Authentication token required to access the Amazon Product API. Users must obtain and provide a valid API key for successful requests.",
            "Authentication token required to access the Amazon data scraping API. Must be a valid API key obtained from the service provider",
            "Authentication token required to access the Amazon data scraping API. Should be obtained from your API provider (e.g., RapidAPI) and kept confidential. Must have active subscription and sufficient usage quota.",
            "Authentication token required to access the Amazon data scraping API. This key grants access to product data and must be kept secure.",
            "Authentication token required to access the Amazon data scraping API. This key identifies the requesting party and enforces rate limiting",
            "Authentication token required to access the Amazon data scraping API. This key must be obtained from the service provider and properly configured for API access.",
            "Authentication token required to access the Amazon data scraping API. This key must be obtained through the service provider and maintained securely.",
            "Authentication token required to access the Amazon data scrapper API. This key is provided by the service provider and must be kept confidential.",
            "Authentication token required to access the Amazon product API. Must be obtained through the service provider's dashboard.",
            "Authentication token required to access the Amazon product data scraping API. Must be obtained through the service provider's dashboard.",
            "Authentication token required to access the Amazon web scraping API. Must be obtained from the service provider and maintained securely.",
            "Authentication token required to access the Junkzon API services. This key must be obtained through the provider's developer portal and maintained securely.",
            "Authentication token required to access the VAT calculation service. Should be kept confidential and rotated periodically.",
            "Authentication token required to access the e-commerce API. This key is provided by the service provider and must be kept confidential.",
            "Authentication token required to access the e-commerce platform's API. This key must be obtained through the platform's developer portal and maintained securely.",
            "Authentication token required to access the eCommerce data scraping API",
            "Authentication token required to access the product catalog API. Must be a valid API key issued by the service provider.",
            "Authentication token required to access the product database or API endpoint. Must be a valid API key with appropriate permissions.",
            "Authentication token required to access the product database. This key must be provisioned by the service provider and maintained securely",
            "Authentication token required to access the product search API. Must be obtained through the service provider and included in all requests for authorization",
            "Authentication token required to access the weather API. Must be obtained from the service provider.",
            "Authentication token required to access the weather data API. Must be a valid API key string provided by the service",
            "Authentication token with required permissions to access Amazon product data. Keep this key secure and do not expose it in client-side code.",
            "Authorized API access key with drossi eCommerce platform. Must be provisioned through the platform's developer portal and maintained in a secure credential store.",
            "Developer API key for authenticating requests to the Amazon product search service. Must be obtained through Amazon's developer portal.",
            "Optional authentication key for accessing the API. If provided, enables authorized access with higher rate limits; if omitted, uses default anonymized scraping behavior.",
            "ScraperAPI authentication key for accessing Amazon data scraping services. Must be obtained from the ScraperAPI dashboard.",
            "ScraperAPI authentication key for service access. Must be obtained from ScraperAPI's dashboard and maintained securely",
            "ScraperAPI authentication key required for requests. Must be a valid key obtained from the ScraperAPI service",
            "The API key associated with the user account.",
            "The API key for authenticating requests to the Amazon product reviews service. Must be obtained through proper authorization channels and kept secure.",
            "The API key for authenticating requests to the Amazon product search service. A valid API key is required for successful operation.",
            "The API key for authenticating requests to the OpenAI API.",
            "The API key for authentication",
            "The API key from ScraperAPI",
            "The API key obtained from 2Factor.in",
            "The API key of the user making the API call",
            "The API key provided by the medical institution",
            "The API key required for accessing Amazon's search results",
            "The API key required for authentication",
            "The API key required for scraping data from Amazon Turkey",
            "The API key to authenticate the request.",
            "The API key used for authenticating the request.",
            "The Blockmate API authentication token with appropriate project permissions. This key authenticates the requesting service to the crypto account connector.",
            "The ConvertKit account's API key used for authentication. This identifier is typically found in your ConvertKit account settings alongside the API secret.",
            "The authentication key for accessing the Amazon data scraping API. This key must be obtained from the service provider and has predefined rate limits and permissions.",
            "The authentication key required to access the Amazon Product API. This key must be obtained through Amazon's developer portal and maintained securely.",
            "The authentication key required to access the Amazon Scraper API service. This key must be obtained from the service provider and must be valid for successful API requests.",
            "The authentication key required to access the Amazon Scraper API. This key must be obtained separately and must be valid for the API to return results.",
            "The authentication key required to access the Amazon Scraper API. This key must be obtained separately and passed as a string for successful API requests.",
            "The authentication key required to access the e-commerce platform's API. This key must be obtained and configured by the user prior to use.",
            "The unique API key credential issued by the service provider for client authentication.",
            "The unique identifier for the API key.",
            "Unique API key for authenticating requests to the VAT validation service. Must be obtained through service provider registration.",
            "Unique authentication token obtained from the service provider. Keep this private and pass it in all API requests for identification.",
            "User authentication key for API access. This key verifies the requester's permissions to interact with the store's data.",
            "User-specific API authentication key. This key verifies the caller's authorization to access the store's data.",
            "User-specific authentication token for accessing the e-commerce data scraping API. Must be kept confidential and obtained through the service provider's registration process.",
            "Valid API authentication key for accessing the Amazon data scraper service",
            "Valid API key for authenticating requests to the Amazon data scraper service. Must be obtained through the service provider's authentication system",
            "Valid API key for authenticating requests to the Amazon data scraping service. Ensure this key has appropriate permissions.",
            "Valid API key from ScraperAPI service. Obtain your key at https://www.scraperapi.com/ before using this function.",
            "Valid Amazon Product API authentication key with appropriate access permissions. Must be kept confidential.",
            "Valid ScraperAPI key for accessing the scraping service. Obtain one at https://www.scraperapi.com/. Required for authentication.",
            "Your API key for authentication. Obtain a free API key by registering at https://ar-code.com/",
            "Your Amazon Product API authentication key. This key is required to authenticate requests to Amazon's Product API. Obtain it through Amazon's Developer Console after registering for product advertising API access.",
            "Your CoinMarketCap API key for authentication. Obtain this key from your CoinMarketCap developer account dashboard.",
            "Your ScraperAPI key for authentication. Obtain it by registering at https://www.scraperapi.com/",
            "Your specific API key for the service. Defaults to 'your_api_key'.",
            "Your unique API key for authenticating with the Amazon data scraper service. Keep this secure and do not share it publicly.",
            "api key",
        }
    ),
    "api_secret": frozenset(
        {
            "The ConvertKit account's API secret used for authentication. This sensitive value is typically found in your ConvertKit account settings under API credentials.",
        }
    ),
    "api_token": frozenset(
        {
            "Authentication token for accessing the Autonix API. This token authorizes access to your account's usage data and must be included in all requests. The token can be obtained through the Autonix dashboard or administrator interface.",
            "Authentication token for accessing the Autonix API. This token must be generated in your account settings and have appropriate permissions",
            "Authentication token for accessing the Sendbird API. Must be a valid API key with appropriate permissions.",
            "Authentication token required for API access. Register for a token at http://deep.social/ and view pricing options at http://deep.social/prices",
            "Authentication token required to access the API. Must be included in request headers as 'Authorization: Bearer <token>'",
        }
    ),
    "apieyk": frozenset(
        {
            "Authentication token or API key required to access the Amazon product search API. This key must be obtained from the service provider and maintained securely.",
        }
    ),
    "apikey": frozenset(
        {
            "API authentication key for accessing the Amazon Scraper service. This key authenticates and authorizes access to the scraping API",
            "API authentication key required for accessing the Repustate service. Must be obtained through service registration.",
            "API authentication token with administrative permissions to modify user quotas",
            "API key for FMP Cloud authentication (obtain at https://fmpcloud.io/register)",
            "API key for authenticating requests to the Amazon India data scraping service. This key grants access to the product data API.",
            "API key for authenticating requests to the Monitr Financial Data service. This key must be obtained through the platform's API management console.",
            "API key for authenticating requests to the financial data service. Contact the data provider for access credentials.",
            "API key for authenticating requests to the geolocation simulation service. This key must be obtained through the service's authentication system and must be included in all API calls.",
            "API key for authenticating the request.",
            "API key for authenticating the request. This key grants access to the geolocation simulation service and must be kept confidential.",
            "API key for authenticating with the Amazon search API. Must be a valid string provided by the service provider",
            "API key for authenticating with the Amazon search API. This key should be obtained through the appropriate Amazon developer portal or service.",
            "API key for authenticating with the MyMappi service.",
            "API key for authenticating with the financial data service",
            "API key for authenticating with the financial data service. Must be provided as a string.",
            "API key for authenticating with the geolocation service. Obtain a free key by registering at https://ipworld.info/signup",
            "API key for authenticating with the sentiment analysis service. This key should be kept secure and not exposed in client-side code.",
            "API key for authentication",
            "API key for authentication. Defaults to 'demo'.",
            "API key for the Financial Modeling Prep API.",
            "API key with administrative permissions to modify minimum duration settings. Format: String value issued by the service provider.",
            "Alpha Vantage API authentication key. Users must register at https://www.alphavantage.co to obtain a valid API key.",
            "An API key for authentication and rate limiting.",
            "Authentication API key required to access the Amazon India data scraping service. This key must be obtained from the API provider and maintained securely",
            "Authentication API key with required permissions for system access",
            "Authentication API key with required permissions to access simulation data",
            "Authentication key for API access. Register at https://fmpcloud.io/register to obtain an API key.",
            "Authentication key for accessing Amazon Product API services. Must be obtained through Amazon Associates Program registration and maintained securely. Required for API access.",
            "Authentication key for accessing Amazon's product API. This should be obtained from your service provider or API documentation.",
            "Authentication key for accessing Amazon's product data API. Must have appropriate permissions for product data access",
            "Authentication key for accessing the API. Obtain from your account dashboard at the service provider's website.",
            "Authentication key for accessing the Agify.io API service. Must be a valid API key string obtained from the service provider.",
            "Authentication key for accessing the Amazon India API. Must be obtained through authorized channels",
            "Authentication key for accessing the Amazon India API. This key must be obtained from Amazon's developer portal or service provider.",
            "Authentication key for accessing the Amazon Product API. Must have appropriate permissions for review data access",
            "Authentication key for accessing the Amazon Product Scraper API. When not provided, uses default API key with limited rate limits.",
            "Authentication key for accessing the Amazon Scraper API. Must be kept secure and comply with Amazon's API usage policies.",
            "Authentication key for accessing the Amazon Scraper API. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon Scraper API. Must be obtained through the service provider and maintained securely. Ensure compliance with API usage policies and rate limits.",
            "Authentication key for accessing the Amazon Scraper API. Must be obtained through the service provider.",
            "Authentication key for accessing the Amazon Scraper API. This must be a valid API key with sufficient permissions to fetch product reviews.",
            "Authentication key for accessing the Amazon Scraper API. This should be kept confidential and obtained through authorized channels.",
            "Authentication key for accessing the Amazon data scraping API service. Must be a valid API key with sufficient permissions",
            "Authentication key for accessing the Amazon data scraping API. Must be obtained from the service provider or API marketplace.",
            "Authentication key for accessing the Amazon product data API. This key must be obtained through the service provider's registration process and maintained securely.",
            "Authentication key for accessing the Amazon product review API. This key must be obtained from the service provider and included in all requests for authorization.",
            "Authentication key for accessing the Amazon product search API. Must be obtained from the service provider.",
            "Authentication key for accessing the Amazon web scraping API service. This key must be obtained from the API provider and maintained securely.",
            "Authentication key for accessing the commerce API. Obtain this from your API provider or service administrator.",
            "Authentication key for accessing the commerce platform's API. Must be a valid API key with appropriate permissions.",
            "Authentication key for accessing the financial data API. Must be a valid API key issued by the service provider.",
            "Authentication key for accessing the financial data API. Must be obtained from the service provider and maintained securely.",
            "Authentication key for accessing the financial data API. Replace with your actual API key if required by the service.",
            "Authentication key for accessing the financial_modeling_prep API. A valid API key is required for successful requests.",
            "Authentication key for accessing the premium financial data API",
            "Authentication key for the requesting user. Must be a valid API key string.",
            "Authentication key granting access to the API endpoint",
            "Authentication key granting access to the geolocation simulation API. Must be a valid API key with appropriate permissions for group management operations.",
            "Authentication key required to access the API. Must be a valid string obtained from the service provider. This key is used to authenticate and authorize access to application data.",
            "Authentication key required to access the API. This sensitive credential should be kept confidential and rotated periodically.",
            "Authentication key required to access the Amazon Scraper API. This key is provided by the service provider.",
            "Authentication key required to access the IP geolocation API. This key identifies the client application and authorizes access to the service.",
            "Authentication token for API access. Required for production environments (test key may work for development).",
            "Authentication token for API access. Required for production environments. Default value provided for testing/demo purposes only.",
            "Authentication token for accessing the geolocation simulation service. Must be obtained through prior registration or authorization process.",
            "Authentication token required to access the API. This key must be obtained through the platform's authentication system and must be included in all API requests.",
            "Authentication token required to access the Amazon product API. This should be obtained through the appropriate service provider or Amazon API documentation.",
            "Authentication token used to authorize access to the geolocation simulation API. Format: 'Bearer <token>'",
            "Authentication token with required permissions to access simulation data. Must be a valid API key string formatted as a UUID (e.g., 'a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8')",
            "Authentication token with required permissions to execute this operation",
            "Stormglass API authentication key with valid access permissions for tide data endpoints",
            "Stormglass API authentication key with valid permissions",
            "The API key associated with the account. You can obtain the API key at https://app.rivet.solutions/ApiDocument/ApiDocs once your account is created.",
            "The API key for accessing Alpha Vantage API.",
            "The API key for accessing the Earning Call Transcript API.",
            "The API key for accessing the mymappi service.",
            "The API key for accessing the service.",
            "The API key for authentication",
            "The API key for authentication.",
            "The API key obtained from FMP Cloud",
            "The API key required for accessing the Financial Modeling Prep service.",
            "The API key required for authenticating requests to the Amazon API",
            "The API key used to authenticate access to the Amazon Web Scraper service. This key must be kept secure and should not be shared publicly.",
            "The Rivet SMS API key. Obtain it from https://app.rivet.solutions/ApiDocument/ApiDocs#.",
            "The cubiculus application key used for authentication",
            "User authentication key for accessing the financial data API",
            "User's API key for authentication and authorization",
            "Valid API key for authenticating the request. This key grants access to the Pixmac Stock Photos API and must be kept confidential.",
            "Valid API key with authorized access to the Amazon product API. Must be properly configured for authentication.",
            "Valid Amazon Product API authentication key with appropriate permissions. Must be obtained from Amazon's developer portal and maintained securely.",
            "Your API Key - Obtain the API key from your dashboard",
            "Your API access key for service authentication",
            "Your API key for authenticating requests to the Amazon web scraper service. This key grants access to the underlying API endpoint.",
            "Your API key from https://fmpcloud.io/register.",
        }
    ),
    "apitokeninstance": frozenset(
        {
            "API token for Green API authentication. This token grants access to the WhatsApp instance management endpoints and must be kept secure.",
            "Authentication token for API access, used to validate the request",
            "Authentication token for API access. Format: string (e.g., 'abcd1234-5678-efgh-90ab') - should be kept secret and securely stored",
            "Authentication token for API access. This secure token must be generated through the Green API dashboard and grants authorized access to receive notifications for the specified WhatsApp instance.",
        }
    ),
    "app_hash": frozenset(
        {
            "Telegram API application hash obtained from https://my.telegram.org/apps. This serves as a secret key for authenticating API requests.",
        }
    ),
    "app_key": frozenset(
        {
            "API key for application authentication. This key identifies the requesting application to the Twitter API.",
            "Your unique Yotpo application key obtained during account registration. This serves as your API authentication credential.",
        }
    ),
    "appid": frozenset(
        {
            "Authentication token or API key required to access the employment archive system",
            "Shopee affiliate application ID for API authentication. This ID is provided in your Shopee affiliate program dashboard.",
            "Shopee application ID for API authentication and identification",
        }
    ),
    "appkey": frozenset(
        {
            "API authentication key with required permissions for organization data access",
            "Authentication key identifying the application with access to organization data. Must be obtained from your application dashboard and securely stored.",
        }
    ),
    "apptoken": frozenset(
        {
            "Application authentication token obtained from the /signinfo/ endpoint.",
            "Application token for API access.",
            "Application token for API authentication and authorization",
            "Application token obtained from the authentication endpoint",
            "Application-specific authentication token obtained through the authorization process",
            "Application-specific authentication token provided by OnlyFans for API access",
            "Application-specific token for API access authentication",
        }
    ),
    "auth": frozenset(
        {
            "Authentication token",
        }
    ),
    "authToken": frozenset(
        {
            "The authentication token to access the healthcare database. This should be a string of alphanumeric characters.",
        }
    ),
    "auth_id": frozenset(
        {
            "Authentication ID obtained through the auth endpoint",
            "Authentication session ID from cookie.auth_id. Must remain consistent across requests.",
            "Authentication session identifier obtained from prior authorization endpoint",
            "Authentication token obtained through the authorization endpoint",
            "Unique identifier for the authenticated account session",
            "Unique identifier for the authenticated session, obtained from the auth endpoint",
        }
    ),
    "auth_token": frozenset(
        {
            "API key or bearer token required for authentication with the third-party service. When not provided, requests will be made without authentication headers.",
            "Authentication token or API key for accessing the advertising API",
        }
    ),
    "authentication_details": frozenset(
        {
            "Authentication details for both Salesforce and Pega Platform.",
        }
    ),
    "authentication_token": frozenset(
        {
            "The authentication token for API access.",
            "The authentication token of the user",
            "The token to authenticate the API call.",
        }
    ),
    "authenticationguid": frozenset(
        {
            "Authentication GUID or API key required to access the Australian Business Register service",
        }
    ),
    "authkey": frozenset(
        {
            "Authentication key required to access the API. Must be obtained through service provider authorization.",
            "Authentication token for API access. Must be a valid API key with datacenter read permissions.",
            "Unique API key or token required for authenticating requests to the OS data service",
        }
    ),
    "authoriza": frozenset(
        {
            "An additional authorization token. Defaults to None.",
        }
    ),
    "authorization": frozenset(
        {
            "API access key for authentication. This key must have appropriate permissions to access product data.",
            "API access token for authentication",
            "API access token for authentication, typically in 'Bearer <token>' format",
            "API access token or OAuth bearer token for authenticating the request.",
            "API access token or authentication credential required to access the e-commerce platform's resources",
            "API access token with appropriate permissions to access order data. Format: 'Bearer <token>'",
            "API authentication token with appropriate permissions for product image access",
            "API authentication token with proper permissions for order access (e.g., 'Bearer <token>')",
            "API authorization token for authenticating the request. Typically a bearer token prefixed with 'Bearer '",
            "API authorization token required for accessing the location database service. Format: Bearer token (e.g., 'Bearer YOUR_API_KEY')",
            "API key for authenticating the request. Format: 'Bearer <your_api_key>'",
            "API key or bearer token for authenticating the request. Format depends on the cleardil API requirements (e.g., 'Bearer <token>' or a raw API key string).",
            "API key or bearer token for authenticating with the tax rate service. Format: 'Bearer <token>' or 'ApiKey <key>' depending on service requirements.",
            "Access token for API authentication in 'Bearer <token>' format",
            "Access token for authenticating the request, typically in 'Bearer <token>' format",
            "Access token for authenticating with the Genius API. This should be a valid bearer token formatted as 'Bearer <token>'",
            "Access token for authentication in the format 'Bearer <token>'. Must have sufficient permissions to access the requested image.",
            "Access token or API key used to authenticate the request and verify user permissions. Format depends on the authentication system (e.g., Bearer token, API key string)",
            "Alternative authentication token (e.g., OAuth 2.0 bearer token) that can be used for API access when provided. Defaults to empty string if not specified.",
            "Authentication bearer token for API access. Format: 'Bearer <base64-encoded-token>'",
            "Authentication bearer token for API access. Format: 'Bearer <token>'",
            "Authentication credentials for the request, typically formatted as a bearer token or API key",
            "Authentication token (e.g., 'Bearer <token>')",
            "Authentication token (e.g., Bearer token or API key) required to access the compliance service",
            "Authentication token (e.g., Bearer token or API key) required to authorize the request",
            "Authentication token for API access. Format depends on configured authentication scheme (e.g., Bearer token, OAuth2).",
            "Authentication token for API access. Format depends on the authentication scheme (e.g., 'Bearer <token>' or 'ApiKey <key>').",
            "Authentication token for API access. Format: 'Bearer <API_KEY>'",
            "Authentication token for API access. Format: 'Bearer <token>' or '<api_key>'",
            "Authentication token for API access. Format: 'Bearer <token>' or raw API key string. Must have fulfillment service permissions.",
            "Authentication token for API access. Must be a valid Bearer token formatted as 'Authorization: Bearer <your_api_key>'",
            "Authentication token for API access. Should be a valid Bearer token formatted as 'Bearer <token_value>'",
            "Authentication token or API key for accessing the imagegur API. Must be a valid string with sufficient permissions to retrieve image data.",
            "Authentication token or API key for secure access to the financial service endpoint. Required for authorized operations.",
            "Authentication token or API key required to access the gallery item data. Must be provided in the format specified by the API documentation (e.g., 'Bearer <token>').",
            "Authentication token or API key required to access the gallery service. This ensures secure and authorized access to the resource.",
            "Authentication token or API key required to access the service. Expected format: Bearer token (e.g., 'Bearer <token>') or API key string.",
            "Authentication token or credentials required to access the API (e.g., 'Bearer <token>' or 'Basic <base64_credentials>')",
            "Authentication token required for accessing social media account data. Format: 'Bearer <token>' for OAuth2 or 'API_KEY=<value>' for API key authentication",
            "Authentication token required to access the API. Format should be 'Bearer <token>' where <token> is a valid API key with appropriate permissions.",
            "Authentication token required to access the API. Format: Bearer <token>",
            "Authentication token required to access the API. Format: Bearer <token>. Must be obtained through prior authentication with the social media platform's API.",
            "Authentication token required to access the API. This should be a valid bearer token formatted as 'Bearer <your_token>'",
            "Authentication token required to access the API. Typically a Bearer token formatted as 'Bearer <your_token>'",
            "Authentication token required to access the API. Typically formatted as a bearer token (e.g., 'Bearer <your_token>')",
            "Authentication token required to access the social media API. Format: 'Bearer <token>' where <token> is a valid access token with appropriate permissions",
            "Authentication token with Bearer prefix. Example: 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.x'",
            "Authentication token with bearer credential format (e.g., 'Bearer <token>'). Must have sufficient permissions to access payee data.",
            "Authentication token with required permissions to access gallery submissions. Format: Bearer <token>",
            "Authorization token for API access.",
            "Authorization token for the API request",
            "Authorization token for the API.",
            "Authorization token for the admin user",
            "Authorization token required for the API request.",
            "Authorization token required to access the API.",
            "Bearer authentication token for accessing user account data. Format: 'Bearer <token>'",
            "Bearer token for API authentication",
            "Bearer token for API authentication and access control",
            "Bearer token for API authentication in 'Bearer <token>' format",
            "Bearer token for API authentication, formatted as 'Bearer <token_value>'",
            "Bearer token for API authentication. Format: 'Bearer <token>'",
            "Bearer token for API authentication. Format: 'Bearer <token>' where <token> is the API access token obtained from the authentication service.",
            "Bearer token for API authentication. Format: 'Bearer <token_value>'",
            "Bearer token for API authentication. Format: 'Bearer <your_token>' (e.g., 'Bearer abcdefghijklmnopqrstvwxyz')",
            "Bearer token for GitHub API authentication. Format: 'Bearer {token}'. Required for accessing private resources or when higher rate limits are needed. Example: 'Bearer ghp_0123456789abcdef0123456789abcdef01234567'",
            "Bearer token for Instagram API authentication. Format: 'Bearer <access_token>'",
            "Bearer token for alternative authentication. Format: 'Bearer <access_token>'. Takes precedence over authorization1 when provided.",
            "Bearer token for authenticated access. Optional.",
            "Bearer token for authenticating API requests. Format: 'Bearer <your_access_token>'",
            "Bearer token for authenticating API requests. Format: 'Bearer <your_token>'",
            "Bearer token for authenticating API requests. This token must be obtained through the platform's authentication system and must have appropriate permissions for accessing product data.",
            "Bearer token for authenticating the request (e.g., 'Bearer YOUR_TOKEN')",
            "Bearer token for authenticating the request to access resettlement market data",
            "Bearer token for authenticating the request. Required for accessing the API.",
            "Bearer token for authentication (e.g., 'Bearer <access_token>'). Must be included in the request header.",
            "Bearer token for authentication (e.g., 'Bearer <token>'). Must be included in the request header.",
            "Bearer token for authentication in the format 'Bearer <token>'. Must have appropriate permissions to access check attachments.",
            "Bearer token for authentication in the format 'Bearer {token_value}'. Required for accessing user account data.",
            "Bearer token for authentication, formatted as 'Bearer <token>'",
            "Bearer token for authentication, formatted as 'Bearer <token_value>'. Must be a valid session token with account access permissions.",
            "Bearer token for authentication. Format: 'Bearer <token>'",
            "Bearer token for authentication. Format: Bearer <token>",
            "Bearer token for request authentication. Format: 'Bearer <token>'. If not provided, alternative authentication methods may be used.",
            "Bearer token or API key for authenticating the request to access protected financial data.",
            "Bearer token or API key for authenticating the request. Format: 'Bearer <token>' or direct API key string",
            "Bearer token or API key for authenticating the request. Must have appropriate permissions to access the document.",
            "Bearer token used for authentication. Format: 'Bearer <token_value>'",
            "FRaaS API access token with appropriate permissions to access face repositories",
            "Instagram API access token for authentication. Format: Bearer <token>",
            "Instagram API access token formatted as 'Bearer <token>'",
            "Instagram API authentication token required for accessing user data. Format: 'Bearer <access_token>'",
            "OAuth 2.0 Bearer token for authenticating API requests. Format: 'Bearer {token}'",
            "OAuth 2.0 access token with 'account' scope for authenticated requests",
            "OAuth 2.0 bearer token for authenticating the request to access user favorites. Must be included in the format 'Bearer <token>'",
            "OAuth 2.0 bearer token for user authentication (e.g., 'Bearer <token>'). Must be included in the request header.",
            "OAuth Bearer token for authentication. Required to access restricted albums (marked as secret or hidden).",
            "OAuth2 authorization token for API access, formatted as 'Bearer <token>'",
            "Optional alternative authentication credential for API access. Format may vary based on authentication type",
            "Secondary authorization token for additional security layers. Optional parameter that may be used for multi-factor authentication scenarios.",
            "The authorization token for accessing the API.",
            "The authorization token required for API access.",
            "The authorization token required for accessing the API.",
            "The authorization token required for the API request.",
            "The authorization token required to access the API.",
            "The authorization token to access the API.",
            "WhatsApp Business API administrative token with 'business_settings' scope. Format: 'Bearer <API_TOKEN>' where <API_TOKEN> is a valid WhatsApp Business API key with administrative privileges.",
            "WhatsApp Business Admin API token with required permissions. Format: 'Bearer <your_api_token>'",
        }
    ),
    "authorization1": frozenset(
        {
            "Bearer token for authenticating API requests. Format: 'Bearer YOUR_API_KEY'",
            "PandaDoc API key with appropriate document access permissions. Required when no bearer token is provided in the 'authorization' parameter. Format: 'YOUR_API_KEY'",
            "Primary API key or access token for authenticating the request to the PandaDoc API. Must be a valid credential with document access permissions.",
            "Primary API key or access token for authenticating the request. Typically formatted as a Bearer token (e.g., 'Bearer <your_token>').",
            "Primary authentication token (API key or bearer token) required for API access",
            "Primary authentication token for API access (e.g., Bearer token or API key)",
            "Primary authentication token for API access. This should be a valid PandaDoc API key with appropriate document access permissions.",
            "Primary authentication token required for API access. Format: 'Bearer <token>'",
            "Primary authorization token for API access. Format depends on the authentication scheme (e.g., 'Bearer <token>' or 'ApiKey <key>').",
            "Primary authorization token for API access. Should contain a valid Bearer token or API key with sufficient permissions to access log data.",
            "Primary bearer authentication token (e.g., 'Bearer YOUR_API_KEY'). Required for API access.",
        }
    ),
    "authorization_token": frozenset(
        {
            "A secure token for authorizing the action.",
        }
    ),
    "authtoken": frozenset(
        {
            "Authentication token for Zoho API access",
            "Authentication token for accessing the Zoho Creator API.",
            "The authentication token for the API request",
        }
    ),
    "bot_token": frozenset(
        {
            "The Telegram bot's API token, obtained from the BotFather. Format: '123456789:ABCdefGhIJKlmNoPQRstUvwxYZ'",
        }
    ),
    "cert_key": frozenset(
        {
            "Certificate key for authentication",
        }
    ),
    "channel_token": frozenset(
        {
            "LINEチャンネルアクセストークンを指定します。API認証に必要なシークレットキーです。",
        }
    ),
    "channelaccesscode": frozenset(
        {
            "Access code provided by the publisher for subscriber authentication. Some publishers may require this code to authorize access to their active trades.",
            "Authentication token granting access to the specific trade channel. Must be a valid, pre-configured access code with appropriate permissions.",
        }
    ),
    "cid": frozenset(
        {
            "Client ID for authentication.",
        }
    ),
    "client": frozenset(
        {
            "API client identifier for authenticating with the Kiniscore service",
            "Authentication client ID or API key provided by Kiniscore for API access. Must be a valid string credential.",
            "Client identifier used to authenticate API requests. This string should uniquely identify the client application or user.",
        }
    ),
    "clientInfo": frozenset(
        {
            "Information about the client requesting the token.",
        }
    ),
    "client_id": frozenset(
        {
            "Authentication key for accessing the Ditto Photo Reader API. Obtain your API key by contacting help@dittolabs.io",
            "The unique application identifier issued by QuantiModo for your client application. Contact info@quantimo.do to obtain this credential.",
        }
    ),
    "client_secret": frozenset(
        {
            "Confidential application secret associated with client_id. Required for server-side token exchanges. Must be stored securely and never exposed to client-side code",
            "The client secret issued to the application during the registration process.",
            "The client secret issued to the client during the registration process.",
            "The client's secret key. Defaults to None.",
            "The confidential key associated with your client_id. Used to authenticate your application during token exchange. Keep this value secure.",
            "Your client secret key, found on your settings page.",
        }
    ),
    "clientid": frozenset(
        {
            "Client account identifier for authentication purposes",
            "Client application ID registered with the RosaCrypto API for authentication.",
            "Unique API client identifier provided by ROSA Crypto for application authentication",
            "Unique identifier for the API client account. Used in conjunction with the secret key for authentication.",
        }
    ),
    "clientsecert": frozenset(
        {
            "Authentication secret key for API access. This sensitive credential should be securely managed by the client application.",
            "Client secret key for API authentication. Must be kept confidential and match the client ID.",
            "Confidential API secret key used for request authentication. Must be kept secure and never exposed publicly.",
            "Secret key associated with the client ID for secure API authentication. Must be kept confidential.",
            "Secret key associated with the client application for authentication",
        }
    ),
    "clientsecret": frozenset(
        {
            "API client secret key for authenticating requests to the RosaCrypto exchange API",
            "Client secret key for API authentication",
        }
    ),
    "code": frozenset(
        {
            "The SMS code sent to the user's mobile number.",
            "The verification code received by the user",
        }
    ),
    "connection_id": frozenset(
        {
            "Unique identifier for the account connection. Must be a valid connection ID previously established through authentication handshake.",
        }
    ),
    "cookie": frozenset(
        {
            "A cookie to authenticate the request.",
            "Authentication cookie containing valid user session credentials. Must have permissions to initiate live streams in the specified room.",
            "Authentication cookie for accessing restricted API endpoints. When not provided, only public data will be accessible. Expected format: session cookie string (e.g., 'session_token=abc123xyz')",
            "Authentication cookie for session management. Must be a valid session cookie obtained through prior authentication.",
            "Authentication cookie for session persistence. If not provided, the request will be unauthenticated, which may limit access to certain match data.",
            "Authentication cookie for session persistence. Required for maintaining stateful connections.",
            "Authentication cookie for the Social platform",
            "Authentication cookie for the social media platform",
            "Authentication cookie for the specified account",
            "Authentication cookie for user session validation",
            "Authentication cookie for user session validation. Required for accessing protected user data.",
            "Authentication cookie required to access the social media platform's API. This cookie must contain valid session credentials with comment-read permissions.",
            "Cookie for authentication",
            "HTTP cookie for session authentication. Required for endpoints needing session-based authorization",
            "LinkedIn session cookie value (li_at) for authentication. Must be obtained from an active LinkedIn account session.",
            "Session authentication cookie granting permissions to manage live streams. Must be valid and authorized for stream termination actions.",
            "Session cookie for authenticated requests. Format should match standard HTTP cookie headers.",
            "Session cookie for authenticating the API request. Must contain valid credentials with streaming permissions.",
            "Session cookie for authentication. Required when maintaining user session context. If not provided, anonymous access will be attempted.",
            "Session cookie for maintaining API authentication",
            "Session cookie for maintaining authentication state during paginated requests. Optional for public searches.",
            "TikTok authentication cookie string in Netscape format, used to verify user identity and permissions",
            "User session cookie containing authentication tokens and preference data. Providing this parameter ensures offers are tailored to the user's account status and betting history.",
            "User's authentication cookie",
            "Valid Twitter session cookie obtained from an authenticated browser session. Must contain the complete cookie header value after logging into Twitter (e.g., 'auth_token=...; ct0=...')",
            "Value of the 'li_at' cookie for authentication.",
        }
    ),
    "credentials": frozenset(
        {
            "Credentials required to access the asset.",
            "The credentials required to access the external storage.",
            "The credentials to authenticate with the video streaming service. This is typically a username and password or API key.",
        }
    ),
    "creds_uuid": frozenset(
        {
            "GUID for credential identification",
            "Globally Unique Identifier (GUID) for the authentication credentials",
            "User identifier for authentication. Required if token is not provided.",
        }
    ),
    "ctrlkey": frozenset(
        {
            "Authentication control key provided by the service provider for API access. Must be a valid alphanumeric string",
        }
    ),
    "cvv": frozenset(
        {
            "Card verification value (3-4 digits). Must match card issuer's security code requirements.",
            "The 3-digit card verification value (CVV) of the credit card.",
        }
    ),
    "database_credentials": frozenset(
        {
            "The credentials for the database connection",
        }
    ),
    "e_mail": frozenset(
        {
            "Email address associated with the marketplace seller account, used for request authorization",
            "Registered seller email address used for account authentication and identification.",
            "Registered seller email address used for marketplace authentication",
            "Seller's registered email address for API authentication",
            "Seller's registered email address used for account verification and communication.",
        }
    ),
    "figmatoken": frozenset(
        {
            "User-generated access token granting programmatic access to Figma files. This token must be obtained through Figma's developer portal and should be kept confidential.",
        }
    ),
    "getkey": frozenset(
        {
            "Authentication key or access token required to retrieve financial market data. This key verifies access permissions to the financial data API",
        }
    ),
    "holderapikey": frozenset(
        {
            "Authorization key for the holder account with permission to modify duration limits.",
            "Holder's API key for authorization context, typically representing an administrative or supervisory account",
        }
    ),
    "is_id": frozenset(
        {
            "User's unique identifier required for authentication. Must be provided as a string value representing the user ID.",
        }
    ),
    "jwt": frozenset(
        {
            "A JSON Web Token for authentication.",
        }
    ),
    "k": frozenset(
        {
            "Private API key for authentication with the Pushsafer service. Must be kept secure and match the format provided in your Pushsafer account settings.",
        }
    ),
    "kapi_proxy": frozenset(
        {
            "Proxy authentication token for accessing Kwai API services. If not provided, system will use default proxy configuration.",
        }
    ),
    "key": frozenset(
        {
            "API authentication key for accessing protected resources. Leave empty if no authentication is required.",
            "API authentication key for service access. Required for production usage",
            "API authentication key for service access. This key authorizes usage of the LinkedIn outreach functionality.",
            "API authentication key for testing purposes",
            "API authentication key required for accessing the service. Must be obtained from the service provider.",
            "API authentication key required to access the service. This key authenticates and authorizes your application to use the Autocomplete API.",
            "API key for authenticating requests (availability may depend on service configuration)",
            "API key for authentication.",
            "API key granting access to private translation memories and customized API rate limits",
            "API key to access private memories and customize API limits.",
            "API key with 10 allowed uses per day. Must be a string obtained through the service's authentication system.",
            "Authentication API key for service access. Users must register to obtain a unique API key.",
            "Authentication API key for the Amazon Japan scraper service. This identifies the user and authorizes access to the API.",
            "Authentication API key or session token required for accessing Shopee's data. Must have proper permissions for seller profile access",
            "Authentication API key required to access the Amazon Japan Scraper service. This key must be obtained through proper authorization channels.",
            "Authentication access key or API token provided by the service provider. This key grants authorized access to the CNPJ lookup service.",
            "Authentication access key or token required to access the Amazon Japan Scraper API. This should be a valid API key provided by the service administrator.",
            "Authentication access key required to authorize API requests",
            "Authentication key for accessing the financial data API. Must be a valid API key string provided by the service administrator.",
            "Authentication key granting access to the route calculation service",
            "Authentication key used to validate request origin. Must match predefined system secret.",
            "Authentication token for API access. Must be a valid string formatted as a UUID (e.g., '550e8400-e29b-41d4-a716-446655440000') or API key string provided by the service. This parameter ensures secure access to the communication platform.",
            "GitHub personal access token (PAT) for authentication. Required for accessing private repositories or when rate limits require authentication. Omitting this parameter will result in unauthenticated requests.",
            "The API Key used to access the API functions.",
            "The API access key for authenticating requests to the ZIP code lookup service. Users must provide a valid key for successful API calls.",
            "The API key obtained from registering at https://geocoder.opencagedata.com/.",
            "The API key required for authentication.",
            "The API key used to authenticate requests to the LocationIQ service. This key identifies the user account whose credit balance will be checked. While optional in the schema, providing a valid key is required for accurate account-specific results.",
            "The access token for Github API authentication.",
            "The account API key.",
            "The user's unique API key, obtained through account registration. This key authenticates API requests and provides access to service functionality.",
            "TrumpetBox Cloud API KEY",
            "Unique API key for authentication. Users must create an account to obtain this key.",
            "User-specific API authentication key. Required for accessing the API service. New users must register to obtain an API key.",
            "Your API Key that you get from your account on our website API key",
            "Your API Key. Each user has a unique API Key that can be used to access the API functions.",
            "Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.",
            "Your secret API key, found in the Mopapp dashboard under API settings. Required for secure API access.",
            "Your unique API key for authentication. Required for all API requests.",
        }
    ),
    "keyapi": frozenset(
        {
            "The API key required for authorization to access the list of accounts.",
        }
    ),
    "keyid": frozenset(
        {
            "Authentication key or token for API access. If not provided, uses the default system key. Users should replace this with their actual API key for secure access.",
            "The Key ID used for authentication or to identify the associated encryption key. If not provided, a default empty string is used.",
        }
    ),
    "license": frozenset(
        {
            "API license key for authenticating the request. Must be a valid, active key with sufficient permissions for the Proxy Detection API.",
            "API license key for authentication and access control",
            "API license key for authentication. Required for accessing the fraud detection service.",
            "API license key required for authentication. Must be a valid string provided by the service provider.",
        }
    ),
    "m_auth": frozenset(
        {
            "Authentication token obtained from login or registration, used to verify user identity.",
            "Authentication token obtained from successful login or registration, used to verify user identity and permissions",
            "Authentication token obtained from successful login/registration. Format: Bearer token string.",
            "User authentication token obtained after successful login or registration. Ensures verified user identity for comment submission.",
            "User authentication token obtained after successful login or registration. This token verifies the user's identity and permissions.",
            "User authentication token obtained after successful login or registration. Used to verify user identity and permissions.",
            "User authentication token obtained from login/registration API response",
            "User authentication token obtained from login/signup response",
            "User authentication token obtained through login or registration process",
            "User authentication token obtained through successful login or registration. Must be a valid session token with messaging permissions.",
        }
    ),
    "member_id": frozenset(
        {
            "Unique identifier for the seller account in Alibaba's system. This alphanumeric string is used to authenticate and authorize API requests for shop-specific operations.",
        }
    ),
    "merchant": frozenset(
        {
            "The merchant identifier used to authenticate API requests. This ID must be pre-registered with the Amazon data service API. When not provided, requests will be made without authentication which may limit available data.",
        }
    ),
    "merchant_id": frozenset(
        {
            "Unique identifier for the merchant account. Required to authenticate and authorize transaction data retrieval.",
        }
    ),
    "msid": frozenset(
        {
            "Active session identifier for the current user interaction. Must be a valid session token.",
        }
    ),
    "muid": frozenset(
        {
            "User identifier for the authenticated account holder",
            "User identifier representing the account holder. Must correspond to an authenticated user session.",
        }
    ),
    "oauth_consumer_key": frozenset(
        {
            "OAuth 1a consumer key for API authentication",
        }
    ),
    "oauth_token": frozenset(
        {
            "OAuth 1a access token for user authentication",
        }
    ),
    "orderful_api_key": frozenset(
        {
            "The API key to access Orderful.",
        }
    ),
    "otp": frozenset(
        {
            "Numeric verification code received from Telegram. Must be a positive integer (typically 5-6 digits). Note: Codes sent via Telegram's official application may expire immediately after being used by the client.",
        }
    ),
    "otp_input": frozenset(
        {
            "One-time password entered by the user for verification. Must match the format and length of the code sent during the initial OTP request (typically 4-6 numeric digits).",
            "The OTP value input by the end user",
            "User-entered one-time password (e.g., '123456'). Must match the format and length of the sent OTP.",
        }
    ),
    "p": frozenset(
        {
            "API Key",
            "API Key from Sms77.io.",
            "API key from Sms77.io",
            "API key from Sms77.io.",
            "The API Key for authentication.",
            "The API Key to authenticate the API request.",
            "Your API key from [Sms77.io](https://sms77.io).",
        }
    ),
    "password": frozenset(
        {
            "API authentication password. Must be kept secure and confidential",
            "Access password required to authenticate and verify permissions. Should contain the valid credentials for accessing restricted content.",
            "Account password associated with the provided username. Must match the credentials stored in the system",
            "Authentication credential for secure account access. Must be a string value matching the account's password requirements.",
            "Authentication password for accessing the SMS service",
            "Authentication password or API key for secure access to Wavecell services. Should be treated as sensitive information.",
            "Hajana One account password used for API authentication. Handle with care and ensure secure storage.",
            "Optional decryption password for accessing encrypted credential content. Must meet API security requirements (minimum 12 characters with mixed case, numbers, and symbols). If omitted or empty, only metadata will be returned without decrypted content.",
            "Password associated with the username.",
            "Password for the Gmail account authentication",
            "Password for the username, same as the one used during account creation",
            "Password of the Gmail account",
            "Secret credential associated with the user account. Must be provided in plain text format for authentication",
            "SensSMS API key for authenticating the request. Should be kept confidential.",
            "The Instagram account password used for authentication",
            "The password",
            "The password associated with the username.",
            "The password for logging in, in clear text.",
            "The password for logging in.",
            "The password for login",
            "The password for login in clear text",
            "The password for login in clear text.",
            "The password for login in plain text.",
            "The password for the MySQL server authentication.",
            "The password for the PostgreSQL server authentication.",
            "The password for the WiFi network",
            "The password for the social media account",
            "The password of the user",
            "The password of the user attempting to access the database",
            "The password of the user attempting to log in.",
            "The password of the user for authentication.",
            "The password of the user to log out.",
            "The password of the user to verify",
            "The password of the user. A token can also be used.",
            "The password to authenticate the shortcode",
            "The password to log in with",
            "The password to use for authentication",
            "The password to use when connecting to the database.",
            "The user's account password. Should contain at least 8 characters with a mix of letters, numbers, and symbols.",
            "The user's password for login in clear text.",
            "The user's password in plain text format. Should meet system security requirements (e.g., minimum length, complexity rules)",
            "The user's plaintext password for authentication. Must be transmitted securely and meet the system's password complexity requirements.",
            "The user's secret credential for authentication. Must be a string.",
            "The user's secret credential for authentication. Must be provided in plain text format.",
            "The user's secret credential used for authentication. Must be provided in plain text.",
            "User account password for authentication. Required unless a token is provided. Should be omitted when using token-based authentication.",
            "User authentication credential. Alternatively, a token may be used for authentication.",
            "User password for authentication. Required unless a token is provided via the 'token' parameter.",
            "User's two-factor authentication password if configured. Leave empty if 2FA is not enabled.",
            "Your SensSMS API key.",
        }
    ),
    "payment_token": frozenset(
        {
            "Secure payment token for transaction authorization",
        }
    ),
    "project_id": frozenset(
        {
            "The unique identifier of the project requiring authentication. Must match the project ID registered in the Blockmate system.",
        }
    ),
    "proxy": frozenset(
        {
            "Proxy identifier or authentication token used to validate and route API requests through the service proxy layer. Must be a string value formatted as a valid proxy key (e.g., 'proxy-12345')",
        }
    ),
    "proxy_secret": frozenset(
        {
            "X-RapidAPI proxy secret key for authentication. Required for secure API access.",
        }
    ),
    "publisherkey": frozenset(
        {
            "API key associated with the publisher account for service authorization. This parameter is typically required for API access.",
        }
    ),
    "publishertoken": frozenset(
        {
            "Authentication token for publisher account access. This parameter is typically required for API authentication.",
        }
    ),
    "pwd": frozenset(
        {
            "Password for the freesms8 account authentication",
        }
    ),
    "request_id": frozenset(
        {
            "The unique request identifier received from the /sendCode endpoint. This ID is required to track the specific authentication request.",
            "Unique identifier from the initial /sendCode request to associate this submission with a specific authentication attempt",
        }
    ),
    "secret": frozenset(
        {
            "API authentication key for accessing Kiniscore services. This should be a secure string provided by the service administrator.",
            "API authentication secret key provided to sellers for secure access to marketplace operations.",
            "API authentication token for accessing Kiniscore services. Must be a valid alphanumeric string provided by the service administrator.",
            "API client secret for authenticating with the Kiniscore service",
            "API secret key for authentication",
            "API secret key for authentication with the ecombr marketplace system",
            "API secret key for authentication. This confidential credential should be securely stored and never exposed publicly.",
            "API secret key for request authentication. Must be kept confidential and match the platform's expected value",
            "API secret key for secure authentication and authorization",
            "API secret key for secure authentication. Must match the registered account's secret.",
            "API secret key used for request authentication and signature generation",
            "Authentication client secret used to validate API requests. Must be a valid string credential matching the client ID.",
            "Authentication secret key associated with the client parameter. Example: 'api_key_abcdef12345'",
            "Authentication secret key for API access validation. Must be a securely generated alphanumeric string.",
            "Authentication secret key for API access. Should be a secure string provided by the feature flag management system administrator.",
            "Authentication secret key for API access. This identifies the seller's account securely.",
            "Authentication secret required for API access. This value should be a secure string provided by the service for authorized access.",
            "Authentication secret/token required for accessing the Kiniscore API. This value must be kept confidential and should be provided by the service administrator.",
            "Authentication token or API key for accessing the player database. Must be kept confidential and obtained through proper authorization channels.",
            "LoginRadius API secret key for application authentication. This secret should be securely stored and never exposed publicly.",
            "LoginRadius API secret key for application authentication. This sensitive credential should be securely stored and never exposed publicly.",
            "LoginRadius API secret key for authenticating requests. This should be obtained from your LoginRadius dashboard.",
            "LoginRadius API secret key used for request authentication. This serves as the credential to validate API access permissions.",
            "LoginRadius API secret key used for server-to-server authentication",
            "LoginRadius API secret key with required permissions for Facebook event access. Must be a valid alphanumeric string.",
            "Secret key for additional authentication.",
            "Secret key for authentication with the API.",
            "Shared secret key for request signature verification",
            "The LoginRadius API secret key used to authenticate API requests. This key should be kept confidential and sourced from your LoginRadius dashboard.",
            "The secret key for authenticating the API request.",
            "Your API Secret that you get from your account on our website",
        }
    ),
    "secret_key": frozenset(
        {
            "API authentication key required to access the provider data",
            "API authentication token required to authorize access to user data. This key must be generated through the OnePost developer portal and included in all requests.",
            "Authentication key used to verify identity and access permissions. This sensitive value must be kept confidential and should match the format specified by the API documentation.",
            "Authentication secret key required to access the API. This key should be generated through the provider's dashboard and maintained securely.",
            "Authentication token granting access to the user's event data. This key should be obtained through the service's authentication flow and must be kept confidential.",
            "Authentication token required to access event data. This secret key must be issued by the API provider and maintained securely.",
            "Authentication token required to authorize API access. Must be a valid API secret key associated with the user's account.",
            "Authentication token used to validate access to the webhook resource. Must match the secret key associated with the specified webhook.",
            "Authentication token with minimum 32 characters, containing alphanumeric and special characters for secure access",
            "Authentication token with required permissions to access social media resources. Must be a valid alphanumeric string provided by the service.",
            "Confidential authentication token required to validate access permissions. Must match the system-stored secret key for the specified page identifier.",
            "The API authentication secret key used to validate the request. This key ensures the caller has permission to access webhook data.",
            "The user's API authentication token with required permissions to access social media post data. Must be a valid secret key issued by the service",
        }
    ),
    "sess": frozenset(
        {
            "Session cookie value obtained from authentication",
            "Session identifier from authentication cookie (cookie.sess).",
            "Session token for maintaining authenticated state",
            "Session token from the authentication response",
            "Session token representing the current authenticated state for API operations",
        }
    ),
    "session": frozenset(
        {
            "Authentication session key from user/login endpoint. Session keys are non-expiring credentials - store and reuse them instead of re-authenticating for each request",
            "Authentication session key obtained from the user/login endpoint. Session keys are persistent and should be securely stored for future API interactions.",
            "Authentication session key obtained from user/login. Session keys are persistent and should be securely stored for subsequent API calls.",
            "Authentication session key obtained from user/login. Session keys are persistent and should be securely stored locally after initial login.",
            "Authentication session key obtained from user/login. Session keys are persistent and should be stored securely for future API calls",
            "Authentication session key obtained from user/login. Session keys remain valid indefinitely and should be securely stored for subsequent API calls.",
            "Authentication token obtained from the user/login endpoint. Session keys are non-expiring and should be stored securely for subsequent API calls.",
            "Authentication token obtained from the user/login function. Session keys are persistent and do not expire.",
            "The authentication session key obtained from user/login. Session keys are permanent and should be securely stored for subsequent API calls.",
            "The session key returned from the user/login API, used to authenticate the request.",
            "Unique identifier of the active session to be terminated. This typically corresponds to an authentication token or session ID provided during login.",
            "User session token for authentication. This should be a valid session ID obtained through prior authentication.",
        }
    ),
    "session_id": frozenset(
        {
            "A unique identifier representing the document session or temporary access token. This value is obtained through the PandaDoc interface (via cURL export) or API integration. Format: Alphanumeric string (e.g., 'sess_123456')",
            "The verification session ID returned in the send OTP step",
            "Unique verification session identifier obtained from the send OTP endpoint response. Used to associate the verification attempt with the original request.",
            "Verification session identifier returned by the send OTP API endpoint. Required to validate the associated OTP.",
        }
    ),
    "session_key": frozenset(
        {
            "Authentication token for Instagram API access. This session key must be valid and have appropriate permissions to retrieve user data.",
            "Authentication token obtained from Instagram login API for user validation",
            "Authentication token obtained from the Instagram login API. This key is required to maintain an authenticated session.",
            "Authentication token obtained from the login API, required for accessing Instagram's user data",
            "Authentication token obtained from the login API. Grants temporary access to Instagram's unofficial API endpoints.",
            "Authentication token obtained from the login API. Must be a valid, active session key.",
            "Authentication token obtained from the login API. Required for all Instagram account actions.",
            "Authentication token obtained from the login API. Used to verify user identity and permissions.",
            "Authentication token obtained through the login API. Required for accessing Instagram's unofficial API endpoints.",
            "Authentication token obtained via the login API. Required for accessing Instagram user stories.",
            "Instagram session authentication token for API access. Required for authenticated requests.",
        }
    ),
    "session_token": frozenset(
        {
            "The session token of the user",
        }
    ),
    "sessionid": frozenset(
        {
            "A valid session ID string obtained from the Login endpoint. This token authenticates the user and grants access to their private messages.",
            "Authentication session ID obtained from the login endpoint. Identifies the user's active session.",
            "Authentication token obtained from the Login endpoint to identify and authorize the wearer",
            "Authentication token obtained from the Login endpoint to validate the user's session.",
            "Authentication token obtained from the Login endpoint. This session ID validates the user's identity and permissions for accessing message data.",
            "Authentication token obtained from the login endpoint. This session identifier must be valid and active to access protected resources.",
            "User authentication session ID obtained from the Login endpoint. This identifies the active user session.",
        }
    ),
    "store_key": frozenset(
        {
            "API key uniquely identifying the store to query. This key authenticates access to the store's attribute data.",
            "Unique identifier for the store's API integration. This key authenticates the store's identity and grants access to its resources.",
        }
    ),
    "subscribe_key": frozenset(
        {
            "Unique PubNub subscription key used to authenticate and authorize message retrieval requests.",
        }
    ),
    "sucid": frozenset(
        {
            "The unique identifier for the user account (sucid) used to authenticate and track user activity",
        }
    ),
    "tok_proxy": frozenset(
        {
            "Authentication token for proxy service (if required for API access)",
            "Authentication token for the SMS service. Required if the service requires explicit authorization.",
            "Proxy token for API authentication. Required for authenticated requests.",
            "Proxy token for authenticated requests through intermediaries. Required for restricted access scenarios.",
        }
    ),
    "token": frozenset(
        {
            "A free token obtained by sending a WhatsApp message with the command 'get-token' to the number +34 631 428 039.",
            "A valid token obtained by sending a WhatsApp message with the command 'get-token' to the provided number.",
            "API access token for authentication. Tokens must be obtained through your account manager and maintained securely.",
            "API authentication token (contact account manager for access)",
            "API authentication token for accessing Twitter data. This token grants access to analyze user data according to Twitter's API terms of service.",
            "API authentication token for seller account access",
            "API authentication token. Required for authorized access to the geocoding service.",
            "API key for authentication",
            "API key for authentication.",
            "API token",
            "Access token with appropriate permissions to access group member data. If not provided, operations will rely on the session cookie for authorization.",
            "An existing authentication token to renew or validate. If omitted, a new token will be generated based on the provided credentials.",
            "Authentication token for API access",
            "Authentication token for API access, obtained through seller account credentials",
            "Authentication token for API access. If provided, takes precedence over the token in the Authorization header.",
            "Authentication token for an active user session. Must be a valid session identifier.",
            "Authentication token for secure API access",
            "Authentication token for the Ecombr API.",
            "Authentication token for the current user",
            "Authentication token for webhook activation, obtained through the subscription registration process",
            "Authentication token obtained from the WhatsApp API service. To acquire a free token, message 'get-token' to +34 631 42 80 39 using WhatsApp.",
            "Authentication token obtained through a prior check-user request. Required unless a password is provided. Takes precedence over password if both are provided.",
            "Authentication token obtained through check-user API. Takes precedence over password if both are provided.",
            "Authentication token obtained via check-user API. Valid until explicitly reset.",
            "Authentication token or session identifier required to access Kayak's API endpoints or retrieve specific search results",
            "Authentication token required to access Skyscanner's API endpoints. Must be obtained through prior API authentication.",
            "Authentication token required to activate the webhook. This token verifies the caller's identity and access permissions to manage webhook configurations.",
            "Authentication token used alongside the secret key for API access validation",
            "Authentication token used to validate the current session or API request. Typically formatted as a JWT (JSON Web Token) or API key string.",
            "Authentication token with required permissions to access follower data. Must be a valid API access token.",
            "Authentication token. Defaults to None.",
            "Free API token obtained via WhatsApp by sending 'get-token' to 34631428039. Visit https://wa.me/34631428039?text=get-token for instructions.",
            "OAuth2 bearer token for authenticated API access",
            "Optional API access token for additional authentication layers. If provided, must be a valid bearer token string.",
            "Telegram bot API token. Format: '123456:ABC-defGHIjkl-MNOPQRSTUvwyxz'",
            "Temporary access token for API authentication. Must be obtained through prior authorization flow.",
            "Temporary access token for authenticated API operations. Must be valid and unexpired.",
            "Temporary access token obtained through prior authentication, used to validate the seller's session.",
            "The API token for authentication. Defaults to 'TokenDemoRapidapi'.",
            "The authentication token required to validate and activate the webhook endpoint. This token must match the one provided during the initial subscription registration process.",
            "The token for API authentication. Defaults to 'TokenDemoRapidapi'.",
            "The token for validating the API request.",
            "The token of the user. A token can be obtained through check-user, and is valid until reset.",
            "The token that needs to be invalidated.",
            "Token for authentication with the API.",
            "Token sent in the email to confirm registration",
            "User profile session token granting access to Facebook posts. This token must be obtained through successful user authentication.",
            "User profile session token obtained through LoginRadius authentication flow. This token grants access to the user's Twitter mentions data.",
            "User session token obtained through successful authentication with a supported OAuth provider",
            "User session token obtained through successful social login authentication. This token identifies the user's active session for profile data retrieval.",
            "User session token representing the authenticated LinkedIn profile. Must be obtained through prior successful authentication flow.",
            "User-specific session token obtained through LoginRadius authentication. Must be a valid OAuth 2.0 access token string.",
            "User-specific session token obtained through LoginRadius authentication. This token grants temporary access to the user's Facebook group data and must be obtained prior to API calls.",
        }
    ),
    "twttr_session": frozenset(
        {
            "Authentication session token for Twitter API access. Format: alphanumeric string (e.g., 'session_token_abc123')",
            "Authentication session token for Twitter API access. Required for authorized requests.",
            "Authentication session token for user verification. This value must be obtained through prior login operations and maintained for subsequent API interactions.",
            "Authentication token obtained from Twitter login endpoint. Required for accessing restricted content",
            "Authentication token obtained from the login endpoint for accessing protected Twitter/X resources",
            "Authentication token obtained via Twitter's login endpoint. This token is required to access the API and must be generated prior to making requests.",
            "Authentication token or session identifier for Twitter API access. When provided, ensures authorized access to user data with reduced rate limiting. If not specified, defaults to an empty string which may result in restricted access.",
            "Authentication token or session identifier for accessing Twitter API resources. If not provided, the function will attempt to use default authentication mechanisms.",
            "Authentication token or session identifier for accessing protected user data. Must be obtained through prior authentication flows.",
            "Authentication token or session identifier for accessing the API. Required for all requests.",
            "Authentication token required for API access. Must be obtained via the login endpoint before use",
            "The Twitter session ID for authenticated requests. Defaults to None.",
            "The session token for authenticating with the Twitter API.",
            "Twitter API session token for authenticated requests. Required for accessing protected user endpoints. Format should be a valid session token string.",
            "Twitter API session token obtained from the login endpoint for authenticated requests",
            "Twitter API session token or bearer token for authentication. Required for accessing protected Twitter media content.",
            "Twitter API session token or cookies for authentication. Required for accessing protected tweets or when rate limits require authentication. If not provided, request is made without authentication.",
            "Twitter authentication session token (e.g., bearer token) for authorizing the API request.",
            "Twitter session authentication token required for API access. This token must be generated through Twitter's OAuth authentication flow.",
            "Valid Twitter API session token with read access. Must be generated through Twitter's OAuth authentication flow.",
        }
    ),
    "user": frozenset(
        {
            "Authentication token for accessing the Lemlist API. This is a unique string identifier for the user account, typically formatted as a hexadecimal string.",
        }
    ),
    "user_credentials": frozenset(
        {
            "Username and password for authentication",
        }
    ),
    "user_token": frozenset(
        {
            "Optional user authentication token for existing accounts",
        }
    ),
    "userid": frozenset(
        {
            "The user's authentication identifier with access permissions to the requested attachment. Must match the user associated with the monetary account.",
            "Unique identifier for the bunq user account associated with the monetary account. This ID is required to authenticate and scope the request to the correct user.",
            "Unique identifier of the user associated with the cash register. This ensures proper access control and record ownership verification.",
            "User account identifier for authentication with the SMS service",
            "Your unique user identifier for authentication and access control",
        }
    ),
    "username": frozenset(
        {
            "Hajana One account username used for API authentication.",
            "SensSMS account username for request authentication.",
            "The Instagram account username used for authentication",
            "The user's account name or email address used for authentication. Expected to follow standard account naming conventions.",
            "The user's account name used for authentication.",
            "The user's unique identifier for authentication. This typically represents the user's account name or email address.",
            "The user's unique identifier used for authentication. Must be a string.",
            "The username for the MySQL server authentication.",
            "The username for the PostgreSQL server authentication.",
            "The username to use for authentication",
        }
    ),
    "wainstanceidinstance": frozenset(
        {
            "Unique identifier for the WhatsApp instance. This ID is required to authenticate and associate the QR code with a specific WhatsApp business account.",
        }
    ),
    "wsapikey": frozenset(
        {
            "Your Walk Score API Key",
        }
    ),
    "x_api_key": frozenset(
        {
            "API authentication key for accessing the crypto market data service. Authenticates the caller and determines access permissions. While technically optional, providing a valid API key is strongly recommended for full functionality and higher rate limits.",
            "API authentication token with catalog access permissions",
            "API key for authenticating requests to the crypto market data service. This key grants access to blockchain and market data endpoints.",
            "API key for authenticating the request to access Bitcoin blockchain data. Must be provided in the request headers.",
            "API key for authenticating the request. Must be obtained from the service provider.",
            "API key for authenticating the request. This key must be obtained from the service provider and included in all API calls.",
            "API key for authenticating with the service",
            "API key for client identification and rate limiting purposes. Must be a valid key provisioned through the API gateway.",
            "API key for the Dream Diffusion service.",
            "Application-specific API key for service authentication. This key identifies the calling application to the API.",
            "Authentication API key for accessing the cryptocurrency exchange data API. This key grants access to market data and must be included in all requests.",
            "Authentication API key for accessing the service. This key must be obtained from the provider and maintained securely.",
            "Authentication API key provided by the blockchain data service provider. Must be kept confidential and included in all requests for access authorization.",
            "Authentication credential for API access. Must be a valid API key issued by the service provider.",
            "Authentication credential for accessing the cryptocurrency market data API. Must be a valid API key string obtained from the service provider.",
            "Authentication credential required to access the API. Obtain this key from your service provider or API dashboard",
            "Authentication key for accessing the API, obtained through service provider registration",
            "Authentication token required to access the cryptocurrency market data API. Must be included in request headers as 'x-api-key'.",
            "Authentication token required to access the cryptocurrency market data API. This unique identifier must be obtained from the service provider.",
            "Authentication token required to access the exchange rate API",
        }
    ),
    "x_funtranslations_api_secret": frozenset(
        {
            "API Key for accessing the FunTranslations Old English Translator.",
            "API key for the FunTranslations service. Obtain one at http://funtranslations.com/api/shakespeare.",
        }
    ),
    "x_hojininfo_api_token": frozenset(
        {
            "API access token for authentication with the Japan Patent Office data service. Must be obtained through the service provider's authorization process.",
            "Authentication token required to access the company database API. Must be obtained through the service's authorization system.",
        }
    ),
    "x_mashape_key": frozenset(
        {
            "API authentication key for accessing financial data services",
        }
    ),
    "x_rapidapi_key": frozenset(
        {
            "API authentication key required for accessing the soccer data service. This unique identifier ensures authorized access to the API endpoint.",
            "Authentication key for accessing Nakho API services. Obtain from https://docs.nakho.com/api/coupons for access.",
            "Authentication key for accessing the RapidAPI service",
            "RapidAPI access key for authentication. Must be obtained from RapidAPI dashboard.",
            "RapidAPI authentication key for accessing the tradingradar service. A valid key is required for successful API calls.",
            "RapidAPI key for authenticating with the photo service API",
            "The RapidAPI key for accessing the `data_visualisation_` API. Defaults to 'demo'.",
            "The RapidAPI key provided for API requests.",
            "Your RapidAPI key for authenticating requests to the Gymshark API",
        }
    ),
    "x_rapidapi_proxy_secret": frozenset(
        {
            "The RapidAPI proxy secret key for authenticating requests to the backend service. Required if using a RapidAPI proxy setup. Leave empty if authentication is not required.",
        }
    ),
    "x_rapidapi_user": frozenset(
        {
            "The RapidAPI username associated with the account. This header is used to authenticate requests and create a unique identifier in the database.",
        }
    ),
    "x_xsrf_token": frozenset(
        {
            "Cross-site request forgery protection token. Must be included in requests to prevent unauthorized actions.",
            "Security token to prevent cross-site request forgery attacks. Must be obtained from a prior authentication response.",
            "Security token to prevent cross-site request forgery attacks. Must be obtained through prior authentication handshake.",
        }
    ),
    "xbc": frozenset(
        {
            "BC token hash stored in client-side localStorage (localStorage.bcTokenSha).",
            "Session identifier token for maintaining connection state",
            "XBC token for API access.",
        }
    ),
    "xid": frozenset(
        {
            "Unique identifier for the user or device associated with the sleep data. This ID must be obtained through prior authentication or device registration and is used to authenticate and route the request to the correct data source.",
        }
    ),
}

_NO_DESCRIPTION_FUNCTIONS = frozenset(
    {
        (
            "API_KEY",
            "biggest_cities",
            "Fetches the biggest cities' data from the specified API using the provided ID and API key.",
        ),
        (
            "Authorization",
            "validate_access_using_basic_authentication",
            "Validates user access by verifying Basic Authentication credentials provided in the request headers. This function checks for the presence of an Authorization header containing valid Basic Authentication credentials (username and password) and authenticates the user against the system. Use this function to secure endpoints requiring user authentication via Basic Auth.",
        ),
        (
            "api_key",
            "address",
            "Sends a request to an address correction and geocoding API using the provided address lines and API key.",
        ),
        (
            "api_key",
            "amazon_product_details",
            "Retrieves detailed product information from Amazon including price, description, specifications, and availability. Requires an API key from ScraperAPI (registration provides 5,000 free request points).",
        ),
        (
            "api_key",
            "filter_cook_time_in_minutes",
            "Fetches keto recipes within a specified range of cooking times using the provided API key.",
        ),
        (
            "api_key",
            "filter_group_giveaways",
            "Filters and groups giveaways based on the provided platform and type, using the provided API key to make the request.",
        ),
        (
            "api_key",
            "full_data",
            "Fetches data from the RapidAPI based on the provided keyword, optional full data, and API key parameters.",
        ),
        (
            "api_key",
            "generate_cc_number",
            "Generates a fake credit card number using the specified brand and API key.",
        ),
        (
            "api_key",
            "generate_recipe",
            "This function generates a recipe using an ingredient and a provided RapidAPI key.",
        ),
        (
            "api_key",
            "get_answer_to_question",
            "Fetches the answer to a given question from the Question Answered API using the provided RapidAPI key.",
        ),
        (
            "api_key",
            "get_cci_by_number",
            "Fetches the CCI (Control Correlation Identifier) and its definition based on a 6-digit CCI identification number using the specified RapidAPI key.",
        ),
        (
            "api_key",
            "get_individual_bank",
            "Fetches exchange rates from a specified central bank using a given bank ID and API key.",
        ),
        (
            "api_key",
            "geteventtypes",
            "Fetches event types data from the specified API using the provided skin name and RapidAPI key.",
        ),
        (
            "api_key",
            "image_generation_endpoint",
            "Generates an image using the Bruzu API with the specified width, height, and background image URL. The API key for the Toolbench RapidAPI is also required to authenticate the request.",
        ),
        (
            "api_key",
            "ip_history",
            "Retrieves torrent history data for a specified IP address. This function provides access to peer history information including shared content over a specified time period. Returns torrent activity history for the specified IP address. (Beta version) The API key must be provided separately for authentication.",
        ),
        (
            "api_key",
            "list_of_room_types",
            "Returns data related to room types from the Booking.com API. This function optionally takes a room type ID and an API key to fetch specific room type details.",
        ),
        (
            "api_key",
            "manga",
            "Fetches manga information from a specified API using pagination and a provided API key.",
        ),
        (
            "api_key",
            "profile_highlights",
            "Fetches the Instagram profile highlights for a given username using the specified RapidAPI key.",
        ),
        (
            "api_key",
            "retrieve_file",
            "Retrieves a file from the server using the provided file path and API key.",
        ),
        (
            "api_key",
            "revise",
            "Revise and correct the given text using a specified content type and API key.",
        ),
        (
            "api_key",
            "search_on_ebay",
            "Searches for items on eBay using the provided search query and RapidAPI key.",
        ),
        (
            "api_key",
            "search_twitter",
            "Search Twitter based on the specified type, count, and query criteria, with optional cursor for pagination and API key for authorization.",
        ),
        (
            "api_key",
            "settag",
            "Sets a new tag for the SMS receive service using the provided RapidAPI key.",
        ),
        (
            "api_key",
            "stock_get_fund_profile",
            "Fetch the fund profile information for a given stock using the provided ticker ID and API key.",
        ),
        (
            "api_key",
            "test_app_deepfry_get",
            "Sends a GET request to the deepfry endpoint of the RapidAPI Image Processing service with the specified URL and API key.",
        ),
        (
            "api_key",
            "testing_conncetion",
            "Verifies connectivity and authentication status between the client and API endpoint. Use this function to confirm service availability and validate credentials prior to making other API calls.",
        ),
        (
            "apikey",
            "amazon_product_details",
            "Retrieves detailed product information from Amazon including price, description, specifications, and availability. Requires an API key from ScraperAPI (registration provides 5,000 free request points).",
        ),
        (
            "apikey",
            "get_biden_articles_from_specific_newspaper",
            "Fetches all articles related to Biden from a specified newspaper using a given API key.",
        ),
        (
            "apikey",
            "get_game_details",
            "Fetches the basic information about a game using the provided game ID and RapidAPI key.",
        ),
        (
            "apikey",
            "get_individual_bank",
            "Fetches exchange rates from a specified central bank using a given bank ID and API key.",
        ),
        (
            "key",
            "banks_by_country",
            "Fetches a list of banks for a specified country using the given RapidAPI key.",
        ),
        (
            "key",
            "check",
            "Checks the details of a bank card BIN (Bank Identification Number) using the provided BIN number and RapidAPI key.",
        ),
        (
            "key",
            "mensajes_1",
            "Fetches messages for student 1 from the Colegio Santa Ana API using the provided authorization token and API key.",
        ),
        (
            "key",
            "oldsort",
            "Fetches and sorts data from the oldsort endpoint using the given RapidAPI host and key.",
        ),
        (
            "key",
            "retrieve_file",
            "Retrieves a file from the server using the provided file path and API key.",
        ),
        (
            "password",
            "testing_conncetion",
            "Verifies connectivity and authentication status between the client and API endpoint. Use this function to confirm service availability and validate credentials prior to making other API calls.",
        ),
        (
            "password",
            "validate_access_using_basic_authentication",
            "Validates user access by verifying Basic Authentication credentials provided in the request headers. This function checks for the presence of an Authorization header containing valid Basic Authentication credentials (username and password) and authenticates the user against the system. Use this function to secure endpoints requiring user authentication via Basic Auth.",
        ),
        (
            "session_token",
            "login",
            "Authenticates a user session and grants access to protected social media functionality. This function should be called before performing any user-specific operations to establish an authorized session context.",
        ),
        (
            "token",
            "authentication_status",
            "Verifies the validity of an authentication token provided in request headers. Returns HTTP 200 status code along with API version and associated token claims when authentication is successful. Use this endpoint to validate credentials and inspect granted permissions before making protected API calls.",
        ),
    }
)

_PUBLIC_ACCESS_CONSTANTS = frozenset(
    {
        (
            "x_rapidapi_key",
            "The RapidAPI key for accessing the `data_visualisation_` API. Defaults to 'demo'.",
            "demo",
        ),
        (
            "api_key",
            "API authentication key. Use 'test' for limited access (rate-limited) or obtain a premium key from https://ipdata.co/ for production use.",
            "test",
        ),
        (
            "api_key",
            "Authentication key for API access. Use 'test' (default) for limited access, or a personal API key from https://ipdata.co/ for production use",
            "test",
        ),
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
    if (
        isinstance(description, str)
        and isinstance(value, str)
        and (field, description, value) in _PUBLIC_ACCESS_CONSTANTS
    ):
        return False
    if isinstance(description, str) and description in _EXISTING_DESCRIPTIONS.get(
        field, ()
    ):
        return True
    if (
        description is None
        and isinstance(definition.get("description"), str)
        and (field, name, definition.get("description")) in _NO_DESCRIPTION_FUNCTIONS
    ):
        return True
    # These names and slots were independently audited before the complete
    # descriptor census, including source variants with no parameter description.
    return field == "api_key" and name in {
        "get_amazon_product_details",
        "get_amazon_search_results",
    }
