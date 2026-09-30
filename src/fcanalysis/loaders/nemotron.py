"""Pinned Nemotron v1/v2 source adapters; universal mechanics live in stages.

The release has explicit OpenAI-style roles, tools and call batches. v1's
anonymous tool-calling results cannot prove multi-call pairing; singleton
results are unambiguous. v2 and v1 interactive use complete source IDs. No
upstream row scores, UUIDs, domain labels or simulator metadata grant context.
"""

from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator
from copy import deepcopy
from functools import partial
import re
from typing import Any

from ..format import ConversationSample
from .base import FilterConfig, LoadReport
from .context import contains_literal_token, iter_call_evidence, json_scalar_values
from .curation import CurationConfig, CurationInput, CurationScope, curate
from .nemotron_context import validate_async
from .nemotron_credentials import credential_kind, visible_spelled_digits
from .nemotron_temporal import validate_temporal_inputs
from .normalization import Reject, normalize_tools, parse_json, serialize_result
from .pipeline import (
    Pipeline,
    RowState,
    Stage,
    link_calls,
    override_system,
    reconcile_definitions,
    remove_reasoning,
    validate_capabilities,
    validate_structure,
    validate_termination,
)

_FUNCTION_FIELDS = {"name", "description", "parameters", "strict", "required"}
_ROLE_FIELDS = {
    "system": {"role", "content"},
    "user": {"role", "content"},
    "assistant": {
        "role",
        "content",
        "tool_calls",
        "reasoning_content",
        "function_call",
    },
    "tool": {"role", "content", "tool_call_id", "name"},
}
# Exact credential/state slots inspected in the released definitions. Method
# selectors and free-form newly generated values are deliberately not included.
_PROTECTED = frozenset(
    {
        "verification_code",
        "verificationCode",
        "security_code",
        "otp",
        "otp_code",
        "sms_verification_code",
        "mobile_verification_code",
        "two_fa_code",
        "verification_pin",
        "security_pin",
        "x_apikey",
        "api_key",
        "apikey",
        "apiKey",
        "auth_token",
        "access_token",
        "session_token",
        "security_token",
        "biometric_token",
        "two_factor_token",
        "dual_factor_token",
        "verification_token",
        "acceptance_token",
        "event_confirmation_code",
        "reservation_confirmation_code",
    }
)
_AUTH_FUNCTIONS = frozenset(
    {
        "authenticate",
        "authenticate_user",
        "authenticate_customer",
        "authenticate_client",
        "authenticate_patient",
        "authenticate_parent",
        "authenticate_pet_owner",
        "authenticate_member",
        "authenticate_congregant",
        "authenticate_guest",
        "authenticate_student",
        "authenticate_rider",
        "authenticate_visitor",
        "authenticate_influencer",
        "authenticate_bidder",
        "authenticate_user_by_id_code",
        "authenticate_user_by_email",
        "authenticate_user_registered",
        "authenticate_corporate_account",
        "verify_identity",
        "verify_user",
        "verify_user_identity",
        "verify_customer_identity",
        "verify_client_identity",
        "verify_contact_code",
        "verify_code",
        "verify_authentication_code",
        "verifyAuthenticationCode",
        "validate_authentication_code",
        "validate_verification_code",
        "verify_auth_code",
        "verify_user_code",
        "verify_email_code",
        "verify_email_otp",
        "verify_mfa",
        "check_verification_status",
        "check_verification_code",
        "message_send",
        "send_sms",
        "login_copy",
        "loginuser",
        "sendsms_php",
        "get_delivery_status",
        "single_sms_api",
        "submit_code_telegram_submitcode_get",
        "form_check_user",
        "form_check_instance",
        "check_gmail",
        "datavare_gmail_backup_converter",
        "account_balance",
        "user_login",
        "form_chat",
        "get_your_account_balance",
        "gettoken",
        "credentials_id",
        "login",
    }
)
_AUTH_VALUES = frozenset(
    {
        "credentials",
        "credential",
        "authentication_data",
        "verification_data",
        "verification_value",
        "auth_value",
        "confirmation_code",
        "confirmation_number",
        "booking_confirmation",
        "order_confirmation",
        "security_answer",
        "password",
        "pin",
    }
)
_SELECTORS = frozenset(
    {
        "method",
        "auth_method",
        "authentication_method",
        "verification_method",
        "verification_type",
    }
)


def convert_row(
    raw: Any,
    *,
    dataset: str,
    split: str,
    line: int,
    version: int,
    transforms: Counter[str] | None = None,
) -> ConversationSample:
    """Copy known model-visible fields, preserving absence, nulls and strings."""
    if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list):
        raise Reject("malformed_source_row")
    out = []
    for source in raw["messages"]:
        if not isinstance(source, dict):
            raise Reject("malformed_message")
        role = source.get("role")
        if role not in _ROLE_FIELDS:
            raise Reject("unknown_role")
        if source.keys() - _ROLE_FIELDS[role]:
            raise Reject("unknown_message_field")
        msg = deepcopy(source)
        if role == "assistant":
            if "function_call" in msg:
                # Full pinned search census: this legacy response slot is null.
                # A non-null value is not silently discarded or preferred.
                if (
                    version != 2
                    or split != "search"
                    or msg["function_call"] is not None
                ):
                    raise Reject("unresolved_legacy_function_call")
                del msg["function_call"]
                if transforms is not None:
                    transforms["null_legacy_function_call_fields_removed"] += 1
            calls = msg.get("tool_calls")
            if calls is None:
                if "tool_calls" in msg:
                    del msg["tool_calls"]
            elif not isinstance(calls, list):
                raise Reject("malformed_tool_calls")
            else:
                for call in calls:
                    if not isinstance(call, dict):
                        raise Reject("malformed_tool_call")
                    if "index" in call:
                        # Stream assembly position is protocol metadata, not content.
                        if type(call["index"]) is not int or call["index"] < (
                            -1 if version == 2 else 0
                        ):
                            raise Reject("invalid_source_call_index")
                        del call["index"]
                        if transforms is not None:
                            transforms["source_call_index_fields_removed"] += 1
        elif role == "tool":
            if "content" not in msg:
                raise Reject("missing_tool_content")
            if not isinstance(msg["content"], str) and transforms is not None:
                transforms["structured_tool_results_serialized"] += 1
            msg["content"] = serialize_result(msg["content"])
        out.append(msg)
    tools = normalize_tools(raw.get("tools", []))
    for tool in tools:
        if tool.keys() - {"type", "function"}:
            raise Reject("unknown_tool_field")
        function = tool["function"]
        if function.keys() - _FUNCTION_FIELDS:
            raise Reject("unknown_function_definition_field")
        if "required" not in function:
            continue
        required = function.pop("required")
        if not isinstance(required, list) or any(
            not isinstance(x, str) for x in required
        ):
            raise Reject("invalid_legacy_required")
        if required:
            parameters = function.get("parameters", {})
            if not isinstance(parameters, dict):
                raise Reject("invalid_legacy_required")
            prior = parameters.get("required", [])
            if not isinstance(prior, list) or any(
                not isinstance(x, str) for x in prior
            ):
                raise Reject("invalid_legacy_required")
            parameters["required"] = prior + [x for x in required if x not in prior]
            function["parameters"] = parameters
        if transforms is not None:
            transforms[
                "legacy_function_required_lifted"
                if required
                else "empty_legacy_function_required_removed"
            ] += 1
    metadata = raw.get("metadata")
    uuid = metadata.get("uuid") if isinstance(metadata, dict) else None
    uuid = uuid or raw.get("uuid")
    sample_id = (
        f"{split}_{uuid}" if isinstance(uuid, str) and uuid else f"{split}_line{line}"
    )
    return ConversationSample(
        messages=out,
        tools=tools,
        dataset=f"{dataset}/{split}",
        sample_id=sample_id,
        raw=raw,
    )


def source_linkage(
    state: RowState, *, anonymous: bool, parallel: bool, align_results: bool = True
) -> None:
    seen_ids: set[str] = set()
    for message in state.sample.messages:
        for call in message.get("tool_calls", []):
            source_id = call.get("id")
            if source_id is not None:
                if (
                    not isinstance(source_id, str)
                    or not source_id
                    or source_id in seen_ids
                ):
                    raise Reject("invalid_source_tool_linkage")
                seen_ids.add(source_id)
    if anonymous:
        # The audited v1 tool-calling file supplies unreferenced call IDs but no
        # result IDs or names. We may bind one call to one immediate result;
        # the release does not establish an ordering contract for two or more.
        for msg in state.sample.messages:
            if msg["role"] == "tool" and ("tool_call_id" in msg or "name" in msg):
                raise Reject("unexpected_v1_anonymous_result_linkage")
            calls = msg.get("tool_calls", [])
            if len(calls) > 1:
                raise Reject("ambiguous_source_tool_linkage")
            for call in calls:
                if "id" in call:
                    if not isinstance(call["id"], str) or not call["id"]:
                        raise Reject("invalid_source_tool_linkage")
                    del call["id"]
                    state.transforms["source_linkage_fields_removed"] += 1
    link_calls(state, positional=False, parallel=parallel, align_results=align_results)


def _leaves(value: Any) -> Iterator[str]:
    yield from json_scalar_values(value, skip_keys=_SELECTORS)


_LAST4_FIELDS = frozenset(
    {
        "ssn_last4",
        "ssn_last_4",
        "last_4_ssn",
        "phone_last4",
        "phone_last_four",
        "phone_last_4",
        "last_four_phone",
        "last_4_phone",
        "tax_id_last4",
        "last_four_card",
    }
)
_LAST4_METHODS = {
    ("authenticate_member", "verification_code"): (
        "verification_method",
        "phone_digits",
    ),
    ("authenticate_client", "verification_value"): (
        "verification_method",
        "phone_digits",
    ),
    ("authenticate_customer", "verification_value"): (
        "verification_method",
        "ssn_last4",
    ),
    ("verify_identity", "verification_value"): ("verification_type", "phone_last4"),
    ("verify_identity", "verification_code"): ("verification_type", "ssn_last4"),
    ("authenticate_guest", "auth_value"): ("auth_method", "phone"),
    ("verify_user_identity", "verification_pin"): ("method", "id_pin"),
}
_PHONE_PIN_DESCRIPTION = (
    "Verify user identity using one of the supported methods: 1. User ID + PIN "
    "(last 4 digits of phone), 2. Full name + email, 3. Government ID (Verified "
    "Plus/Premium only). Required before account actions."
)


def _last4_method(
    name: str, field: str, definition: dict[str, Any], args: dict[str, Any]
) -> bool:
    selected = _LAST4_METHODS.get((name, field))
    if selected is None or args.get(selected[0]) != selected[1]:
        return False
    parameters = definition.get("parameters")
    properties = (
        parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    )
    schema = properties.get(selected[0], {}) if isinstance(properties, dict) else {}
    if not isinstance(schema, dict) or selected[1] not in schema.get("enum", []):
        return False
    if name == "verify_user_identity":
        return definition.get("description") == _PHONE_PIN_DESCRIPTION
    if name == "authenticate_guest":
        return (
            properties.get(field, {}).get("description")
            == "Booking ID, email, or last 4 phone digits"
        )
    return True


def _credential_values(
    value: Any, *, last4: bool = False
) -> Iterator[tuple[str, bool]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in _SELECTORS:
                yield from _credential_values(child, last4=key in _LAST4_FIELDS)
    elif isinstance(value, list):
        for child in value:
            yield from _credential_values(child, last4=last4)
    else:
        for leaf in json_scalar_values(value):
            yield leaf, last4


def _visible_last4(value: str, texts: list[str]) -> bool:
    """Audited last-four slots may use a visible formatted identifier suffix."""
    if re.fullmatch(r"\d{4}", value) is None:
        return False
    pattern = (
        r"(?<!\w)(?:[Xx*#]+(?:-[Xx*#]+)*|\d{3}-\d{2}|"
        r"(?:\+?\d{1,3}[- .]?)?(?:\(?\d{3}\)?[- .])?\d{3})[- .]"
        + re.escape(value)
        + r"(?!\w|[.,]\d)"
    )
    return any(re.search(pattern, text) for text in texts)


# These are exact existing-state input descriptions found in the pinned
# definitions, not keyword matches. In particular a freshly selected OTP,
# password, or newly generated access token does not match this contract.
_EXISTING_CREDENTIAL_DESCRIPTIONS = frozenset(
    {
        "API-KEY required to access this information.",
        "Access token obtained from authentication",
        "Active authentication token",
        "Authenticated session token",
        "Valid access token",
        "Authentication token",
        "Authentication token from verification",
        "Authentication token from initial verification",
        "Active session token for extended interactions",
        "Active session token from authentication",
        "Authentication token from authenticate_admin",
        "Authentication token from authentication",
        "Authentication token from verify_authentication_code",
        "Current session token",
        "Token from request_verification_code",
        "Token from surge acceptance",
        "Token from age verification",
        "Token from successful identity verification",
        "Token from validate_verification_code",
        "Token from verify_recipient_identity",
        "Identity verification token from verify_authentication_code",
        "Identity verification token from verify_player_identity",
    }
)
_VIRUSTOTAL_TOOLS = frozenset(
    {
        "vt_add_comment_to_ip_address",
        "vt_add_votes_to_ip_address",
        "vt_get_comments_on_domain",
        "vt_get_comments_on_ip_address",
        "vt_get_dns_resolution_object",
        "vt_get_domain_report",
        "vt_get_ip_address_report",
        "vt_get_object_descriptors_related_to_domain",
        "vt_get_object_descriptors_related_to_ip_address",
        "vt_get_objects_related_to_domain",
        "vt_get_objects_related_to_ip_address",
    }
)


def _requires_credential(name: str, field: str, definition: dict[str, Any]) -> bool:
    if credential_kind(name, field, definition) is not None:
        return True
    if name in _AUTH_FUNCTIONS and field in (_AUTH_VALUES | _PROTECTED):
        return True
    parameters = definition.get("parameters")
    properties = (
        parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    )
    schema = properties.get(field) if isinstance(properties, dict) else None
    description = schema.get("description") if isinstance(schema, dict) else None
    if (
        name in _VIRUSTOTAL_TOOLS
        and field == "x_apikey"
        and description == "Your API key"
    ):
        return True
    return field in _PROTECTED and description in _EXISTING_CREDENTIAL_DESCRIPTIONS


def validate_visible_inputs(state: RowState) -> None:
    """Bounded exact evidence gate for audited credential slots.

    Previous user/system text and linked tool-result scalar leaves are evidence.
    Assistant prose/reasoning, definition examples, raw metadata and future
    results never supply a credential. This gate is not a proof of arbitrary
    policy compliance, natural-language confirmations, or calendar arithmetic.
    Native reasoning removal cannot change these evidence channels; system
    overrides can, so the full gate runs again on the final visible view.
    """
    for name, definition, args, text, results in iter_call_evidence(
        state, result_values=lambda parsed, message: json_scalar_values(parsed)
    ):
        for field, value in args.items():
            if not _requires_credential(name, field, definition):
                continue
            kind = credential_kind(name, field, definition)
            for leaf, last4 in _credential_values(
                value, last4=_last4_method(name, field, definition, args)
            ):
                if leaf in results or contains_literal_token(
                    leaf, text, strict_numeric_tokens=True
                ):
                    continue
                if kind == "bearer" and leaf.startswith("Bearer "):
                    token = leaf.removeprefix("Bearer ")
                    if token in results or contains_literal_token(
                        token, text, strict_numeric_tokens=True
                    ):
                        continue
                if kind == "digits" and visible_spelled_digits(
                    leaf, text + list(results)
                ):
                    continue
                if not (
                    name in _AUTH_FUNCTIONS and last4 and _visible_last4(leaf, text)
                ):
                    raise Reject("ungrounded_credential")


def pipeline(filters: FilterConfig, *, anonymous: bool, parallel: bool) -> Pipeline:
    stages: list[tuple[str, Stage]] = [
        ("definitions", reconcile_definitions),
        ("structure", validate_structure),
        (
            "linkage",
            partial(
                source_linkage,
                anonymous=anonymous,
                parallel=parallel,
                align_results=filters.align_results,
            ),
        ),
        ("capabilities", validate_capabilities),
        ("visible_inputs", validate_visible_inputs),
        ("temporal_inputs", validate_temporal_inputs),
        ("async_lifecycle", validate_async),
    ]
    if filters.strip_thinking:
        stages.append(("reasoning", remove_reasoning))
    if filters.system_message_override is not None:
        stages.append(
            (
                "system_override",
                partial(override_system, override=filters.system_message_override),
            )
        )
    # Only native assistant content/fields and system positions changed. No
    # call/result value/order changed after source binding: the final positional
    # pass rechecks cardinality and rebuilds shifted coordinates from that proof.
    stages.extend(
        [
            ("final_structure", validate_structure),
            (
                "final_linkage",
                partial(
                    link_calls,
                    positional=True,
                    parallel=parallel,
                    align_results=filters.align_results,
                ),
            ),
            ("final_capabilities", validate_capabilities),
            ("final_visible_inputs", validate_visible_inputs),
            ("final_temporal_inputs", validate_temporal_inputs),
            ("final_async_lifecycle", validate_async),
            ("termination", validate_termination),
        ]
    )
    return Pipeline(stages)


def iter_dataset(
    *,
    dataset: str,
    revision: str,
    files: dict[str, str],
    splits: tuple[str, ...],
    version: int,
    lines: Callable[[str], Iterable[str]],
    filters: FilterConfig,
    curation: CurationConfig,
    raw_select: Callable[[str, Any], str | None] | None = None,
) -> tuple[Generator[ConversationSample, None, None], LoadReport]:
    if len(set(splits)) != len(splits) or any(x not in files for x in splits):
        raise ValueError(f"splits must be distinct members of {tuple(files)}")
    if filters.system_message_override is not None and not isinstance(
        filters.system_message_override, str
    ):
        raise TypeError("system_message_override must be a string or None")
    report = LoadReport(
        dataset=dataset,
        raw_count=0,
        stage1_count=0,
        filter_config=filters,
        strip_thinking_applied=filters.strip_thinking,
    )
    source: dict[str, Any] = {
        "revision": revision,
        "files": [files[s] for s in splits],
        "config": "default",
        "subsets": {},
    }
    report.dataset_config_transform_counts["source"] = source
    report.dataset_config_transform_counts["pipeline_by_subset"] = {}
    report.dataset_config_transform_counts["curation_by_subset"] = {}

    def output() -> Generator[ConversationSample, None, None]:
        total = valid = 0
        for split in splits:
            stages = pipeline(
                filters,
                anonymous=version == 1 and split == "tool_calling",
                parallel=False,
            )
            counts: Counter[str] = Counter()
            transforms: Counter[str] = Counter()
            conversion_drops: Counter[str] = Counter()
            selection_drops: Counter[str] = Counter()

            def inputs() -> Iterator[CurationInput]:
                nonlocal valid
                for line_number, encoded in enumerate(lines(split), 1):
                    counts["physical"] += 1
                    report.raw_count += 1
                    try:
                        raw = parse_json(encoded)
                        counts["parseable"] += 1
                        reason = (
                            raw_select(split, raw) if raw_select is not None else None
                        )
                        if reason is not None:
                            counts["source_excluded"] += 1
                            selection_drops[reason] += 1
                            continue
                        sample = convert_row(
                            raw,
                            dataset=dataset,
                            split=split,
                            line=line_number,
                            version=version,
                            transforms=transforms,
                        )
                    except Reject as exc:
                        conversion_drops[exc.reason] += 1
                        continue
                    counts["converted"] += 1
                    report.stage1_count += 1
                    state = stages.process(sample)
                    if state is not None:
                        valid += 1
                        counts["validated"] += 1
                        yield CurationInput(
                            state.sample, state.batches, state.parsed_arguments
                        )
                report.stage1_drop_reasons.update(
                    Counter(report.stage1_drop_reasons) + conversion_drops
                )
                report.dataset_config_drop_reasons.update(
                    Counter(report.dataset_config_drop_reasons) + selection_drops
                )
                report.filter_drop_reasons.update(
                    Counter(report.filter_drop_reasons) + stages.drops
                )
                source["subsets"][split] = dict(counts)
                report.dataset_config_transform_counts["pipeline_by_subset"][split] = {
                    "passed": dict(stages.passed),
                    "stage_drops": dict(stages.stage_drops),
                    "drops": dict(stages.drops),
                    "conversion_drops": dict(conversion_drops),
                    "transformations": dict(transforms + stages.transforms),
                }

            result = curate(
                inputs(),
                scope=CurationScope(f"{dataset}/{split}", split, split),
                config=curation,
            )
            try:
                for sample in result:
                    total += 1
                    yield sample
                report.dataset_config_transform_counts["curation_by_subset"][split] = [
                    row.as_dict() for row in result.reports.values()
                ]
            finally:
                result.close()
        report.dataset_config_count = report.stage1_count
        report.filtered_count = valid
        report.final_count = total

    return output(), report
