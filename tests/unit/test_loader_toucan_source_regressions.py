"""Released source regressions from pinned Qwen3 train-00023-of-00044.parquet.

These are exact uuid/subset_name/messages/available_tools projections of the
three affected source rows. Audit labels and server metadata are not model
context and are unnecessary to reproduce the missing-clock defect.
"""

import pytest

from fcanalysis.loaders import toucan
from fcanalysis.loaders.base import FilterConfig


RAW_DATE_ROWS = [
    {
        "uuid": "04bd5f16-a8b7-5899-9f30-cc5d9bbc113e",
        "subset_name": "single-turn-original",
        "messages": '[{"role": "user", "content": "I’m planning a trip from Beijing to '
        "Shanghai next Monday and need to find out if there are any available "
        'train tickets on that day."}, {"role": "assistant", "content": "", '
        '"function_call": {"name": "12306-mcp-server-search", "arguments": '
        '"{\\"date\\": \\"2023-10-02\\", \\"fromCity\\": \\"Beijing\\", '
        '\\"toCity\\": \\"Shanghai\\"}"}}, {"role": "function", "content": '
        '"日期需为0~15天内", "name": "12306-mcp-server-search"}, {"role": "assistant", '
        '"content": "The selected date is beyond the 15-day limit for train ticket '
        "inquiries. Please choose a date within the next 15 days. Let me know if "
        "you'd like to check for tickets on an alternative date!\"}]",
        "available_tools": '[{"type": "function", "function": {"name": '
        '"12306-mcp-server-search", "description": "查询12306火车票", '
        '"parameters": {"type": "object", "properties": {"date": {"type": '
        '"string", "format": "date", "description": "出发日期 格式：YYYY-MM-DD"}, '
        '"fromCity": {"type": "string", "description": "出发城市"}, "toCity": '
        '{"type": "string", "description": "到达城市"}}, "required": ["date", '
        '"fromCity", "toCity"], "additionalProperties": false, "$schema": '
        '"http://json-schema.org/draft-07/schema#"}}}]',
    },
    {
        "uuid": "f875e001-473d-5533-bf53-6d13a6bb6f4b",
        "subset_name": "single-turn-original",
        "messages": '[{"role": "user", "content": "I need to find all available high-speed '
        "trains from Shanghai Hongqiao to Beijing West on next Tuesday afternoon. "
        "Can you help me search for the schedules and seat availability?\\n\\nYou "
        "need to solve this question using search tool from the list of available "
        'tools."}, {"role": "assistant", "content": "", "function_call": {"name": '
        '"12306-mcp-server-search", "arguments": "{\\"date\\": \\"2023-11-07\\", '
        '\\"fromCity\\": \\"上海虹桥\\", \\"toCity\\": \\"北京西\\"}"}}, {"role": '
        '"function", "content": "日期需为0~15天内", "name": "12306-mcp-server-search"}, '
        '{"role": "assistant", "content": "The date you selected is beyond the '
        "15-day limit for ticket searches. Please choose a date within the next 15 "
        "days, and I will help you find the available high-speed trains from "
        'Shanghai Hongqiao to Beijing West."}]',
        "available_tools": '[{"type": "function", "function": {"name": '
        '"12306-mcp-server-search", "description": "查询12306火车票", '
        '"parameters": {"type": "object", "properties": {"date": {"type": '
        '"string", "format": "date", "description": "出发日期 格式：YYYY-MM-DD"}, '
        '"fromCity": {"type": "string", "description": "出发城市"}, "toCity": '
        '{"type": "string", "description": "到达城市"}}, "required": ["date", '
        '"fromCity", "toCity"], "additionalProperties": false, "$schema": '
        '"http://json-schema.org/draft-07/schema#"}}}]',
    },
    {
        "uuid": "8f44647b-6dd4-503b-841c-2168b0373f0f",
        "subset_name": "single-turn-original",
        "messages": '[{"role": "user", "content": "I\'m planning a trip from Chengdu to Guilin '
        "next Saturday, but I want to minimize travel time. What's the fastest "
        "train option available, and what are the exact departure/arrival times "
        'and ticket prices for a second-class seat?"}, {"role": "assistant", '
        '"content": "", "function_call": {"name": "12306-mcp-server-search", '
        '"arguments": "{\\"date\\": \\"2023-10-14\\", \\"fromCity\\": '
        '\\"Chengdu\\", \\"toCity\\": \\"Guilin\\"}"}}, {"role": "function", '
        '"content": "日期需为0~15天内", "name": "12306-mcp-server-search"}, {"role": '
        '"assistant", "content": "It seems that I cannot query for dates more than '
        "15 days in advance. Since your trip is next Saturday, let's adjust the "
        "date and try again. I'll check for the fastest train options available "
        "for the upcoming Saturday within the allowed range. Let me know if you'd "
        'like to proceed with the corrected date."}]',
        "available_tools": '[{"type": "function", "function": {"name": '
        '"12306-mcp-server-search", "description": "查询12306火车票", '
        '"parameters": {"type": "object", "properties": {"date": {"type": '
        '"string", "format": "date", "description": "出发日期 格式：YYYY-MM-DD"}, '
        '"fromCity": {"type": "string", "description": "出发城市"}, "toCity": '
        '{"type": "string", "description": "到达城市"}}, "required": ["date", '
        '"fromCity", "toCity"], "additionalProperties": false, "$schema": '
        '"http://json-schema.org/draft-07/schema#"}}}]',
    },
]


@pytest.mark.parametrize("raw", RAW_DATE_ROWS, ids=lambda raw: raw["uuid"])
@pytest.mark.parametrize("strip", [False, True])
def test_released_qualified_train_calls_have_no_visible_calendar_anchor(raw, strip):
    sample, _ = toucan._convert_sample(raw, "Qwen3")
    pipeline = toucan._pipeline(FilterConfig(strip_thinking=strip))
    assert pipeline.process(sample) is None
    assert pipeline.drops == {"ungrounded_relative_date": 1}
