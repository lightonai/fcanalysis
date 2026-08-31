"""Small, deterministic fcanalysis smoke test with no dataset download."""

from fcanalysis import ConversationSample, __version__
from fcanalysis.core import analyze_sample
from fcanalysis.validation import validate_arguments


def main() -> None:
    sample = ConversationSample(
        dataset="synthetic-smoke",
        sample_id="weather-1",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Return the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        messages=[
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temperature_c":21}'},
            {"role": "assistant", "content": "It is 21 °C in Paris."},
        ],
    )

    analysis = analyze_sample(sample.messages, extract_function_names=True)
    argument_errors = validate_arguments(
        {"city": "Paris"}, sample.tools[0]["function"]["parameters"]
    )

    print(f"fcanalysis {__version__}")
    print(f"turns={analysis.num_real_turns} pattern={analysis.turn_patterns[0].value}")
    print(f"arguments_valid={not argument_errors}")


if __name__ == "__main__":
    main()
