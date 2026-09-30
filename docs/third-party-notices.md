# Third-party notices

The project's own code remains under the [MIT license](../LICENSE). The
following incorporated material retains its upstream license.

## Qwen3.5 chat template

The text renderer in
[`src/fcanalysis/tokenization.py`](../src/fcanalysis/tokenization.py) adapts the
Qwen team's
[`Qwen/Qwen3.5-4B` chat template](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/chat_template.jinja)
at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. In particular,
`_TOOL_INSTRUCTIONS` reproduces its fixed tool instructions verbatim.
The [pinned model card](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md)
identifies the upstream license as Apache License, Version 2.0. A copy of those
terms is included in [`LICENSES/Apache-2.0.txt`](../LICENSES/Apache-2.0.txt).

The adaptation implements the text template in Python and records category
spans for token accounting. It preserves reasoning on every assistant message,
preserves structured reasoning whitespace and safe inline native spans, and
reports unresolved inline boundaries explicitly. These intentional changes are
described in [the tokenization module](../src/fcanalysis/tokenization.py). Tokenizer
assets are loaded separately; they are not included in this repository.

## ToolMind-Web source protocol

[`src/fcanalysis/loaders/toolmind_web.py`](../src/fcanalysis/loaders/toolmind_web.py)
incorporates the fixed tool instructions and embedded catalog from
[`Nanbeige/ToolMind-Web-QA`](https://huggingface.co/datasets/Nanbeige/ToolMind-Web-QA/blob/2690dcdfdd82ab147aad4a65ab231d4e344f5cd0/README.md),
revision `2690dcdfdd82ab147aad4a65ab231d4e344f5cd0`. The source card declares
Apache License, Version 2.0; its terms are included in
[`LICENSES/Apache-2.0.txt`](../LICENSES/Apache-2.0.txt).

The adapter uses the fixed text to recognize the released protocol. It
normalizes the wrapper's XML calls into canonical calls, extracts the callable
description and catalog into the tool channel, and removes only the recognized
transport-format instructions. The module docstring specifies these changes.
Original conversations and tool observations are loaded separately and are not
included in the package.

## Harbor Terminus-2 parser and Nemotron-Terminal instructions

[`_terminus_json.py`](../src/fcanalysis/loaders/_terminus_json.py) adapts
Harbor's pure
[Terminus JSON parser](https://github.com/harbor-framework/harbor/blob/265c303150b08d0e4aab695dd96ff5f6b4159db9/src/harbor/agents/terminus_2/terminus_json_plain_parser.py)
at revision `265c303150b08d0e4aab695dd96ff5f6b4159db9`. Harbor declares
[Apache-2.0](https://github.com/harbor-framework/harbor/blob/265c303150b08d0e4aab695dd96ff5f6b4159db9/LICENSE);
the terms are included in [LICENSES/Apache-2.0.txt](../LICENSES/Apache-2.0.txt).
The parser's behavior is preserved, with module documentation and modernized
type spelling/formatting. The adapter separately adds conservative source gates,
a complete-object parsing fast path, and narrowly defined feedback comparisons.
It never executes commands. The comparison pin is not asserted to be NVIDIA's
exact deployed producer revision.

[`_nemotron_terminal_protocol.py`](../src/fcanalysis/loaders/_nemotron_terminal_protocol.py)
reproduces NVIDIA's fixed initial instruction prefix solely to recognize its
released interface. Attribution: NVIDIA, *Nemotron-Terminal-Corpus*, revision
`a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9`, associated with Pi et al.,
[*On Data Engineering for Scaling LLM Terminal Capabilities*](https://arxiv.org/abs/2602.21193).
The [pinned dataset card](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus/blob/a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9/README.md)
declares [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The prefix is unchanged; task-specific instructions, conversations, terminal
observations and dataset files are loaded separately and are not shipped in the
package. Dataset attribution does not clear inherited-source terms.
