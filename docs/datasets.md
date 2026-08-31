# Dataset sources and licenses

`fcanalysis` contains loader code and small regression metadata, not dataset
content. Each loader pins an upstream repository revision so that a given code
revision names the source bytes it expects. Availability, access, and legal
terms remain controlled by the upstream publisher.

The table records source-declared terms at the pinned revision. It is an
engineering inventory, not legal advice. Review the linked card and any linked
component terms before downloading, transforming, training on, or
redistributing data. Preserve required attribution, citations, notices, license
links, and change statements.

| Loader | Pinned source | Source-declared license or terms | Important source boundary |
| --- | --- | --- | --- |
| `apigen_mt` | [`Salesforce/APIGen-MT-5k@abc4a517`](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k/blob/abc4a517d67c541f85f6470cbd8fd3186b36830e/README.md) | CC BY-NC 4.0; the card also requests acceptance of usage terms and citation | Non-commercial license. The card describes the release as research-only and calls for downstream accuracy, safety, fairness, and legal review. |
| `dolci` | [`allenai/Dolci-Instruct-SFT-Tool-Use@dc042846`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use/blob/dc042846f0f2de0f15eedae3d6ced04223ed47eb/README.md) | ODC-BY; Ai2 Responsible Use Guidelines | The subset card delegates fuller documentation to the main Dolci dataset card. ODC-BY attribution applies to the database. |
| `nemotron_agentic_v1` | [`nvidia/Nemotron-Agentic-v1@650d5909`](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1/blob/650d590978ca35c8f1ecea2faf136e5fac421b62/README.md) | CC BY 4.0; the card identifies an Apache 2.0 Glaive component | Attribution is required. The publisher asks downstream users to assess industry requirements and misuse risks. |
| `nemotron_agentic_v2` | [`nvidia/Nemotron-SFT-Agentic-v2@49e79a3b`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2/blob/49e79a3be5ab8cf7511a12958b95cfd6408cd8db/README.md) | CC BY 4.0 with Apache 2.0 and MIT component licenses | The tool-use subset draws on several named function-calling sources and includes TOUCAN trajectories. Component attribution and terms still matter. |
| `nemotron_terminal` | [`nvidia/Nemotron-Terminal-Corpus@a1667c4f`](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus/blob/a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9/README.md) | CC BY 4.0 | Terminal/code trajectories can contain failed commands, incomplete episodes, unsafe operations, or environment-specific output. Loader filters expose some structural checks, not a safety guarantee. |
| `toolmind` | [`Nanbeige/ToolMind@8020ed1c`](https://huggingface.co/datasets/Nanbeige/ToolMind/blob/8020ed1c03c367e4eb720ac3828ab4b0b95d8baf/README.md) | Apache 2.0 at the aggregate card | The card says ToolMind combines synthetic data with processed open-source sources, including APIGen-MT-5k. Do not assume the aggregate label removes source-level restrictions such as non-commercial terms. The card warns about harmful, biased, or discriminatory output. |
| `toucan` | [`Agent-Ark/Toucan-1.5M@0df3cf37`](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M/blob/0df3cf37f2abefb380370cfb02eabea2a35ae782/README.md) | Apache 2.0 | The card says responses were collected from June through September 2025 and may be time-bound. Community MCP servers can fail or drift; some failure trajectories are intentionally retained. |
| `txt360` | [`LLM360/TxT360-3efforts@bfc4a082`](https://huggingface.co/datasets/LLM360/TxT360-3efforts/blob/bfc4a082d11967cd7810fe0b773be87bf54fb32e/README.md) | CC BY 4.0 | The repository now resolves as `IFM/TxT360-3efforts`. The card describes a mixture of permissively licensed public sources and synthetic answers. Its quality, deduplication, and decontamination statements are upstream claims, not independently established by this package. |

## Reproducibility contract

The exact revisions above are constants in the loader modules. Dataset loaders
also pin relevant splits, configurations, or file names. Some source hosts can
redirect a repository while retaining commit access, as currently happens for
TxT360.

The full-dataset E2E tests reconstruct normalized output and compare:

- the accepted sample count;
- the complete `LoadReport`; and
- the SHA-256 of canonical uncompressed JSONL bytes.

Those checks detect implementation or source drift. They do not establish that
the source is accurate, lawful for a particular downstream use, uncontaminated,
safe, representative, or scientifically valid.

## Attribution and redistribution

When publishing a transformed dataset or work derived from one:

1. identify each actual source and pinned revision used;
2. follow the source card, license, citation, and responsible-use terms;
3. retain notices and state material modifications where required;
4. account for component datasets rather than relying only on an aggregate
   repository label; and
5. avoid redistributing local caches or generated source rows through this
   software repository.

The MIT license in this repository covers `fcanalysis` source code only.
