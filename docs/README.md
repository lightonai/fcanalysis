# fcanalysis documentation

`fcanalysis` loads function-calling datasets into a common conversation format,
validates and deduplicates them, and measures tool use.

Start with [Getting started](getting-started.md) for an example you can run
without downloading a dataset. Then read [How conversations are loaded](conversations.md)
to understand what the output means.

| What you want to do | Read |
| --- | --- |
| Install the package and run a first example | [Getting started](getting-started.md) |
| Understand the conversation format and its relationship to SFT | [How conversations are loaded](conversations.md) |
| Choose a loader, configure it, and understand its report | [Using loaders](loading.md) |
| Support your own source format | [Adding a loader](adding-loaders.md) |
| Measure tool use, inspect supervision, or combine loaded data | [Analyzing conversations](analysis.md) |

For a specific behavior, see [call/result matching](adding-loaders.md#matching-calls-and-results),
[assistant-target prefixes](loading.md#assistant-target-prefixes),
[Terminal conversations](loading.md#terminal-conversations), or the
[shared components](adding-loaders.md#shared-components).

The [loader catalogue](loading.md#available-loaders) links to each adapter's
source revision, defaults, and limitations. [Third-party notices](third-party-notices.md)
cover incorporated templates and parser material.
