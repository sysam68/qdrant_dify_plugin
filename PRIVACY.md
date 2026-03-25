# Qdrant Dify Plugin Privacy Policy

This document follows the [Dify Privacy Policy Guidelines](https://docs.dify.ai/plugin-dev-zh/0312-privacy-protection-guidelines) and applies to every build submitted to the Dify Marketplace or distributed as a `.difypkg`.

## Data We Process

| Data Type | Details | Purpose |
| --- | --- | --- |
| Connection configuration | Qdrant `base_url`, API key, default vector size/distance metric (set by the user in Dify) | Establishes a secure connection to the user-owned Qdrant cluster and enables automatic collection creation |
| Request payloads | Collection names, point IDs, payload objects, vectors, texts, filters supplied by workflow nodes | Execute upsert/search/scroll/delete operations; the data exists only for the lifetime of the API request |
| Embedding inputs | Texts that require embeddings when `embedding_model_config` is provided | Forwarded through Dify’s reverse model invocation pipeline; the plugin never communicates with the embedding provider directly |

The plugin does **not** collect names, emails, device identifiers, or any other personal data beyond what the workflow owner explicitly supplies.

## Third-Party Services

| Service | Usage | Privacy Policy |
| --- | --- | --- |
| Qdrant Vector Database (user-owned cluster) | Stores and retrieves vectors/payloads. All data stays inside the user-controlled Qdrant deployment. | https://qdrant.tech/privacy/ |
| Embedding providers selected in Dify | Convert text to vectors via Dify’s embedding selector. Credentials remain managed by Dify. | Refer to the chosen provider’s policy (e.g., OpenAI, Azure, Cohere). |

No data is transmitted to the plugin author’s infrastructure or to any vendor not listed above.

## Storage & Retention

- The plugin **does not persist** point data, query results, or credentials locally; persistence happens only within the user’s Qdrant cluster.
- In-memory data exists solely within the current request context and is discarded immediately afterward.

## Security Measures

- Relies on Dify’s credential storage; API keys are never logged.
- All outbound traffic uses HTTPS.
- The codebase contains no telemetry, analytics, or unsolicited logging.

## User Control

- Users may rotate or delete the Qdrant API key from the plugin credential panel at any time; updates take effect immediately.
- To remove stored vectors/payloads, use the Delete/Data-Management/Collection-Management tools or operate directly inside Qdrant.

## Contact

Questions about this policy can be raised via GitHub Issues or through the developer contact listed in the Dify Marketplace submission. Please keep the contact information accurate before opening a PR.

This policy will be updated whenever Marketplace requirements or plugin capabilities change. Always reference the latest `PRIVACY.md` in the repository.