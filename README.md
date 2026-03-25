# Qdrant Plugin for Dify

A comprehensive Dify plugin for Qdrant vector database integration. Store, search, and manage vector embeddings directly within Dify workflows.

**Key Capabilities:**
- Automatic text-to-vector conversion using Dify's embedding models
- Dense and hybrid (dense + sparse BM25) similarity search
- Flexible point storage with standard Qdrant format support
- Full collection lifecycle management

Ideal for building RAG applications, knowledge bases, and semantic search systems.

## Features

### 🔍 Core Operations

- **Upsert Text**: Insert or update text content with automatic embedding generation
- **Upsert Point**: Insert or update points using standard point format (id, vector, payload)
- **Vector Search**: Similarity search using text or pre-computed vectors (finds nearest neighbors)
- **Hybrid Search**: Combine dense vectors, sparse vectors (BM25) with fusion (RRF) for improved retrieval
- **Data Management**: Query points by IDs/filter, scroll through, or delete points in collections
- **Collection Management**: Create, delete, and manage collections

### ✨ Key Advantages

- **Text-to-Vector Integration**: Supports direct text input with automatic embedding generation
- **Flexible Input**: Support both text input (auto-embed) and pre-computed vectors
- **Seamless Dify Integration**: Uses Dify's embedding model selector directly in tool parameters
- **Auto-Collection Creation**: Automatically creates collections if they don't exist

## Installation

### Prerequisites

- Dify platform
- Qdrant instance (Cloud or self-hosted)

### Quick Start

1. **Get Qdrant Instance**:
   - Option 1 (Recommended): Create a free account at [Qdrant Cloud](https://cloud.qdrant.io/)
   - Option 2: Deploy locally: `docker run -p 6333:6333 qdrant/qdrant`

2. **Install Plugin**:
   - Upload the `.difypkg` file in Dify's plugin management interface

3. **Configure Credentials**:
   - **Qdrant URL**: Your Qdrant instance URL (e.g., `https://xxx.cloud.qdrant.io:6333`)
   - **API Key**: Your Qdrant API key (required for Cloud, optional for local)
   - **Default Vector Dimensions**: Set according to your embedding model (e.g., `1536` for OpenAI)
   - **Default Distance Metric**: Select `Cosine` (recommended for text embeddings)

## Usage

### Upsert Text

Store text content with automatic vector generation:

- **Collection Name**: Name of the collection to store data
- **Texts**: JSON format `{"chunks": [{"text": "chunk1"}, {"text": "chunk2"}]}`
- **Embedding Model**: Select an embedding model (e.g., OpenAI text-embedding-3-small)
- **Point IDs** (optional): Custom IDs for each chunk, or auto-generated UUIDs
- **Wait for Completion**: Wait for operation to complete (default: true)

**Example Input Format:**
```json
{
  "chunks": [
    {"text": "First paragraph of text"},
    {"text": "Second paragraph of text"}
  ]
}
```

### Upsert Point

Store points using standard Qdrant point format:

- **Collection Name**: Name of the collection to store data
- **Data**: Array of point objects, each with `id`, `vector`, and `payload` fields
- **Wait for Completion**: Wait for operation to complete (default: true)

**Example Input:**
```json
[
  {
    "id": 1,
    "vector": [0.1, 0.2, 0.3],
    "payload": {"text": "hello"}
  },
  {
    "id": 2,
    "vector": [0.4, 0.5, 0.6],
    "payload": {"text": "world"}
  }
]
```

### Vector Search

Run dense vector similarity search with either raw text (auto-embedded) or a pre-computed vector:

- **Collection Name**: Target collection
- **Query Text**: Plain text query (auto-converted via the selected embedding model)
- **Embedding Model**: Required when using `text`
- **Query Vector**: Optional manual vector override (JSON array or comma-separated numbers)
- **Filter**: Qdrant metadata filter in JSON string form
- **Limit / Score Threshold**: Control result count and minimum score
- **With Payload / With Vector**: Toggle payloads or embeddings in the response

When both `text` and `vector` are supplied, the explicit `vector` takes precedence for similarity search.

### Hybrid Search

Combine dense and sparse retrieval using Qdrant’s Query API (1.10+):

- **Text**: Required; drives both dense embeddings and a BM25-style sparse vector
- **Embedding Model**: Used for dense vector generation
- **Dense Vector / Sparse Vector**: Optional overrides if you pre-compute either side
- **Fusion Method**: Defaults to `rrf` (Reciprocal Rank Fusion) for best coverage
- **Prefetch Limit**: Candidates per method before fusion (recommended 2–5× the final `limit`)
- **Filter / With Payload / With Vector**: Same semantics as vector search

Use Hybrid Search for production-grade RAG answers where keyword grounding and semantic recall must be balanced automatically.

### Data Management

Perform non-similarity operations on points:

- **Operation**: `query` (retrieve by `point_ids` and/or `filter`), `scroll` (paginate every point), or `delete`
- **Point IDs**: JSON array string (e.g., `[1,"uuid-2"]`)
- **Filter**: Qdrant filter JSON for metadata-based selection (indexes required)
- **Limit**: Max points per call (applies to `query`+filter and `scroll`)
- **With Payload / With Vector**: Decide whether to return metadata and/or embeddings

This tool is ideal for audit, maintenance, or deterministic retrieval by primary key—use Vector/Hybrid search for similarity.

### Collection Management

Create, delete, or inspect collections from the same node:

- **Operation**: `create_collection`, `delete_collection`, or `get_collection_info`
- **Collection Name**: Required for every operation
- **Vector Size / Distance**: Optional overrides when creating a new collection (falls back to provider defaults)

Use this when automations need to bootstrap or clean up Qdrant resources without leaving Dify.

## Supported Operations

### Data Operations

- **Upsert Text**: Store text with automatic embedding generation
- **Upsert Point**: Store points using standard format
- **Vector Search**: Dense similarity search with text or vector
- **Hybrid Search**: Dense + sparse retrieval with RRF fusion
- **Data Management**: Deterministic operations (query by ID/filter, scroll, delete)

### Collection Operations

- **Collection Management**: Create, delete, or inspect collections via one tool (respects provider defaults and per-call overrides)

## Common Embedding Model Dimensions

- OpenAI text-embedding-ada-002: **1536**
- OpenAI text-embedding-3-small: **1536**
- OpenAI text-embedding-3-large: **3072**
- BERT-base: **768**
- Cohere embed-english-v3.0: **1024**

⚠️ **Important**: Ensure the embedding model dimension matches your collection's vector dimension.

## Workflow Examples

### Example 1: Store and Search Text

```
1. Start Node (with text input)
2. Qdrant Upsert Text
   - Collection: "my_docs"
   - Texts: {{#start.text#}}
   - Embedding Model: text-embedding-3-small
3. Qdrant Query
   - Collection: "my_docs"
   - Query Text: "search query"
   - Embedding Model: text-embedding-3-small
   - Limit: 5
```

### Example 2: Store Pre-computed Vectors

```
1. Code Node (generate vectors)
   - Output: points array
2. Qdrant Upsert Point
   - Collection: "my_vectors"
   - Data: {{#code.points#}}
```

## Troubleshooting

### Vector Dimension Mismatch

If you see "Vector dimension mismatch" error:
1. Check your Provider settings → Default Vector Dimensions
2. Ensure the embedding model matches this dimension
3. Or recreate the collection with the correct dimension

### 403 Forbidden Error

If you see "403 Forbidden" error:
1. Check your API Key has 'write' or 'admin' permissions
2. Verify the API Key in Qdrant Cloud Dashboard
3. Ensure the collection exists or API Key can create collections

### Collection Auto-Creation

The plugin automatically creates collections if they don't exist:
- Uses default vector dimensions from Provider settings
- Uses default distance metric (Cosine)
- Collection will be created on first upsert operation

## License

See [LICENSE](./LICENSE) file for details.

## Support

For issues and questions:
- Check Dify plugin documentation
- Review Qdrant documentation: https://qdrant.tech/documentation/
- Open an issue on the plugin repository

