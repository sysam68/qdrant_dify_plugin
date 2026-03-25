from __future__ import annotations

import json
import math
import uuid
from collections.abc import Generator
from typing import Any

import httpx

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from utils.qdrant_helpers import build_headers, resolve_endpoint


class QdrantTool(Tool):
    """
    Qdrant vector database tool supporting text input with automatic embedding generation.
    
    Similar to Weaviate plugin:
    - Accepts text input (texts for upsert, text for query)
    - Optionally accepts pre-computed vectors
    - If vector not provided, automatically generates embeddings using configured embedding provider
    
    Phase 1 (exposed in UI):
    - operation=upsert             -> write points into a collection (auto-creates collection if missing)
                                      Accepts: texts (auto-embed) or vectors (pre-computed)
    - operation=query              -> similarity search within a collection
                                      Accepts: text (auto-embed) or vector (pre-computed)
    - operation=scroll             -> paginate through points in a collection (via Data Management action)
    - operation=delete             -> delete points from a collection (via Data Management action)

    Advanced operations (implemented but not necessarily exposed in the UI yet):
    - operation=embed              -> call server-side vectorizers (requires paid Qdrant Cloud account; currently hidden)
    - operation=recommend          -> get recommendations based on positive/negative examples
    - operation=create_collection  -> create a new collection (optional, upsert auto-creates with default config)
    - operation=delete_collection  -> delete a collection
    - operation=get_collection_info -> get collection information

    Automatic Embedding:
    - Configure 'Embedding Provider' and 'Embedding Model' in Provider settings
    - When texts/text are provided without vectors, plugin automatically calls embedding API
    - Supported providers: OpenAI, HuggingFace, Cohere, Jina, Voyage AI
    """

    PROVIDER = "qdrant"
    DEFAULT_TIMEOUT = 60.0  # Increased timeout to handle network latency

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        credentials = self.runtime.credentials or {}

        base_url = (credentials.get("base_url") or "").strip()
        if not base_url:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "status": "error",
                    "message": "Missing `base_url` in credentials.",
                }
            )
            return

        # Determine operation from tool_parameters
        # For specific tools (qdrant-upsert-text, qdrant-vector-search, etc.), 
        # operation may not be in tool_parameters, so we infer from parameters
        operation = tool_parameters.get("operation")
        
        # If operation is not provided, infer from tool parameters
        if not operation:
            # Check for new format parameters to infer operation
            if "collection" in tool_parameters:
                # Check for upsert indicators
                if (
                    "texts" in tool_parameters
                    or "vectors" in tool_parameters
                    or "data" in tool_parameters
                ):
                    operation = "upsert"
                # Check for query/vector search indicators
                elif "text" in tool_parameters or "vector" in tool_parameters:
                    operation = "query"
                # Check for hybrid search indicators
                elif "dense_vector" in tool_parameters or "sparse_vector" in tool_parameters:
                    operation = "hybrid_search"
                # Check for data management indicators
                elif "point_ids" in tool_parameters or "filter" in tool_parameters:
                    operation = "query"  # Data management query
                elif "scroll" in tool_parameters or tool_parameters.get("operation") == "scroll":
                    operation = "scroll"
                elif "delete" in tool_parameters or tool_parameters.get("operation") == "delete":
                    operation = "delete"
                # Default: don't assume embed (which requires vectorizer)
            else:
                    operation = None
        
        if operation:
            operation = operation.lower()
        else:
            # If we can't determine operation, return error instead of defaulting to embed
            yield self.create_json_message({
                "provider": self.PROVIDER,
                "status": "error",
                "message": "Unable to determine operation. Please provide 'operation' parameter or use tool-specific parameters (texts, text, vector, etc.).",
            })
            return
        items, items_error = self._parse_items(tool_parameters.get("items", "[]"))
        if items_error:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "status": "error",
                    "message": items_error,
                }
            )
            return

        options, options_error = self._parse_options(tool_parameters.get("options") or "{}")
        if options_error:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "status": "error",
                    "message": options_error,
                }
            )
            return
        
        headers, header_error = build_headers(credentials, include_content_type=True)
        if header_error:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "status": "error",
                    "message": header_error,
                }
            )
            return

        if operation == "embed":
            yield from self._handle_embed(base_url, items, options, headers, credentials)
            return
        if operation == "upsert":
            # Support both new simple fields and legacy JSON format
            if "collection" in tool_parameters:
                # New format: simple fields (support batch)
                collection = tool_parameters.get("collection", "").strip()
                vector_name, vector_name_error = self._normalize_optional_string(
                    tool_parameters.get("vector_name"), "vector_name"
                )
                if vector_name_error:
                    yield self.create_json_message({
                        "provider": self.PROVIDER,
                        "operation": "upsert",
                        "status": "error",
                        "message": vector_name_error,
                    })
                    return
                
                # New point format: direct "data" array (array[object]) from qdrant-upsert-point tool
                data_value = tool_parameters.get("data")
                if data_value not in (None, "", []):
                    try:
                        if isinstance(data_value, str):
                            parsed_data = json.loads(data_value)
                        else:
                            parsed_data = data_value
                    except json.JSONDecodeError as exc:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "upsert",
                            "status": "error",
                            "message": f"Invalid JSON in 'data' parameter: {exc}",
                        })
                        return
                    
                    if isinstance(parsed_data, dict):
                        parsed_data = [parsed_data]
                    
                    if not isinstance(parsed_data, list):
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "upsert",
                            "status": "error",
                            "message": "'data' must be an array of objects. Received unsupported format.",
                        })
                        return
                    
                    items = []
                    for idx, point in enumerate(parsed_data):
                        if not isinstance(point, dict):
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "upsert",
                                "status": "error",
                                "message": f"Point {idx} is not an object. Each entry must be an object with id, vector, payload.",
                            })
                            return
                        
                        point_id = point.get("id")
                        vector = point.get("vector")
                        payload = point.get("payload", {})
                        
                        if point_id is None:
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "upsert",
                                "status": "error",
                                "message": f"Point {idx} is missing 'id'.",
                            })
                            return
                        if vector is None:
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "upsert",
                                "status": "error",
                                "message": f"Point {idx} is missing 'vector'.",
                            })
                            return
                        if not isinstance(vector, list):
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "upsert",
                                "status": "error",
                                "message": f"Point {idx} vector must be an array of numbers.",
                            })
                            return
                        try:
                            vector_clean = [float(val) for val in vector]
                        except (TypeError, ValueError):
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "upsert",
                                "status": "error",
                                "message": f"Point {idx} vector contains non-numeric values.",
                            })
                            return
                        
                        if payload is None:
                            payload = {}
                        elif not isinstance(payload, dict):
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "upsert",
                                "status": "error",
                                "message": f"Point {idx} payload must be an object (can be empty {{}}).",
                            })
                            return
                        
                        items.append({
                            "id": point_id,
                            "vector": vector_clean,
                            "payload": payload,
                        })
                    
                    if not items:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "upsert",
                            "status": "error",
                            "message": "'data' array is empty. Provide at least one point object.",
                        })
                        return
                    
                    options = {
                        "collection": collection,
                    }
                    if vector_name:
                        options["vector_name"] = vector_name
                    if "wait" in tool_parameters:
                        options["wait"] = bool(tool_parameters.get("wait", True))
                    
                    yield from self._handle_upsert(base_url, items, options, headers)
                    return
                
                # Handle texts: can be string, list, or dict (from mapped variables)
                texts_value = tool_parameters.get("texts", "")
                if isinstance(texts_value, (dict, list)):
                    texts_str = json.dumps(texts_value)
                elif isinstance(texts_value, str):
                    texts_str = texts_value.strip()
                else:
                    texts_str = str(texts_value).strip() if texts_value else ""
                
                # Handle vectors: can be string or list
                vectors_value = tool_parameters.get("vectors") or tool_parameters.get("vector", "")
                if isinstance(vectors_value, list):
                    # If vectors is already a list, keep as is (will be processed later)
                    vectors_str = json.dumps(vectors_value)
                elif isinstance(vectors_value, str):
                    vectors_str = vectors_value.strip()
                else:
                    vectors_str = str(vectors_value).strip() if vectors_value else ""
                
                # Handle point_ids: can be string or list
                point_ids_value = tool_parameters.get("point_ids") or tool_parameters.get("point_id", "")
                if isinstance(point_ids_value, list):
                    point_ids_str = ",".join(str(pid) for pid in point_ids_value)
                elif isinstance(point_ids_value, str):
                    point_ids_str = point_ids_value.strip()
                else:
                    point_ids_str = str(point_ids_value).strip() if point_ids_value else ""
                
                payload_object, payload_error = self._parse_optional_payload_object(
                    tool_parameters.get("payload")
                )
                if payload_error:
                    yield self.create_json_message({
                        "provider": self.PROVIDER,
                        "operation": "upsert",
                        "status": "error",
                        "message": payload_error,
                    })
                    return
                
                vectors = []
                texts: list[str] = []
                
                # Priority: vectors > texts (if both provided, use vectors)
                if vectors_str:
                    # Parse vectors (support batch: semicolon-separated)
                    vectors_raw = vectors_str.split(";") if ";" in vectors_str else [vectors_str]
                    for vec_str in vectors_raw:
                        vec_str = vec_str.strip()
                        if not vec_str:
                            continue
                        try:
                            if vec_str.startswith("[") and vec_str.endswith("]"):
                                # JSON array format
                                vec = json.loads(vec_str)
                            else:
                                # Comma-separated format
                                vec = [float(x.strip()) for x in vec_str.split(",") if x.strip()]
                            vectors.append(vec)
                        except (ValueError, json.JSONDecodeError):
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "upsert",
                                "status": "error",
                                "message": f"Invalid vector format: '{vec_str}'. Use comma-separated values (e.g., '0.1,0.2,0.3') or JSON array.",
                            })
                            return
                elif texts_str:
                    # Auto-generate vectors from texts
                    # Parse texts: can be JSON string with chunks format or plain string/list
                    try:
                        # Try parsing as JSON first (expected format: {"chunks": [{"text": "..."}]})
                        parsed = json.loads(texts_str) if isinstance(texts_str, str) else texts_str
                        if isinstance(parsed, dict) and "chunks" in parsed:
                            # Standard format: {"chunks": [{"text": "..."}]}
                            chunks = parsed.get("chunks", [])
                            for chunk in chunks:
                                if isinstance(chunk, dict) and "text" in chunk:
                                    text = chunk.get("text", "").strip()
                                    if text:
                                        texts.append(text)
                        elif isinstance(parsed, list):
                            # List format: [{"text": "..."}] or ["text1", "text2"]
                            for item in parsed:
                                if isinstance(item, dict) and "text" in item:
                                    text = item.get("text", "").strip()
                                    if text:
                                        texts.append(text)
                                elif isinstance(item, str):
                                    text = item.strip()
                                    if text:
                                        texts.append(text)
                        elif isinstance(parsed, str):
                            # Plain string: treat as single text or semicolon-separated
                            texts_raw = parsed.replace("\n", ";").split(";")
                            texts = [t.strip() for t in texts_raw if t.strip()]
                        else:
                            # Fallback: treat as plain string
                            texts_raw = str(texts_str).replace("\n", ";").split(";")
                            texts = [t.strip() for t in texts_raw if t.strip()]
                    except (json.JSONDecodeError, TypeError):
                        # Not JSON, treat as plain string
                        texts_raw = str(texts_str).replace("\n", ";").split(";")
                        texts = [t.strip() for t in texts_raw if t.strip()]
                    
                    if not texts:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "upsert",
                            "status": "error",
                            "message": "No valid texts found. Provide at least one text.",
                        })
                        return
                    
                    # Generate embeddings using Dify's embedding model
                    embedding_model_config = tool_parameters.get("embedding_model_config")
                    # Strict check: embedding_model_config must be a non-empty dict (same logic as hybrid_search)
                    # Handle various falsy cases: None, empty string, empty dict, etc.
                    is_valid_config = (
                        embedding_model_config is not None
                        and embedding_model_config != ""
                        and isinstance(embedding_model_config, dict)
                        and len(embedding_model_config) > 0
                    )
                    
                    if not is_valid_config:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "upsert",
                            "status": "error",
                            "message": "Please select an embedding model in the node configuration to generate vectors from texts. When using 'texts' input, 'Embedding Model' parameter is required.",
                        })
                        return
                    
                    try:
                        vectors = self._generate_embeddings(embedding_model_config, texts)
                    except Exception as e:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "upsert",
                            "status": "error",
                            "message": f"Failed to generate embeddings: {str(e)}",
                        })
                        return
                else:
                    yield self.create_json_message({
                        "provider": self.PROVIDER,
                        "operation": "upsert",
                        "status": "error",
                        "message": "Either 'texts' or 'vectors' must be provided.",
                    })
                    return
                
                if not vectors:
                    yield self.create_json_message({
                        "provider": self.PROVIDER,
                        "operation": "upsert",
                        "status": "error",
                        "message": "No vectors generated. Please check your input.",
                    })
                    return
                
                # Parse point IDs (support batch: comma-separated)
                point_ids = []
                if point_ids_str:
                    id_strs = point_ids_str.split(",")
                    for id_str in id_strs:
                        id_str = id_str.strip()
                        if id_str:
                            try:
                                point_ids.append(int(id_str))
                            except ValueError:
                                point_ids.append(id_str)  # Keep as string (UUID)
                
                # Auto-generate IDs if not provided or not enough
                while len(point_ids) < len(vectors):
                    point_ids.append(str(uuid.uuid4()))
                
                payloads: list[dict[str, Any]] = []
                if payload_object is not None:
                    payloads = [payload_object]
                else:
                    payloads, payloads_error = self._parse_payload_entries(
                        tool_parameters.get("payloads")
                    )
                    if payloads_error:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "upsert",
                            "status": "error",
                            "message": payloads_error,
                        })
                        return
                
                # If single payload but multiple vectors, apply to all
                if len(payloads) == 1 and len(vectors) > 1:
                    payloads = payloads * len(vectors)
                
                # Build items from parsed data
                items = []
                for i, vector in enumerate(vectors):
                    item = {
                        "id": point_ids[i] if i < len(point_ids) else str(uuid.uuid4()),
                        "vector": vector,
                    }
                    payload_for_item = dict(payloads[i]) if i < len(payloads) else {}
                    if i < len(texts):
                        # `data` is always aligned with the chunk content used to build the embedding.
                        payload_for_item["data"] = texts[i]
                    if payload_for_item:
                        item["payload"] = payload_for_item
                    items.append(item)
                
                options = {
                    "collection": collection,
                }
                if vector_name:
                    options["vector_name"] = vector_name
                if "wait" in tool_parameters:
                    options["wait"] = bool(tool_parameters.get("wait", True))
            # Legacy format: continue with existing logic
            yield from self._handle_upsert(base_url, items, options, headers)
            return
        if operation == "query":
            if "collection" in tool_parameters:
                collection = tool_parameters.get("collection", "").strip()
                filter_value = tool_parameters.get("filter")
                point_ids_value = tool_parameters.get("point_ids")
                text_value = tool_parameters.get("text")
                vector_value = tool_parameters.get("vector")
                
                text_str = text_value.strip() if isinstance(text_value, str) else ""
                vector_str = vector_value.strip() if isinstance(vector_value, str) else ""
                vector_is_list = isinstance(vector_value, list) and len(vector_value) > 0
                has_point_ids = point_ids_value not in (None, "", [])
                has_filter_only = (
                    filter_value not in (None, "", {})
                    and not text_str
                    and not vector_str
                    and not vector_is_list
                )
                
                if has_point_ids or has_filter_only:
                    options = {"collection": collection}
                    point_ids: list[Any] = []
                    
                    if has_point_ids:
                        if isinstance(point_ids_value, list):
                            point_ids = [pid for pid in point_ids_value if pid not in (None, "")]
                        elif isinstance(point_ids_value, str):
                            pid_str = point_ids_value.strip()
                            if pid_str:
                                try:
                                    try:
                                        parsed_ids = json.loads(pid_str)
                                        if isinstance(parsed_ids, list):
                                            point_ids = parsed_ids
                                        else:
                                            raise ValueError("Point IDs must be an array")
                                    except json.JSONDecodeError:
                                        for token in pid_str.split(","):
                                            token = token.strip()
                                            if not token:
                                                continue
                                            try:
                                                point_ids.append(int(token))
                                            except ValueError:
                                                point_ids.append(token)
                                except Exception as exc:
                                    yield self.create_json_message({
                                        "provider": self.PROVIDER,
                                        "operation": "query",
                                        "status": "error",
                                        "message": f"Invalid point IDs format: {str(exc)}. Use JSON array or comma-separated list.",
                                    })
                                    return
                        else:
                            point_ids = [point_ids_value]
                    
                    if point_ids:
                        options["ids"] = point_ids
                    
                    if filter_value not in (None, "", {}):
                        if isinstance(filter_value, str):
                            try:
                                options["filter"] = json.loads(filter_value)
                            except json.JSONDecodeError:
                                yield self.create_json_message({
                                    "provider": self.PROVIDER,
                                    "operation": "query",
                                    "status": "error",
                                    "message": f"Invalid filter JSON format: {filter_value[:200]}",
                                })
                                return
                        else:
                            options["filter"] = filter_value
                    
                    if "limit" in tool_parameters and tool_parameters.get("limit") is not None:
                        options["limit"] = int(tool_parameters.get("limit", 10))
                    if "with_payload" in tool_parameters:
                        options["with_payload"] = bool(tool_parameters.get("with_payload", True))
                    if "with_vector" in tool_parameters:
                        options["with_vectors"] = bool(tool_parameters.get("with_vector", False))
                    
                    items = []
                    yield from self._handle_retrieve_points(base_url, items, options, headers)
                    return
                
                vector = None
                if vector_is_list:
                    vector = vector_value
                elif vector_str:
                    try:
                        if vector_str.startswith("[") and vector_str.endswith("]"):
                            vector = json.loads(vector_str)
                        else:
                            vector = [float(x.strip()) for x in vector_str.split(",") if x.strip()]
                    except (ValueError, json.JSONDecodeError):
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "query",
                            "status": "error",
                            "message": "Invalid vector format. Use comma-separated values or JSON array.",
                        })
                        return
                elif text_str:
                    embedding_model_config = tool_parameters.get("embedding_model_config")
                    if not embedding_model_config:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "query",
                            "status": "error",
                            "message": "Please select an Embedding Model in the tool's USER SETTINGS to generate vector from text.",
                        })
                        return
                    
                    try:
                        vectors = self._generate_embeddings(embedding_model_config, [text_str])
                        if vectors and len(vectors) > 0:
                            vector = vectors[0]
                    except Exception as e:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "query",
                            "status": "error",
                            "message": f"Failed to generate embedding: {str(e)}",
                        })
                        return
                else:
                    yield self.create_json_message({
                        "provider": self.PROVIDER,
                        "operation": "query",
                        "status": "error",
                        "message": "Either 'text' or 'vector' must be provided.",
                    })
                    return
                
                if not vector:
                    yield self.create_json_message({
                        "provider": self.PROVIDER,
                        "operation": "query",
                        "status": "error",
                        "message": "No query vector generated. Please check your input.",
                    })
                    return
                
                options = {
                    "collection": collection,
                    "vector": vector,
                }
                if "limit" in tool_parameters and tool_parameters.get("limit") is not None:
                    options["limit"] = int(tool_parameters.get("limit", 10))
                if "with_payload" in tool_parameters:
                    options["with_payload"] = bool(tool_parameters.get("with_payload", True))
                if "with_vector" in tool_parameters:
                    options["with_vectors"] = bool(tool_parameters.get("with_vector", False))
                if filter_value not in (None, "", {}):
                    if isinstance(filter_value, str):
                        try:
                            parsed_filter = json.loads(filter_value)
                            if isinstance(parsed_filter, dict):
                                options["filter"] = parsed_filter
                            else:
                                options["filter"] = {}
                        except json.JSONDecodeError:
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "query",
                                "status": "error",
                                "message": f"Invalid filter JSON format.",
                            })
                            return
                    elif isinstance(filter_value, dict):
                        options["filter"] = filter_value
                    else:
                        options["filter"] = {}
                else:
                    options["filter"] = {}
                if "score_threshold" in tool_parameters and tool_parameters.get("score_threshold") is not None:
                    options["score_threshold"] = float(tool_parameters.get("score_threshold"))
                if "params" in tool_parameters and tool_parameters.get("params"):
                    params_value = tool_parameters.get("params")
                    if isinstance(params_value, str):
                        try:
                            options["params"] = json.loads(params_value)
                        except json.JSONDecodeError:
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "query",
                                "status": "error",
                                "message": f"Invalid params JSON format: {params_value[:200]}",
                            })
                            return
                    else:
                        options["params"] = params_value
                
                items = []
            yield from self._handle_query(base_url, items, options, headers)
            return
        if operation == "hybrid_search":
            # Hybrid search using Qdrant 1.10+ Query API
            if "collection" in tool_parameters:
                # New format: simple fields
                collection = tool_parameters.get("collection", "").strip()
                text_str = (tool_parameters.get("text") or "").strip()
                dense_vector_str = (tool_parameters.get("dense_vector") or "").strip()
                sparse_vector_str = (tool_parameters.get("sparse_vector") or "").strip()
                
                # At least one of text, dense_vector, or sparse_vector must be provided
                if not text_str and not dense_vector_str and not sparse_vector_str:
                    yield self.create_json_message({
                        "provider": self.PROVIDER,
                        "operation": "hybrid_search",
                        "status": "error",
                        "message": "At least one of 'text', 'dense_vector', or 'sparse_vector' must be provided.",
                    })
                    return
                
                options = {"collection": collection}
                
                # Parse dense vector if provided
                dense_vector = None
                if dense_vector_str:
                    try:
                        if dense_vector_str.startswith("[") and dense_vector_str.endswith("]"):
                            dense_vector = json.loads(dense_vector_str)
                        else:
                            dense_vector = [float(x.strip()) for x in dense_vector_str.split(",") if x.strip()]
                    except (ValueError, json.JSONDecodeError):
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "hybrid_search",
                            "status": "error",
                            "message": f"Invalid dense_vector format. Use comma-separated values or JSON array.",
                        })
                        return
                elif text_str:
                    # Generate dense vector from text using embedding model config
                    embedding_model_config = tool_parameters.get("embedding_model_config")
                    
                    # Check if embedding model is configured (form: form parameter)
                    if not embedding_model_config:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "hybrid_search",
                            "status": "error",
                            "message": "Please select an Embedding Model in the tool's USER SETTINGS to generate vector from text.",
                        })
                        return
                    
                    try:
                        vectors = self._generate_embeddings(embedding_model_config, [text_str])
                        if vectors and len(vectors) > 0:
                            dense_vector = vectors[0]
                        else:
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "hybrid_search",
                                "status": "error",
                                "message": "Failed to generate dense vector: embedding service returned empty result.",
                            })
                            return
                    except Exception as e:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "hybrid_search",
                            "status": "error",
                            "message": f"Failed to generate dense vector: {str(e)}",
                        })
                        return
                
                # Parse sparse vector if provided
                sparse_vector = None
                if sparse_vector_str:
                    try:
                        sparse_vector = json.loads(sparse_vector_str) if isinstance(sparse_vector_str, str) else sparse_vector_str
                        if not isinstance(sparse_vector, dict) or "indices" not in sparse_vector or "values" not in sparse_vector:
                            raise ValueError("Sparse vector must have 'indices' and 'values' keys")
                    except (ValueError, json.JSONDecodeError) as e:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "hybrid_search",
                            "status": "error",
                            "message": f"Invalid sparse_vector format: {str(e)}. Expected JSON with 'indices' and 'values' arrays.",
                        })
                        return
                elif text_str:
                    # For now, we'll let Qdrant handle sparse vector generation from text
                    # Qdrant 1.10+ may support server-side sparse vector generation
                    # We'll pass the text and let Qdrant handle it
                    pass
                
                # Build options - only add vectors, never text
                if dense_vector:
                    # Ensure dense_vector is a list, not None or empty
                    if isinstance(dense_vector, list) and len(dense_vector) > 0:
                        options["dense_vector"] = dense_vector
                    else:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "hybrid_search",
                            "status": "error",
                            "message": f"Invalid dense_vector: expected non-empty list, got {type(dense_vector).__name__} with length {len(dense_vector) if isinstance(dense_vector, list) else 'N/A'}.",
                        })
                        return
                if sparse_vector:
                    options["sparse_vector"] = sparse_vector
                
                # Validate: at least one vector must be provided
                if not options.get("dense_vector") and not options.get("sparse_vector"):
                    yield self.create_json_message({
                        "provider": self.PROVIDER,
                        "operation": "hybrid_search",
                        "status": "error",
                        "message": "At least one of 'dense_vector' or 'sparse_vector' must be provided. "
                        "If using 'text' input, ensure 'Embedding Model' is selected to generate dense_vector.",
                    })
                    return
                
                # Don't add text to options - we generate dense_vector on client side
                # Qdrant Query API doesn't support text directly (requires vectorizer)
                # We only pass pre-computed vectors to Qdrant
                
                # Add other parameters
                if "using_dense" in tool_parameters:
                    options["using_dense"] = tool_parameters.get("using_dense", "").strip() or None
                if "using_sparse" in tool_parameters:
                    options["using_sparse"] = tool_parameters.get("using_sparse", "").strip() or None
                if "fusion_method" in tool_parameters:
                    options["fusion_method"] = tool_parameters.get("fusion_method", "rrf").strip() or None
                if "limit" in tool_parameters and tool_parameters.get("limit") is not None:
                    options["limit"] = int(tool_parameters.get("limit", 10))
                if "prefetch_limit" in tool_parameters and tool_parameters.get("prefetch_limit") is not None:
                    options["prefetch_limit"] = int(tool_parameters.get("prefetch_limit", 20))
                if "filter" in tool_parameters:
                    filter_value = tool_parameters.get("filter")
                    if filter_value and isinstance(filter_value, str) and filter_value.strip():
                        try:
                            parsed_filter = json.loads(filter_value)
                            # Ensure parsed filter is a dict
                            if isinstance(parsed_filter, dict):
                                options["filter"] = parsed_filter
                            else:
                                options["filter"] = {}
                        except json.JSONDecodeError:
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "hybrid_search",
                                "status": "error",
                                "message": f"Invalid filter JSON format.",
                            })
                            return
                    elif isinstance(filter_value, dict):
                        options["filter"] = filter_value
                    else:
                        # Empty string, None, or invalid type - use empty dict
                        options["filter"] = {}
                if "with_payload" in tool_parameters:
                    options["with_payload"] = bool(tool_parameters.get("with_payload", True))
                if "with_vector" in tool_parameters:
                    options["with_vectors"] = bool(tool_parameters.get("with_vector", False))
                # MMR parameters
                if "mmr_diversity" in tool_parameters and tool_parameters.get("mmr_diversity") is not None:
                    try:
                        mmr_diversity_float = float(tool_parameters.get("mmr_diversity"))
                        if not (0.0 <= mmr_diversity_float <= 1.0):
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "hybrid_search",
                                "status": "error",
                                "message": f"Invalid mmr_diversity: {mmr_diversity_float}. Must be between 0.0 and 1.0.",
                            })
                            return
                        options["mmr_diversity"] = mmr_diversity_float
                    except (ValueError, TypeError):
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "hybrid_search",
                            "status": "error",
                            "message": f"Invalid mmr_diversity value: {tool_parameters.get('mmr_diversity')}. Must be a number between 0.0 and 1.0.",
                        })
                        return
                if "mmr_candidates_limit" in tool_parameters and tool_parameters.get("mmr_candidates_limit") is not None:
                    options["mmr_candidates_limit"] = int(tool_parameters.get("mmr_candidates_limit", 100))
                
                items = []
            else:
                # Legacy format: use options from parsed JSON
                pass
            
            yield from self._handle_hybrid_search(base_url, items, options, headers)
            return
        if operation == "recommend":
            yield from self._handle_recommend(base_url, items, options, headers)
            return
        if operation == "query" and "collection" in tool_parameters and "point_ids" in tool_parameters:
            # Data management query: retrieve points by IDs
            # Support both new simple fields and legacy JSON format
            if "collection" in tool_parameters:
                # New format: simple fields
                collection = tool_parameters.get("collection", "").strip()
                point_ids_str = tool_parameters.get("point_ids", "").strip()
                filter_value = tool_parameters.get("filter")
                
                # Support both IDs and filter (can use both together)
                if not point_ids_str and not filter_value:
                    yield self.create_json_message({
                        "provider": self.PROVIDER,
                        "operation": "query",
                        "status": "error",
                        "message": "Either 'point_ids' or 'filter' (or both) is required for query operation.",
                    })
                    return
                
                options = {"collection": collection}
                
                # Parse point IDs if provided
                if point_ids_str:
                    try:
                        # Try to parse as JSON array first
                        try:
                            point_ids = json.loads(point_ids_str)
                            if not isinstance(point_ids, list):
                                raise ValueError("Point IDs must be a JSON array")
                        except json.JSONDecodeError:
                            # Fall back to comma-separated format
                            point_ids = []
                            for id_str in point_ids_str.split(","):
                                id_str = id_str.strip()
                                if id_str:
                                    try:
                                        point_ids.append(int(id_str))
                                    except ValueError:
                                        point_ids.append(id_str)  # Keep as string (UUID)
                        options["ids"] = point_ids
                    except Exception as e:
                        yield self.create_json_message({
                            "provider": self.PROVIDER,
                            "operation": "query",
                            "status": "error",
                            "message": f"Invalid point IDs format: {str(e)}. Use JSON array (e.g., '[1,2,3]') or comma-separated (e.g., '1,2,3').",
                        })
                        return
                
                # Parse filter if provided
                if filter_value:
                    if isinstance(filter_value, str):
                        try:
                            options["filter"] = json.loads(filter_value)
                        except json.JSONDecodeError:
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "query",
                                "status": "error",
                                "message": f"Invalid filter JSON format: {filter_value[:200]}",
                            })
                            return
                    else:
                        options["filter"] = filter_value
                
                if "limit" in tool_parameters and tool_parameters.get("limit") is not None:
                    options["limit"] = int(tool_parameters.get("limit", 10))
                if "with_payload" in tool_parameters:
                    options["with_payload"] = bool(tool_parameters.get("with_payload", True))
                if "with_vector" in tool_parameters:
                    options["with_vectors"] = bool(tool_parameters.get("with_vector", False))
                
                items = []
            # Legacy format: continue with existing logic
            yield from self._handle_retrieve_points(base_url, items, options, headers)
            return
        if operation == "scroll":
            # Support both new simple fields and legacy JSON format
            if "collection" in tool_parameters:
                # New format: simple fields
                collection = tool_parameters.get("collection", "").strip()
                options = {"collection": collection}
                if "limit" in tool_parameters and tool_parameters.get("limit") is not None:
                    options["limit"] = int(tool_parameters.get("limit", 10))
                if "filter" in tool_parameters and tool_parameters.get("filter"):
                    # Parse filter JSON string if provided as string
                    filter_value = tool_parameters.get("filter")
                    if isinstance(filter_value, str):
                        try:
                            options["filter"] = json.loads(filter_value)
                        except json.JSONDecodeError:
                            yield self.create_json_message({
                                "provider": self.PROVIDER,
                                "operation": "scroll",
                                "status": "error",
                                "message": f"Invalid filter JSON format: {filter_value[:200]}",
                            })
                            return
                    else:
                        options["filter"] = filter_value
                if "with_payload" in tool_parameters:
                    options["with_payload"] = bool(tool_parameters.get("with_payload", True))
                if "with_vector" in tool_parameters:
                    options["with_vectors"] = bool(tool_parameters.get("with_vector", False))
                items = []
            # Legacy format: continue with existing logic
            yield from self._handle_scroll(base_url, items, options, headers)
            return
        if operation == "delete":
            # Support both new simple fields and legacy JSON format
            if "collection" in tool_parameters:
                # New format: simple fields
                collection = tool_parameters.get("collection", "").strip()
                point_ids_str = tool_parameters.get("point_ids", "").strip()
                
                if not point_ids_str:
                    yield self.create_json_message({
                        "provider": self.PROVIDER,
                        "operation": "delete",
                        "status": "error",
                        "message": "Point IDs are required for delete operation. Provide comma-separated IDs (e.g., '1,2,3').",
                    })
                    return
                
                # Parse comma-separated IDs
                try:
                    point_ids = []
                    for id_str in point_ids_str.split(","):
                        id_str = id_str.strip()
                        if id_str:
                            try:
                                point_ids.append(int(id_str))
                            except ValueError:
                                point_ids.append(id_str)  # Keep as string (UUID)
                except Exception:
                    yield self.create_json_message({
                        "provider": self.PROVIDER,
                        "operation": "delete",
                        "status": "error",
                        "message": "Invalid point IDs format. Use comma-separated values (e.g., '1,2,3').",
                    })
                    return
                
                options = {
                    "collection": collection,
                    "points": point_ids,
                }
                items = []
            # Legacy format: continue with existing logic
            yield from self._handle_delete(base_url, items, options, headers)
            return
        if operation == "create_collection":
            # Support both new simple fields and legacy JSON format
            if "collection" in tool_parameters:
                # New format: simple fields
                collection = tool_parameters.get("collection", "").strip()
                vector_size = tool_parameters.get("vector_size")
                distance = self._normalize_distance(tool_parameters.get("distance"))
                
                # Get defaults from credentials if not provided
                credentials = self.runtime.credentials or {}
                if vector_size is None:
                    vector_size = credentials.get("default_vector_size")
                    # Handle string format from credentials_for_provider
                    if vector_size is not None:
                        try:
                            vector_size = int(vector_size) if isinstance(vector_size, str) else vector_size
                        except (ValueError, TypeError):
                            vector_size = None
                if not distance:
                    distance = self._normalize_distance(credentials.get("default_distance", "Cosine"))
                
                if vector_size is None:
                    yield self.create_json_message({
                        "provider": self.PROVIDER,
                        "operation": "create_collection",
                        "status": "error",
                        "message": "Vector size is required. Provide it in the node or set 'Default Vector Dimensions' in Provider settings.",
                    })
                    return
                
                options = {
                    "collection": collection,
                    "vector_size": int(vector_size),
                    "distance": distance,
                }
                items = []
            # Legacy format: continue with existing logic
            yield from self._handle_create_collection(base_url, items, options, headers)
            return
        if operation == "delete_collection":
            # Support both new simple fields and legacy JSON format
            if "collection" in tool_parameters:
                # New format: simple fields
                collection = tool_parameters.get("collection", "").strip()
                options = {"collection": collection}
                items = []
            # Legacy format: continue with existing logic
            yield from self._handle_delete_collection(base_url, items, options, headers)
            return
        if operation == "get_collection_info":
            # Support both new simple fields and legacy JSON format
            if "collection" in tool_parameters:
                # New format: simple fields
                collection = tool_parameters.get("collection", "").strip()
                options = {"collection": collection}
                items = []
            # Legacy format: continue with existing logic
            yield from self._handle_get_collection_info(base_url, items, options, headers)
            return

        yield self.create_json_message(
            {
                "provider": self.PROVIDER,
                "operation": operation,
                "inputs": items,
                "options": options,
                "status": "error",
                "message": f"Unsupported operation `{operation}`. Expected one of embed/upsert/query/recommend/scroll/delete/create_collection/delete_collection/get_collection_info.",
            }
        )
        return

    def _handle_embed(
        self,
        base_url: str,
        items: list[dict[str, Any]],
        options: dict[str, Any],
        headers: dict[str, str],
        credentials: dict[str, Any],
    ) -> Generator[ToolInvokeMessage]:
        vectorizer = options.get("vectorizer") or credentials.get("default_vectorizer")
        if not vectorizer:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "status": "error",
                    "message": "No vectorizer specified. Set `options.vectorizer` or credential `default_vectorizer`.",
                }
            )
            return

        endpoint = options.get("endpoint")
        if endpoint:
            endpoint = resolve_endpoint(base_url, endpoint)
        else:
            endpoint = resolve_endpoint(base_url, f"/v1/vectorizers/{vectorizer}/embed")

        texts: list[str] = []
        for entry in items:
            text = entry.get("text")
            if text is None:
                yield self.create_json_message(
                    {
                        "provider": self.PROVIDER,
                        "status": "error",
                        "operation": "embed",
                        "message": "Each item must include `text` when operation=embed.",
                    }
                )
                return
            texts.append(str(text))
        body: dict[str, Any] = {"input": texts}

        model_options = {}
        for key in ("model", "truncate", "dimensions", "input_type"):
            if key in options:
                model_options[key] = options[key]
        if "parameters" in options and isinstance(options["parameters"], dict):
            model_options.setdefault("parameters", options["parameters"])
        body.update(model_options)

        response, error_message = self._request("POST", endpoint, headers=headers, json=body)
        if error_message:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "status": "error",
                    "operation": "embed",
                    "inputs": items,
                    "options": options,
                    "message": error_message,
                }
            )
            return

        result_payload = self._extract_embeddings(response.json(), items)

        yield self.create_json_message(
            {
                "provider": self.PROVIDER,
                "operation": "embed",
                "inputs": items,
                "options": options,
                "endpoint": endpoint,
                "result": result_payload,
                "status": "success",
                "message": "Embedding generated successfully.",
            }
        )

    def _handle_upsert(
        self,
        base_url: str,
        items: list[dict[str, Any]],
        options: dict[str, Any],
        headers: dict[str, str],
    ) -> Generator[ToolInvokeMessage]:
        collection = options.get("collection")
        if not collection:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "upsert",
                    "status": "error",
                    "message": "Missing `options.collection` for upsert operation.",
                }
            )
            return

        if not items:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "upsert",
                    "status": "error",
                    "message": "At least one item is required for upsert.",
                }
            )
            return

        points: list[dict[str, Any]] = []
        vector_name = options.get("vector_name")
        for entry in items:
            point_id = entry.get("id")
            vector = entry.get("vector")
            if not point_id:
                yield self.create_json_message(
                    {
                        "provider": self.PROVIDER,
                        "operation": "upsert",
                        "status": "error",
                        "message": "Each item must include a non-empty `id` for upsert.",
                    }
                )
                return
            if vector is None:
                yield self.create_json_message(
                    {
                        "provider": self.PROVIDER,
                        "operation": "upsert",
                        "status": "error",
                        "message": f"Item `{point_id}` is missing `vector`. Provide vectors before upserting.",
                    }
                )
                return
            if not isinstance(vector, list):
                yield self.create_json_message(
                    {
                        "provider": self.PROVIDER,
                        "operation": "upsert",
                        "status": "error",
                        "message": f"Item `{point_id}` has invalid vector type. Expected list of floats.",
                    }
                )
                return

            payload = entry.get("payload")
            if payload is None:
                payload = entry.get("metadata", {})
            point: dict[str, Any] = {"id": point_id, "payload": payload}
            if vector_name:
                point["vectors"] = {vector_name: vector}
            else:
                point["vector"] = vector
            points.append(point)

        endpoint_override = options.get("endpoint")
        if endpoint_override:
            endpoint = resolve_endpoint(base_url, endpoint_override)
            fallback_endpoint = None
        else:
            endpoint = resolve_endpoint(base_url, f"/v1/collections/{collection}/points")
            fallback_endpoint = resolve_endpoint(base_url, f"/collections/{collection}/points")

        params = {}
        if "wait" in options:
            params["wait"] = str(bool(options["wait"])).lower()

        body = {"points": points}
        if options.get("write_ordering"):
            body["write_ordering"] = options["write_ordering"]
        
        response, error_message = self._request(
            "PUT", endpoint, headers=headers, json=body, params=params, allow_404_retry=False
        )

        # Try fallback endpoint if 404
        if (
            error_message
            and "HTTP 404" in error_message
            and fallback_endpoint
            and not endpoint_override
        ):
            response, error_message = self._request(
                "PUT", fallback_endpoint, headers=headers, json=body, params=params, allow_404_retry=False
            )

        # Auto-create collection if it doesn't exist (404 error)
        # Check for 404 in error message (could be "HTTP 404" or "HTTP 404: ...")
        if error_message and ("HTTP 404" in error_message or "404" in error_message):
            # Get vector dimension from first point
            first_vector = points[0].get("vector") or (
                list(points[0].get("vectors", {}).values())[0] if points[0].get("vectors") else None
            )
            if first_vector and isinstance(first_vector, list):
                vector_size = len(first_vector)
                # Auto-create collection with default settings
                create_result = self._auto_create_collection(
                    base_url, collection, vector_size, options, headers
                )
                if create_result:
                    # Retry upsert after creating collection
                    response, error_message = self._request(
                        "PUT", endpoint, headers=headers, json=body, params=params, allow_404_retry=False
                    )
                    if (
                        error_message
                        and ("HTTP 404" in error_message or "404" in error_message)
                        and fallback_endpoint
                        and not endpoint_override
                    ):
                        response, error_message = self._request(
                            "PUT",
                            fallback_endpoint,
                            headers=headers,
                            json=body,
                            params=params,
                            allow_404_retry=False,
                        )

        if error_message:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "upsert",
                    "inputs": items,
                    "options": options,
                    "status": "error",
                    "message": error_message,
                }
            )
            return

        # Process result to remove vectors (too long for display)
        result_data = response.json() if response else {}
        
        # Remove vectors from result if present
        if isinstance(result_data, dict):
            # Remove vector from result summary if present
            if "result" in result_data:
                result_summary = result_data["result"]
                if isinstance(result_summary, dict):
                    # Remove vector-related fields
                    result_summary.pop("vectors", None)
                    result_summary.pop("vector", None)
        
        # Also remove vectors from inputs summary
        items_summary = []
        for item in items:
            item_summary = {
                "id": item.get("id"),
                "payload": item.get("payload"),
            }
            # Add vector dimension info instead of full vector
            if "vector" in item:
                item_summary["vector_dimension"] = len(item["vector"]) if isinstance(item["vector"], list) else None
            elif "vectors" in item:
                # Handle named vectors
                vectors_info = {}
                for name, vec in item["vectors"].items():
                    vectors_info[name] = len(vec) if isinstance(vec, list) else None
                item_summary["vectors_dimensions"] = vectors_info
            items_summary.append(item_summary)

        yield self.create_json_message(
            {
                "provider": self.PROVIDER,
                "operation": "upsert",
                "inputs": items_summary,  # Use summary without vectors
                "options": options,
                "endpoint": endpoint,
                "result": result_data,
                "status": "success",
                "message": f"Points upserted successfully. {len(items)} point(s) inserted/updated.",
            }
        )

    def _handle_query(
        self,
        base_url: str,
        items: list[dict[str, Any]],
        options: dict[str, Any],
        headers: dict[str, str],
    ) -> Generator[ToolInvokeMessage]:
        collection = options.get("collection")
        if not collection:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "query",
                    "status": "error",
                    "message": "Missing `options.collection` for query operation.",
                }
            )
            return

        search_vector = options.get("vector")
        if search_vector is None and items:
            search_vector = items[0].get("vector")
        if search_vector is None:
            yield self.create_json_message({
                "provider": self.PROVIDER,
                "operation": "query",
                "status": "error",
                "message": "Provide search vector via `options.vector` or first item's `vector`.",
            })
            return
        if not isinstance(search_vector, list):
            yield self.create_json_message({
                "provider": self.PROVIDER,
                "operation": "query",
                "status": "error",
                "message": "Search vector must be a list of floats.",
            })
            return

        endpoint_override = options.get("endpoint")
        if endpoint_override:
            endpoint = resolve_endpoint(base_url, endpoint_override)
            fallback_endpoint = None
        else:
            endpoint = resolve_endpoint(base_url, f"/v1/collections/{collection}/points/search")
            fallback_endpoint = resolve_endpoint(base_url, f"/collections/{collection}/points/search")

        body: dict[str, Any] = {
            "vector": search_vector,
            "limit": options.get("limit", 5),
        }
        if "score_threshold" in options:
            body["score_threshold"] = options["score_threshold"]
        if "offset" in options:
            body["offset"] = options["offset"]
        # Qdrant API requires filter to be a dictionary (even if empty)
        body["filter"] = options.get("filter", {}) if isinstance(options.get("filter"), dict) else {}
        if "with_payload" in options:
            body["with_payload"] = options["with_payload"]
        else:
            body["with_payload"] = True
        if "with_vectors" in options:
            body["with_vectors"] = options["with_vectors"]

        response, error_message = self._request(
            "POST", endpoint, headers=headers, json=body, allow_404_retry=False
        )

        if (
            error_message
            and "HTTP 404" in error_message
            and fallback_endpoint
            and not endpoint_override
        ):
            response, error_message = self._request(
                "POST", fallback_endpoint, headers=headers, json=body, allow_404_retry=False
            )
        if error_message:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "query",
                    "inputs": items,
                    "options": options,
                    "status": "error",
                    "message": error_message,
                }
            )
            return

        result_data = response.json()
        hits: list[dict[str, Any]] = []
        for result in result_data.get("result", []):
            hits.append({
                "id": result.get("id"),
                "score": result.get("score"),
                "payload": result.get("payload"),
                # Exclude vector from results (too long for display)
            })

        # Remove vector from options for cleaner output
        options_summary = {k: v for k, v in options.items() if k != "vector"}
        
        yield self.create_json_message({
            "provider": self.PROVIDER,
            "operation": "query",
            "options": options_summary,
            "endpoint": endpoint,
            "result": hits,
            "status": "success",
            "message": f"Query executed successfully. Found {len(hits)} results.",
        })

    def _handle_recommend(
        self,
        base_url: str,
        items: list[dict[str, Any]],
        options: dict[str, Any],
        headers: dict[str, str],
    ) -> Generator[ToolInvokeMessage]:
        collection = options.get("collection")
        if not collection:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "recommend",
                    "status": "error",
                    "message": "Missing `options.collection` for recommend operation.",
                }
            )
            return

        positive = options.get("positive", [])
        negative = options.get("negative", [])

        if not positive:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "recommend",
                    "status": "error",
                    "message": "At least one positive example is required for recommend operation.",
                }
            )
            return

        endpoint_override = options.get("endpoint")
        if endpoint_override:
            endpoint = resolve_endpoint(base_url, endpoint_override)
            fallback_endpoint = None
        else:
            endpoint = resolve_endpoint(base_url, f"/v1/collections/{collection}/points/recommend")
            fallback_endpoint = resolve_endpoint(base_url, f"/collections/{collection}/points/recommend")

        body: dict[str, Any] = {
            "positive": positive,
            "limit": options.get("limit", 10),
        }

        if negative:
            body["negative"] = negative

        if "filter" in options:
            body["filter"] = options["filter"]
        if "score_threshold" in options:
            body["score_threshold"] = options["score_threshold"]
        if "offset" in options:
            body["offset"] = options["offset"]
        if "using" in options:
            body["using"] = options["using"]
        if "strategy" in options:
            body["strategy"] = options["strategy"]
        if "with_payload" in options:
            body["with_payload"] = options["with_payload"]
        else:
            body["with_payload"] = True
        if "with_vectors" in options:
            body["with_vectors"] = options["with_vectors"]

        response, error_message = self._request(
            "POST", endpoint, headers=headers, json=body, allow_404_retry=False
        )

        if (
            error_message
            and "HTTP 404" in error_message
            and fallback_endpoint
            and not endpoint_override
        ):
            response, error_message = self._request(
                "POST", fallback_endpoint, headers=headers, json=body, allow_404_retry=False
            )
        if error_message:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "recommend",
                    "inputs": items,
                    "options": options,
                    "status": "error",
                    "message": error_message,
                }
            )
            return

        result_data = response.json()
        hits: list[dict[str, Any]] = []
        for result in result_data.get("result", []):
            hits.append(
                {
                    "id": result.get("id"),
                    "score": result.get("score"),
                    "payload": result.get("payload"),
                    "vector": result.get("vector"),
                }
            )

        yield self.create_json_message(
            {
                "provider": self.PROVIDER,
                "operation": "recommend",
                "inputs": items,
                "options": options,
                "endpoint": endpoint,
                "result": hits,
                "status": "success",
                "message": "Recommendations generated successfully.",
            }
        )

    def _handle_scroll(
        self,
        base_url: str,
        items: list[dict[str, Any]],
        options: dict[str, Any],
        headers: dict[str, str],
    ) -> Generator[ToolInvokeMessage]:
        collection = options.get("collection")
        if not collection:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "scroll",
                    "status": "error",
                    "message": "Missing `options.collection` for scroll operation.",
                }
            )
            return

        endpoint_override = options.get("endpoint")
        if endpoint_override:
            endpoint = resolve_endpoint(base_url, endpoint_override)
            fallback_endpoint = None
        else:
            endpoint = resolve_endpoint(base_url, f"/v1/collections/{collection}/points/scroll")
            fallback_endpoint = resolve_endpoint(base_url, f"/collections/{collection}/points/scroll")

        body: dict[str, Any] = {
            "limit": options.get("limit", 10),
        }

        if "offset" in options:
            body["offset"] = options["offset"]
        if "filter" in options:
            body["filter"] = options["filter"]
        if "with_payload" in options:
            body["with_payload"] = options["with_payload"]
        else:
            body["with_payload"] = True
        if "with_vectors" in options:
            body["with_vectors"] = options["with_vectors"]
        if "order_by" in options:
            body["order_by"] = options["order_by"]

        response, error_message = self._request(
            "POST", endpoint, headers=headers, json=body, allow_404_retry=False
        )

        if (
            error_message
            and "HTTP 404" in error_message
            and fallback_endpoint
            and not endpoint_override
        ):
            response, error_message = self._request(
                "POST", fallback_endpoint, headers=headers, json=body, allow_404_retry=False
            )
        if error_message:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "scroll",
                    "inputs": items,
                    "options": options,
                    "status": "error",
                    "message": error_message,
                }
            )
            return

        result_data = response.json()
        points: list[dict[str, Any]] = []
        for point in result_data.get("result", {}).get("points", []):
            points.append(
                {
                    "id": point.get("id"),
                    "payload": point.get("payload"),
                    "vector": point.get("vector"),
                }
            )

        yield self.create_json_message(
            {
                "provider": self.PROVIDER,
                "operation": "scroll",
                "inputs": items,
                "options": options,
                "endpoint": endpoint,
                "result": {
                    "points": points,
                    "next_page_offset": result_data.get("result", {}).get("next_page_offset"),
                },
                "status": "success",
                "message": "Scroll executed successfully.",
            }
        )

    def _handle_retrieve_points(
        self,
        base_url: str,
        items: list[dict[str, Any]],
        options: dict[str, Any],
        headers: dict[str, str],
    ) -> Generator[ToolInvokeMessage]:
        """Retrieve points by IDs or filter from Qdrant collection."""
        collection = options.get("collection")
        if not collection:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "query",
                    "status": "error",
                    "message": "Missing `options.collection` for retrieve points operation.",
                }
            )
            return

        endpoint_override = options.get("endpoint")
        if endpoint_override:
            endpoint = resolve_endpoint(base_url, endpoint_override)
            fallback_endpoint = None
        else:
            # Qdrant API: POST /collections/{collection_name}/points
            endpoint = resolve_endpoint(base_url, f"/collections/{collection}/points")
            fallback_endpoint = resolve_endpoint(base_url, f"/v1/collections/{collection}/points")

        body: dict[str, Any] = {}

        # Support both IDs-based and filter-based retrieval
        # Can use both: retrieve by IDs and then apply filter
        if "ids" in options:
            body["ids"] = options["ids"]
            # If filter is also provided, apply it to filter the retrieved points
            if "filter" in options:
                body["filter"] = options["filter"]
        elif "filter" in options:
            # If filter is provided, use scroll endpoint with filter
            scroll_endpoint = resolve_endpoint(base_url, f"/collections/{collection}/points/scroll")
            scroll_fallback = resolve_endpoint(base_url, f"/v1/collections/{collection}/points/scroll")
            
            scroll_body: dict[str, Any] = {
                "filter": options["filter"],
                "limit": options.get("limit", 10),
            }
            if "with_payload" in options:
                scroll_body["with_payload"] = options["with_payload"]
            else:
                scroll_body["with_payload"] = True
            if "with_vectors" in options:
                scroll_body["with_vectors"] = options["with_vectors"]
            else:
                scroll_body["with_vectors"] = options.get("with_vector", False)
            
            response, error_message = self._request(
                "POST", scroll_endpoint, headers=headers, json=scroll_body, allow_404_retry=False
            )
            
            if (
                error_message
                and "HTTP 404" in error_message
                and scroll_fallback
            ):
                response, error_message = self._request(
                    "POST", scroll_fallback, headers=headers, json=scroll_body, allow_404_retry=False
                )
            
            if error_message:
                yield self.create_json_message(
                    {
                        "provider": self.PROVIDER,
                        "operation": "query",
                        "inputs": items,
                        "options": options,
                        "status": "error",
                        "message": error_message,
                    }
                )
                return

            result_data = response.json() if response else {}
            points = result_data.get("result", {}).get("points", [])
            
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "query",
                    "inputs": items,
                    "options": options,
                    "endpoint": scroll_endpoint,
                    "result": points,
                    "status": "success",
                    "message": f"Retrieved {len(points)} points using filter.",
                }
            )
            return
        else:
            # If neither ids nor filter provided, return error
            # But this should be caught earlier in parameter processing
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "query",
                    "status": "error",
                    "message": "Either 'ids' or 'filter' is required for retrieve points operation.",
                }
            )
            return

        # Retrieve by IDs
        if "with_payload" in options:
            body["with_payload"] = options["with_payload"]
        else:
            body["with_payload"] = True
        if "with_vectors" in options:
            body["with_vectors"] = options["with_vectors"]
        elif "with_vector" in options:
            body["with_vectors"] = options["with_vector"]
        else:
            body["with_vectors"] = False

        response, error_message = self._request(
            "POST", endpoint, headers=headers, json=body, allow_404_retry=False
        )

        if (
            error_message
            and "HTTP 404" in error_message
            and fallback_endpoint
            and not endpoint_override
        ):
            response, error_message = self._request(
                "POST", fallback_endpoint, headers=headers, json=body, allow_404_retry=False
            )
        
        if error_message:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "query",
                    "inputs": items,
                    "options": options,
                    "status": "error",
                    "message": error_message,
                }
            )
            return

        result_data = response.json() if response else {}

        raw_result = result_data.get("result")
        normalized_points: list[dict[str, Any]] = []

        if isinstance(raw_result, dict):
            points_iterable = raw_result.get("points") or raw_result.get("items") or []
        elif isinstance(raw_result, list):
            points_iterable = raw_result
        else:
            points_iterable = []

        for point in points_iterable:
            if not isinstance(point, dict):
                continue
            normalized_points.append(
                {
                    "id": point.get("id"),
                    "payload": point.get("payload"),
                    "vector": point.get("vector") if options.get("with_vectors") else None,
                }
            )
        
        yield self.create_json_message(
            {
                "provider": self.PROVIDER,
                "operation": "query",
                "inputs": items,
                "options": options,
                "endpoint": endpoint,
                "result": normalized_points,
                "status": "success",
                "message": f"Retrieved {len(normalized_points)} points by IDs.",
            }
        )

    def _handle_delete(
        self,
        base_url: str,
        items: list[dict[str, Any]],
        options: dict[str, Any],
        headers: dict[str, str],
    ) -> Generator[ToolInvokeMessage]:
        collection = options.get("collection")
        if not collection:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "delete",
                    "status": "error",
                    "message": "Missing `options.collection` for delete operation.",
                }
            )
            return

        endpoint_override = options.get("endpoint")
        if endpoint_override:
            endpoint = resolve_endpoint(base_url, endpoint_override)
            fallback_endpoint = None
        else:
            endpoint = resolve_endpoint(base_url, f"/v1/collections/{collection}/points/delete")
            fallback_endpoint = resolve_endpoint(base_url, f"/collections/{collection}/points/delete")

        body: dict[str, Any] = {}

        # Support both filter-based and points-based deletion
        # Qdrant API requires "points" field (not "ids") for point IDs
        if "filter" in options:
            body["filter"] = options["filter"]
        elif "points" in options:
            body["points"] = options["points"]
        elif "ids" in options:
            # Legacy support: convert "ids" to "points"
            body["points"] = options["ids"]
        elif items:
            # Extract IDs from items if provided
            ids = [item.get("id") for item in items if item.get("id")]
            if ids:
                body["points"] = ids
            else:
                yield self.create_json_message(
                    {
                        "provider": self.PROVIDER,
                        "operation": "delete",
                        "status": "error",
                        "message": "Provide `options.filter`, `options.points`, or items with `id` field for delete operation.",
                    }
                )
                return
        else:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "delete",
                    "status": "error",
                    "message": "Provide `options.filter`, `options.points`, or items with `id` field for delete operation.",
                }
            )
            return

        if "wait" in options:
            body["wait"] = options["wait"]
        if "ordering" in options:
            body["ordering"] = options["ordering"]

        response, error_message = self._request(
            "POST", endpoint, headers=headers, json=body, allow_404_retry=False
        )

        if (
            error_message
            and "HTTP 404" in error_message
            and fallback_endpoint
            and not endpoint_override
        ):
            response, error_message = self._request(
                "POST", fallback_endpoint, headers=headers, json=body, allow_404_retry=False
            )
        if error_message:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "delete",
                    "inputs": items,
                    "options": options,
                    "status": "error",
                    "message": error_message,
                }
            )
            return

        result_data = response.json()

        yield self.create_json_message(
            {
                "provider": self.PROVIDER,
                "operation": "delete",
                "inputs": items,
                "options": options,
                "endpoint": endpoint,
                "result": result_data,
                "status": "success",
                "message": "Points deleted successfully.",
            }
        )

    def _handle_create_collection(
        self,
        base_url: str,
        items: list[dict[str, Any]],
        options: dict[str, Any],
        headers: dict[str, str],
    ) -> Generator[ToolInvokeMessage]:
        collection_name = options.get("name") or options.get("collection")
        if not collection_name:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "create_collection",
                    "status": "error",
                    "message": "Missing `options.name` or `options.collection` for create_collection operation.",
                }
            )
            return

        endpoint_override = options.get("endpoint")
        if endpoint_override:
            endpoint = resolve_endpoint(base_url, endpoint_override)
            fallback_endpoint = None
        else:
            endpoint = resolve_endpoint(base_url, f"/v1/collections/{collection_name}")
            fallback_endpoint = resolve_endpoint(base_url, f"/collections/{collection_name}")

        # Build collection configuration
        body: dict[str, Any] = {}

        # Support vectors config (required)
        if "vectors" in options:
            body["vectors"] = options["vectors"]
        elif "vector_size" in options or "distance" in options:
            # Simplified config: just size and distance
            vectors_config: dict[str, Any] = {}
            if "vector_size" in options:
                vectors_config["size"] = options["vector_size"]
            if "distance" in options:
                vectors_config["distance"] = self._normalize_distance(options["distance"])
            if vectors_config:
                vector_name = options.get("vector_name")
                body["vectors"] = {vector_name: vectors_config} if vector_name else vectors_config
        else:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "create_collection",
                    "status": "error",
                    "message": "Provide `options.vectors` or `options.vector_size` and `options.distance` for create_collection operation.",
                }
            )
            return

        # Optional parameters
        if "optimizers_config" in options:
            body["optimizers_config"] = options["optimizers_config"]
        if "hnsw_config" in options:
            body["hnsw_config"] = options["hnsw_config"]
        if "wal_config" in options:
            body["wal_config"] = options["wal_config"]
        if "quantization_config" in options:
            body["quantization_config"] = options["quantization_config"]
        if "on_disk_payload" in options:
            body["on_disk_payload"] = options["on_disk_payload"]
        if "timeout" in options:
            body["timeout"] = options["timeout"]

        response, error_message = self._request(
            "PUT", endpoint, headers=headers, json=body, allow_404_retry=False
        )

        if (
            error_message
            and "HTTP 404" in error_message
            and fallback_endpoint
            and not endpoint_override
        ):
            response, error_message = self._request(
                "PUT", fallback_endpoint, headers=headers, json=body, allow_404_retry=False
            )
        if error_message:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "create_collection",
                    "inputs": items,
                    "options": options,
                    "status": "error",
                    "message": error_message,
                }
            )
            return

        result_data = response.json()

        yield self.create_json_message(
            {
                "provider": self.PROVIDER,
                "operation": "create_collection",
                "inputs": items,
                "options": options,
                "endpoint": endpoint,
                "result": result_data,
                "status": "success",
                "message": "Collection created successfully.",
            }
        )

    def _handle_delete_collection(
        self,
        base_url: str,
        items: list[dict[str, Any]],
        options: dict[str, Any],
        headers: dict[str, str],
    ) -> Generator[ToolInvokeMessage]:
        collection_name = options.get("name") or options.get("collection")
        if not collection_name:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "delete_collection",
                    "status": "error",
                    "message": "Missing `options.name` or `options.collection` for delete_collection operation.",
                }
            )
            return

        endpoint_override = options.get("endpoint")
        if endpoint_override:
            endpoint = resolve_endpoint(base_url, endpoint_override)
            fallback_endpoint = None
        else:
            endpoint = resolve_endpoint(base_url, f"/v1/collections/{collection_name}")
            fallback_endpoint = resolve_endpoint(base_url, f"/collections/{collection_name}")

        params = {}
        if "timeout" in options:
            params["timeout"] = options["timeout"]

        response, error_message = self._request(
            "DELETE", endpoint, headers=headers, params=params, allow_404_retry=False
        )

        if (
            error_message
            and "HTTP 404" in error_message
            and fallback_endpoint
            and not endpoint_override
        ):
            response, error_message = self._request(
                "DELETE", fallback_endpoint, headers=headers, params=params, allow_404_retry=False
            )
        if error_message:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "delete_collection",
                    "inputs": items,
                    "options": options,
                    "status": "error",
                    "message": error_message,
                }
            )
            return

        result_data = response.json() if response else {}

        yield self.create_json_message(
            {
                "provider": self.PROVIDER,
                "operation": "delete_collection",
                "inputs": items,
                "options": options,
                "endpoint": endpoint,
                "result": result_data,
                "status": "success",
                "message": "Collection deleted successfully.",
            }
        )

    def _handle_get_collection_info(
        self,
        base_url: str,
        items: list[dict[str, Any]],
        options: dict[str, Any],
        headers: dict[str, str],
    ) -> Generator[ToolInvokeMessage]:
        collection_name = options.get("name") or options.get("collection")
        if not collection_name:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "get_collection_info",
                    "status": "error",
                    "message": "Missing `options.name` or `options.collection` for get_collection_info operation.",
                }
            )
            return

        endpoint_override = options.get("endpoint")
        if endpoint_override:
            endpoint = resolve_endpoint(base_url, endpoint_override)
            fallback_endpoint = None
        else:
            endpoint = resolve_endpoint(base_url, f"/v1/collections/{collection_name}")
            fallback_endpoint = resolve_endpoint(base_url, f"/collections/{collection_name}")

        response, error_message = self._request(
            "GET", endpoint, headers=headers, allow_404_retry=False
        )

        if (
            error_message
            and "HTTP 404" in error_message
            and fallback_endpoint
            and not endpoint_override
        ):
            response, error_message = self._request(
                "GET", fallback_endpoint, headers=headers, allow_404_retry=False
            )
        if error_message:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "get_collection_info",
                    "inputs": items,
                    "options": options,
                    "status": "error",
                    "message": error_message,
                }
            )
            return

        result_data = response.json()

        yield self.create_json_message(
            {
                "provider": self.PROVIDER,
                "operation": "get_collection_info",
                "inputs": items,
                "options": options,
                "endpoint": endpoint,
                "result": result_data,
                "status": "success",
                "message": "Collection info retrieved successfully.",
            }
        )

    def _auto_create_collection(
        self,
        base_url: str,
        collection_name: str,
        vector_size: int,
        options: dict[str, Any],
        headers: dict[str, str],
    ) -> bool:
        """Auto-create collection if it doesn't exist. Returns True if created successfully."""
        create_endpoint = resolve_endpoint(base_url, f"/v1/collections/{collection_name}")
        fallback_create_endpoint = resolve_endpoint(base_url, f"/collections/{collection_name}")

        # Get distance from options, or from credentials, or default to Cosine
        credentials = self.runtime.credentials or {}
        distance = self._normalize_distance(
            options.get("distance") or credentials.get("default_distance") or "Cosine"
        )

        vector_params: dict[str, Any] = {
            "size": vector_size,
            "distance": distance,
        }
        vector_name = options.get("vector_name")

        body: dict[str, Any] = {
            "vectors": {vector_name: vector_params} if vector_name else vector_params
        }

        # Optional collection config from options
        if "optimizers_config" in options:
            body["optimizers_config"] = options["optimizers_config"]
        if "hnsw_config" in options:
            body["hnsw_config"] = options["hnsw_config"]

        response, error_message = self._request(
            "PUT", create_endpoint, headers=headers, json=body, allow_404_retry=False
        )

        if error_message and "HTTP 404" in error_message and fallback_create_endpoint:
            response, error_message = self._request(
                "PUT", fallback_create_endpoint, headers=headers, json=body, allow_404_retry=False
            )

        return error_message is None

    @staticmethod
    def _parse_items(items_raw: str) -> tuple[list[dict[str, Any]], str | None]:
        try:
            items = json.loads(items_raw)
        except json.JSONDecodeError as exc:
            return [], f"Invalid items payload: {exc}"

        if not isinstance(items, list):
            return [], "items must be a JSON array"

        normalized: list[dict[str, Any]] = []
        for entry in items:
            if not isinstance(entry, dict):
                return [], "Each item must be an object (dictionary)."
            normalized.append(
                {
                    "id": entry.get("id", ""),
                    "text": entry.get("text", ""),
                    "vector": entry.get("vector"),
                    "metadata": entry.get("metadata", {}),
                    "payload": entry.get("payload"),
                }
            )
        return normalized, None

    @staticmethod
    def _parse_options(options_raw: str) -> tuple[dict[str, Any], str | None]:
        try:
            options = json.loads(options_raw) if options_raw else {}
        except json.JSONDecodeError as exc:
            return {}, f"Invalid options payload: {exc}"

        if not isinstance(options, dict):
            return {}, "options must be a JSON object"

        return options, None

    @staticmethod
    def _normalize_optional_string(value: Any, field_name: str) -> tuple[str | None, str | None]:
        if value in (None, ""):
            return None, None
        if isinstance(value, str):
            normalized = value.strip()
            return (normalized or None), None
        if isinstance(value, (int, float, bool)):
            return str(value), None
        return None, f"`{field_name}` must be a text value or a variable resolving to text."

    @staticmethod
    def _parse_optional_payload_object(value: Any) -> tuple[dict[str, Any] | None, str | None]:
        if value in (None, ""):
            return None, None

        if isinstance(value, dict):
            parsed = value
        elif isinstance(value, str):
            payload_text = value.strip()
            if not payload_text:
                return None, None
            try:
                parsed = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                return None, f"`payload` must be a valid JSON object: {exc.msg}."
        else:
            return None, "`payload` must be a JSON object or a variable resolving to a JSON object."

        if not isinstance(parsed, dict):
            return None, "`payload` must be a JSON object compatible with Qdrant payload."

        return parsed, None

    @staticmethod
    def _parse_payload_entries(value: Any) -> tuple[list[dict[str, Any]], str | None]:
        if value in (None, ""):
            return [], None

        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError as exc:
            return [], f"`payloads` must be valid JSON: {exc.msg}."

        if isinstance(parsed, dict):
            return [parsed], None
        if isinstance(parsed, list):
            if all(isinstance(item, dict) for item in parsed):
                return parsed, None
            return [], "Each entry in `payloads` must be a JSON object."

        return [], "`payloads` must be a JSON object or an array of JSON objects."

    @staticmethod
    def _normalize_distance(distance: Any) -> str:
        if distance in (None, ""):
            return ""

        normalized = str(distance).strip()
        if not normalized:
            return ""

        aliases = {
            "euclidean": "Euclid",
            "Euclidean": "Euclid",
            "euclid": "Euclid",
            "cosine": "Cosine",
            "dot": "Dot",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _extract_http_error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            status_value = payload.get("status")
            if isinstance(status_value, dict):
                for key in ("error", "message", "detail"):
                    detail = status_value.get(key)
                    if isinstance(detail, str) and detail.strip():
                        return detail.strip()
            for key in ("error", "message", "detail"):
                detail = payload.get(key)
                if isinstance(detail, str) and detail.strip():
                    return detail.strip()
            result_value = payload.get("result")
            if isinstance(result_value, dict):
                for key in ("error", "message", "detail"):
                    detail = result_value.get(key)
                    if isinstance(detail, str) and detail.strip():
                        return detail.strip()

        response_text = response.text.strip()
        if response_text:
            return response_text[:500]
        return "No error details returned by Qdrant."

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        allow_404_retry: bool = True,
    ) -> tuple[httpx.Response | None, str | None]:
        try:
            response = httpx.request(
                method,
                url,
                headers=headers,
                json=json,
                params=params,
                timeout=self.DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            return response, None
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if allow_404_retry and status_code == 404:
                return None, "HTTP 404"
            detail = self._extract_http_error_detail(exc.response)
            return None, f"Qdrant responded with HTTP {status_code}: {detail}"
        except httpx.HTTPError as exc:
            return None, f"Request to Qdrant failed: {exc}"

    @staticmethod
    def _extract_embeddings(response_data: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if isinstance(response_data, dict):
            if isinstance(response_data.get("data"), list):
                for idx, item in enumerate(response_data["data"]):
                    embedding = None
                    if isinstance(item, dict):
                        embedding = item.get("embedding") or item.get("vector") or item.get("values")
                    results.append(
                        {
                            "id": items[idx]["id"] if idx < len(items) else None,
                            "embedding": embedding,
                            "raw": item,
                        }
                    )
                return results
            if isinstance(response_data.get("vectors"), list):
                for idx, vector in enumerate(response_data["vectors"]):
                    results.append(
                        {
                            "id": items[idx]["id"] if idx < len(items) else None,
                            "embedding": vector,
                            "raw": vector,
                        }
                    )
                return results
            if isinstance(response_data.get("result"), list):
                for idx, vector in enumerate(response_data["result"]):
                    results.append(
                        {
                            "id": items[idx]["id"] if idx < len(items) else None,
                            "embedding": vector,
                            "raw": vector,
                        }
                    )
                return results

        # Fallback: return raw response once
        results.append(
            {
                "id": items[0]["id"] if items else None,
                "embedding": None,
                "raw": response_data,
            }
        )
        return results

    def _generate_embeddings(
        self, 
        embedding_model_config: dict[str, Any] | None, 
        texts: list[str]
    ) -> list[list[float]]:
        """
        Generate embeddings from texts using Dify's embedding model.
        
        Args:
            embedding_model_config: Model configuration from model-selector parameter.
                                   Should be a dict that can be used to create TextEmbeddingModelConfig.
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        
        Note:
            Uses official Dify API: self.session.model.text_embedding.invoke()
            According to official docs: https://docs.dify.ai/plugin-dev-zh/9242-reverse-invocation-model
            This matches the implementation in siyu-text_embedding_vector_client plugin (verified working).
        """
        from dify_plugin.entities.model.text_embedding import TextEmbeddingModelConfig
        
        if not embedding_model_config:
            raise ValueError("embedding_model_config is required. Please select an embedding model in the node configuration.")
        
        if not isinstance(embedding_model_config, dict):
            raise ValueError(f"embedding_model_config must be a dict, got {type(embedding_model_config)}: {embedding_model_config}")
        
        # Handle nested 'value' format from model-selector
        # Dify may return {'value': {...}} or direct config dict
        if "value" in embedding_model_config and isinstance(embedding_model_config["value"], dict):
            embedding_model_config = embedding_model_config["value"]
        
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        batch_size = 64  # Default batch size
        vectors: list[list[float]] = []
        
        # Process texts in batches
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            
            try:
                # Create model config object (same as siyu plugin, verified working)
                # According to official docs, model-selector returns a dict that can be used to create ModelConfig
                # Debug: Log the config structure
                if not isinstance(embedding_model_config, dict):
                    raise ValueError(
                        f"embedding_model_config must be a dict, got {type(embedding_model_config)}: {embedding_model_config}"
                    )
                
                # Check if config has required fields
                if not embedding_model_config:
                    raise ValueError("embedding_model_config is empty. Please select an embedding model in the Agent node tool configuration.")
                
                try:
                    model_config = TextEmbeddingModelConfig(**embedding_model_config)
                except TypeError as e:
                    raise ValueError(
                        f"Failed to create TextEmbeddingModelConfig. "
                        f"Config keys: {list(embedding_model_config.keys())}, "
                        f"Config values: {list(embedding_model_config.values())[:3]}..., "
                        f"Error: {str(e)}. "
                        f"Please ensure you have selected an embedding model in the Agent node tool configuration."
                    ) from e
                
                # DEBUG: Check available methods in session.model
                # Uncomment to see all available methods:
                # available_methods = [attr for attr in dir(self.session.model) if not attr.startswith('_')]
                # print(f"DEBUG: Available methods in session.model: {available_methods}")
                
                # Call Dify's embedding service using official API
                # Official docs: self.session.model.text_embedding.invoke(model_config, texts)
                try:
                    response = self.session.model.text_embedding.invoke(
                        model_config=model_config,
                        texts=batch
                )
                except Exception as e:
                    error_msg = str(e)
                    if "selector" in error_msg.lower() or "dictionary" in error_msg.lower():
                        raise ValueError(
                            f"Embedding model configuration error: {error_msg}. "
                            f"This usually means the embedding model was not properly configured in the Agent node. "
                            f"Please ensure you have selected an embedding model in the tool configuration. "
                            f"Config keys: {list(embedding_model_config.keys())}"
                        ) from e
                    raise
                
                # Check response structure (same as siyu plugin)
                if not hasattr(response, "embeddings"):
                    raise ValueError(
                        f"Embedding service did not return the 'embeddings' field. "
                        f"Response type: {type(response)}, "
                        f"Response attributes: {dir(response) if hasattr(response, '__dict__') else 'N/A'}"
                    )
                
                # Extract embeddings (same as siyu plugin)
                if not isinstance(response.embeddings, list):
                    raise ValueError(f"Expected embeddings to be a list, got {type(response.embeddings)}")
                
                if len(response.embeddings) != len(batch):
                    raise ValueError(
                        f"Expected {len(batch)} embeddings, got {len(response.embeddings)}. "
                        f"Batch size: {len(batch)}, Embeddings count: {len(response.embeddings)}"
                    )
                
                vectors.extend(response.embeddings)
                
            except TypeError as e:
                # Handle TextEmbeddingModelConfig initialization errors
                raise ValueError(
                    f"Failed to create TextEmbeddingModelConfig from embedding_model_config. "
                    f"Config keys: {list(embedding_model_config.keys()) if isinstance(embedding_model_config, dict) else 'N/A'}. "
                    f"Error: {str(e)}"
                ) from e
            except AttributeError as e:
                # Handle session.model.text_embedding access errors (permission issues)
                raise ValueError(
                    f"Failed to access embedding service. "
                    f"Check if plugin has text_embedding permission in manifest.yaml. "
                    f"Error: {str(e)}"
                ) from e
            except Exception as e:
                # Generic error with full context
                error_type = type(e).__name__
                error_msg = str(e)
                raise ValueError(
                    f"Failed to generate embeddings (batch {start//batch_size + 1}): "
                    f"[{error_type}] {error_msg}. "
                    f"Config keys: {list(embedding_model_config.keys()) if isinstance(embedding_model_config, dict) else 'N/A'}, "
                    f"Texts count: {len(batch)}"
                ) from e
        
        if not vectors:
            raise ValueError("No embeddings generated. Check embedding model configuration and permissions.")
        
        return vectors

    def _handle_hybrid_search(
        self,
        base_url: str,
        items: list[dict[str, Any]],
        options: dict[str, Any],
        headers: dict[str, str],
    ) -> Generator[ToolInvokeMessage]:
        """
        Handle hybrid search using Qdrant 1.10+ Query API.
        
        Supports combining dense vectors, sparse vectors (BM25), and multiple search methods
        with fusion (RRF) and reranking capabilities.
        """
        collection = options.get("collection")
        if not collection:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "hybrid_search",
                    "status": "error",
                    "message": "Missing `options.collection` for hybrid search operation.",
                }
            )
            return

        # Build prefetch list for Query API
        prefetch = []
        
        # Add dense vector search if available
        dense_vector = options.get("dense_vector")
        using_dense = options.get("using_dense")
        prefetch_limit = options.get("prefetch_limit", 20)
        
        # MMR parameters
        mmr_diversity = options.get("mmr_diversity")
        mmr_candidates_limit = options.get("mmr_candidates_limit", 100)
        
        # Ensure dense_vector is a list/array, not a string or None
        if dense_vector:
            # Validate that dense_vector is a list/array
            if not isinstance(dense_vector, list):
                yield self.create_json_message(
                    {
                        "provider": self.PROVIDER,
                        "operation": "hybrid_search",
                        "status": "error",
                        "message": f"Invalid dense_vector type: expected list, got {type(dense_vector)}. Dense vector must be a numeric array.",
                    }
                )
                return
            
            # Build query: use NearestQuery with MMR if mmr_diversity is set, otherwise use vector directly
            if mmr_diversity is not None:
                # Apply MMR to dense vector search
                try:
                    mmr_diversity_float = float(mmr_diversity)
                    if not (0.0 <= mmr_diversity_float <= 1.0):
                        yield self.create_json_message(
                            {
                                "provider": self.PROVIDER,
                                "operation": "hybrid_search",
                                "status": "error",
                                "message": f"Invalid mmr_diversity: {mmr_diversity}. Must be between 0.0 and 1.0.",
                            }
                        )
                        return
                    
                    dense_query = {
                        "nearest": dense_vector,
                        "mmr": {
                            "diversity": mmr_diversity_float,
                            "candidates_limit": int(mmr_candidates_limit)
                        }
                    }
                except (ValueError, TypeError):
                    yield self.create_json_message(
                        {
                            "provider": self.PROVIDER,
                            "operation": "hybrid_search",
                            "status": "error",
                            "message": f"Invalid mmr_diversity value: {mmr_diversity}. Must be a number between 0.0 and 1.0.",
                        }
                    )
                    return
            else:
                # No MMR: use vector directly
                dense_query = dense_vector
            
            dense_prefetch: dict[str, Any] = {
                "query": dense_query,  # Can be vector array or NearestQuery with MMR
                "limit": prefetch_limit,
                "filter": options.get("filter", {}) if isinstance(options.get("filter"), dict) else {},
            }
            if using_dense:
                dense_prefetch["using"] = using_dense
            prefetch.append(dense_prefetch)
        
        # Add sparse vector search if available
        sparse_vector = options.get("sparse_vector")
        using_sparse = options.get("using_sparse")
        
        if sparse_vector:
            # Note: MMR is only supported for dense vectors (NearestQuery), not sparse vectors
            # Sparse vectors are used directly without MMR
            sparse_query = sparse_vector
            
            sparse_prefetch: dict[str, Any] = {
                "query": sparse_query,  # Can be sparse vector dict or NearestQuery with MMR
                "limit": prefetch_limit,
                "filter": options.get("filter", {}) if isinstance(options.get("filter"), dict) else {},
            }
            if using_sparse:
                sparse_prefetch["using"] = using_sparse
            prefetch.append(sparse_prefetch)
        # Note: We don't use text directly in prefetch
        # Text is converted to dense_vector on client side before reaching here
        # Qdrant Query API requires vectorizer for text input, which we avoid
        
        if not prefetch:
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "hybrid_search",
                    "status": "error",
                    "message": "At least one of 'dense_vector' or 'sparse_vector' must be provided for hybrid search.",
                }
            )
            return
        
        # Build Query API request body
        # Try /v1/ endpoint first (newer Qdrant versions), then fallback to non-v1
        endpoint = resolve_endpoint(base_url, f"/v1/collections/{collection}/points/query")
        fallback_endpoint = resolve_endpoint(base_url, f"/collections/{collection}/points/query")
        
        # Validate prefetch: ensure all queries are vectors (list or dict), never text strings
        for i, prefetch_item in enumerate(prefetch):
            query = prefetch_item.get("query")
            if query is None:
                yield self.create_json_message(
                    {
                        "provider": self.PROVIDER,
                        "operation": "hybrid_search",
                        "status": "error",
                        "message": f"Invalid prefetch item {i}: missing 'query' field.",
                    }
                )
                return
            # Query must be either a list (dense vector) or dict (sparse vector), never a string
            if isinstance(query, str):
                yield self.create_json_message(
                    {
                        "provider": self.PROVIDER,
                        "operation": "hybrid_search",
                        "status": "error",
                        "message": f"Invalid prefetch item {i}: 'query' must be a vector (list) or sparse vector (dict), not a string. Text input requires embedding_model_config to generate vectors.",
                    }
                )
                return
        
        # Ensure prefetch only contains vector queries (no text)
        # Each prefetch item should have "query" as vector array or sparse vector dict
        body: dict[str, Any] = {
            "prefetch": prefetch,  # Only vectors, no text strings
        }
        
        # Log for debugging (commented out to avoid logger dependency)
        # logger.debug(f"Hybrid search request: prefetch count={len(prefetch)}, dense={any('query' in p and isinstance(p['query'], list) for p in prefetch)}, sparse={any('query' in p and isinstance(p['query'], dict) for p in prefetch)}")
        
        # Add fusion query if fusion method is specified
        fusion_method = options.get("fusion_method", "rrf")
        if fusion_method and fusion_method.lower() == "rrf":
            # RRF fusion: combine results from multiple prefetch operations
            body["query"] = {"fusion": "rrf"}
        elif fusion_method:
            # For other fusion methods or custom queries, pass as-is
            # If fusion_method is a string like "rrf", convert to dict format
            if isinstance(fusion_method, str) and fusion_method.lower() in ["rrf"]:
                body["query"] = {"fusion": fusion_method.lower()}
            else:
                body["query"] = fusion_method
        else:
            # No fusion: just return prefetch results (single method search)
            # For true hybrid search, fusion should be enabled
            pass
        
        # Add limit
        body["limit"] = options.get("limit", 10)
        
        # Add filter - Qdrant requires filter to be a dictionary (even if empty)
        body["filter"] = options.get("filter", {}) if isinstance(options.get("filter"), dict) else {}
        
        # Add with_payload and with_vectors
        if "with_payload" in options:
            body["with_payload"] = options["with_payload"]
        else:
            body["with_payload"] = True
        
        if "with_vectors" in options:
            body["with_vectors"] = options["with_vectors"]
        
        # Make request to Query API
        response, error_message = self._request(
            "POST", endpoint, headers=headers, json=body, allow_404_retry=False
        )
        
        # Check if response is HTML (wrong URL or auth issue)
        need_fallback = False
        if response and response.text and response.text.strip().startswith("<!DOCTYPE"):
            error_message = f"Qdrant returned HTML instead of JSON. This usually means wrong URL or auth issue. Endpoint: {endpoint}"
            need_fallback = True
        
        if (error_message and ("HTTP 404" in error_message or need_fallback) and fallback_endpoint):
            response, error_message = self._request(
                "POST", fallback_endpoint, headers=headers, json=body, allow_404_retry=False
            )
            # Check fallback also for HTML
            if response and response.text and response.text.strip().startswith("<!DOCTYPE"):
                error_message = f"Qdrant returned HTML instead of JSON on both endpoints. Check your Qdrant URL configuration. Endpoints tried: {endpoint}, {fallback_endpoint}"
        
        # Final check: ensure response is valid JSON before parsing
        if response and response.text:
            response_text = response.text.strip()
            if response_text.startswith("<!DOCTYPE") or response_text.startswith("<html"):
                error_message = f"Qdrant returned HTML instead of JSON. This usually indicates:\n1. Wrong Qdrant URL (check base_url configuration)\n2. Authentication issue (check api_key)\n3. Qdrant instance is not accessible\nEndpoint attempted: {endpoint}"
                if fallback_endpoint:
                    error_message += f"\nFallback endpoint attempted: {fallback_endpoint}"
        
        if error_message:
            # Check if error is related to vectorizer (paid feature)
            if "vectorizer" in error_message.lower() or "No vectorizer specified" in error_message:
                # This error should not occur if we only pass vectors (not text)
                # If it does, it might be a Qdrant server-side issue or configuration problem
                yield self.create_json_message(
                    {
                        "provider": self.PROVIDER,
                        "operation": "hybrid_search",
                        "inputs": items,
                        "options": {k: v for k, v in options.items() if k not in ["dense_vector", "sparse_vector"]},
                        "status": "error",
                        "message": f"Qdrant vectorizer error: {error_message}\n\n"
                        "This error typically occurs when:\n"
                        "1. Qdrant requires vectorizer for text input (paid feature)\n"
                        "2. Our plugin should only pass vectors, not text\n"
                        "3. Please ensure you provided 'embedding_model_config' to generate vectors from text\n"
                        "4. If using Qdrant Cloud, vectorizer may require a paid plan\n"
                        "5. Solution: Always provide pre-computed vectors or use embedding_model_config to generate vectors on client side",
                    }
                )
                return
            
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "hybrid_search",
                    "inputs": items,
                    "options": options,
                    "status": "error",
                    "message": error_message,
                }
            )
            return
        
        # Safe JSON parsing with error handling
        try:
            result_data = response.json() if response else {}
        except Exception as json_error:
            # If JSON parsing fails, check if it's HTML
            if response and response.text:
                response_text = response.text.strip()
                if response_text.startswith("<!DOCTYPE") or response_text.startswith("<html"):
                    yield self.create_json_message(
                        {
                            "provider": self.PROVIDER,
                            "operation": "hybrid_search",
                            "inputs": items,
                            "options": options,
                            "status": "error",
                            "message": f"Qdrant returned HTML instead of JSON. This usually indicates:\n1. Wrong Qdrant URL (check base_url configuration)\n2. Authentication issue (check api_key)\n3. Qdrant instance is not accessible\nEndpoint: {endpoint}\nFirst 200 chars of response: {response_text[:200]}",
                        }
                    )
                    return
            yield self.create_json_message(
                {
                    "provider": self.PROVIDER,
                    "operation": "hybrid_search",
                    "inputs": items,
                    "options": options,
                    "status": "error",
                    "message": f"Failed to parse Qdrant response as JSON: {json_error}\nResponse text (first 500 chars): {response.text[:500] if response else 'No response'}",
                }
            )
            return
        
        # Qdrant Query API 返回格式: {"result": {"points": [...]}, "status": "ok", "time": ...}
        # 直接返回 Qdrant API 的原始结果，不做额外处理
        qdrant_result = result_data.get("result", {})
        
        # 如果 result 是字典（包含 points），直接返回
        # 如果 result 是列表（某些版本可能直接返回列表），也直接返回
        if isinstance(qdrant_result, dict):
            points = qdrant_result.get("points", [])
            result_count = len(points)
        elif isinstance(qdrant_result, list):
            points = qdrant_result
            result_count = len(points)
        else:
            points = qdrant_result
            result_count = 1 if points else 0
        
        # Remove vectors from points (too long for display)
        points_summary = []
        for point in points:
            if isinstance(point, dict):
                point_summary = {
                    "id": point.get("id"),
                    "score": point.get("score"),
                    "payload": point.get("payload"),
                }
                # Add vector dimension info instead of full vector
                if "vector" in point:
                    point_summary["vector_dimension"] = len(point["vector"]) if isinstance(point["vector"], list) else None
                elif "vectors" in point:
                    # Handle named vectors
                    vectors_info = {}
                    for name, vec in point["vectors"].items():
                        vectors_info[name] = len(vec) if isinstance(vec, list) else None
                    point_summary["vectors_dimensions"] = vectors_info
                points_summary.append(point_summary)
            else:
                points_summary.append(point)
        
        yield self.create_json_message(
            {
                "provider": self.PROVIDER,
                "operation": "hybrid_search",
                "inputs": items,
                "options": {k: v for k, v in options.items() if k not in ["dense_vector", "sparse_vector"]},  # Remove large vectors from output (text is not in options anymore)
                "endpoint": endpoint,
                "result": points_summary,  # Return summary without vectors
                "status": "success",
                "message": f"Hybrid search executed successfully. Found {result_count} results.",
            }
        )
