"""Parent/child chunks — search small pieces, expand to wider context when useful."""

from typing import Any, Literal

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode, NodeRelationship

from app.chunkers.base_chunker import (
    BaseChunker,
    Chunk,
    ChunkMetadata,
    ContextChunk,
)

# EmbedAt is the level at which to embed the nodes so that the node is searchable.
# Which hierarchy level is searchable: "leaves" means smallest chunks; int means level number.
EmbedAt = int | Literal["leaves"]


def _parent_ref(node: BaseNode) -> Any | None:
    # Returns the parent node for a given node
    raw = node.relationships.get(NodeRelationship.PARENT)
    if isinstance(raw, list):
        return raw[0] if raw else None
    return raw


def _child_refs(node: BaseNode) -> list[Any]:
    # Returns the child nodes for a given node

    # the child nodes for a given node
    raw = node.relationships.get(NodeRelationship.CHILD)
    if raw is None:
        return []
    return raw if isinstance(raw, list) else [raw]


def _node_level(node: BaseNode, by_parser_id: dict[str, BaseNode]) -> int:
    # Returns the level of a given node
    level = 0
    ref = _parent_ref(node)
    while ref is not None:
        level += 1
        parent = by_parser_id.get(ref.node_id)
        if parent is None:
            break
        ref = _parent_ref(parent)
    return level


def _rewrite_relationship_ids(node: BaseNode, id_map: dict[str, str]) -> None:
    # Rewrites the relationship ids for a given node
    parent_ref = _parent_ref(node)
    # Rewrite the parent id
    if parent_ref is not None:
        stable = id_map.get(parent_ref.node_id, parent_ref.node_id)
        node.relationships[NodeRelationship.PARENT] = parent_ref.model_copy(
            update={"node_id": stable}
        )
    children = _child_refs(node)
    if children:
        node.relationships[NodeRelationship.CHILD] = [
            c.model_copy(update={"node_id": id_map.get(c.node_id, c.node_id)})
            for c in children
        ]


class HierarchicalChunker(BaseChunker):
    """N-level hierarchy: embed chosen level; store full tree for auto-merge at search."""

    name = "hierarchical"

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        *,
        chunk_sizes: list[int] | None = None,
        parent_multiplier: int = 3,
        embed_at: EmbedAt = "leaves",
    ) -> None:
        self.chunk_overlap = chunk_overlap
        self.chunk_sizes = self._resolve_chunk_sizes(
            chunk_size, chunk_sizes, parent_multiplier
        )
        self.embed_at = embed_at
        self._hierarchy_nodes: list[BaseNode] | None = None
        if embed_at != "leaves":
            if not isinstance(embed_at, int) or not 0 <= embed_at < len(self.chunk_sizes):
                raise ValueError(
                    f"embed_at must be 'leaves' or 0..{len(self.chunk_sizes) - 1}"
                )

    @staticmethod
    def _resolve_chunk_sizes(
        chunk_size: int,
        chunk_sizes: list[int] | None,
        parent_multiplier: int,
    ) -> list[int]:
        if chunk_sizes:
            sizes = sorted({int(s) for s in chunk_sizes if int(s) > 0}, reverse=True)
            if len(sizes) < 2:
                raise ValueError("hierarchical chunk_sizes needs at least 2 levels")
            return sizes
        mult = max(2, parent_multiplier)
        return [chunk_size * mult, chunk_size]

    def _parse_documents(self, documents: list[Document]) -> list[BaseNode]:
        self._hierarchy_nodes = None
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes,
            chunk_overlap=self.chunk_overlap,
            include_metadata=True,
        )
        return list(parser.get_nodes_from_documents(documents))

    def _assign_chunk_ids(self, chunks: list[BaseNode]) -> list[BaseNode]:
        # 1. Assigns chunk ids to the given chunks. The chunk ids look like this:
        #   <source>#p<page>#h<index> for root nodes
        #   <parent_id>n<index> for child nodes
        # 2. This also assigns ids to the relationships for the chunks.
        if not chunks:
            self._hierarchy_nodes = None
            return chunks
        counts: dict[tuple[str, int | None, str], int] = {}
        id_map: dict[str, str] = {}
        by_parser_id = {c.node_id: c for c in chunks}
        # sort the chunks by node level
        # This is important because child ids are based on the parent id, so we need to ensure the parent is assigned first.
        # level 0 is parent, level 1 is child, etc.
        ordered = sorted(chunks, key=lambda n: _node_level(n, by_parser_id))

        for chunk in ordered:
            parser_id = chunk.node_id
            md = dict(chunk.metadata or {})
            source, page_num, prefix = self._file_prefix(md)
            children = _child_refs(chunk)
            parent_ref = _parent_ref(chunk)
            level = _node_level(chunk, by_parser_id)

            if parent_ref is None:
                i = self._next_index(counts, source, page_num, "root")
                cid = f"{prefix}h{i}"
            else:
                parent_stable = id_map.get(parent_ref.node_id, parent_ref.node_id)
                i = self._next_index(counts, source, page_num, f"children:{parent_stable}")
                cid = f"{parent_stable}n{i}"
                md["parent_id"] = parent_stable

            md["chunk_role"] = "parent" if children else "leaf"
            md["level"] = level
            # add the chunk id to the id map
            id_map[parser_id] = cid
            chunk.metadata, chunk.node_id = md, cid

        # rewrite the relationship ids for the chunks to use the new chunk ids
        for chunk in chunks:
            _rewrite_relationship_ids(chunk, id_map)
        self._hierarchy_nodes = chunks
        return chunks

    @property
    def hierarchy_nodes(self) -> list[BaseNode] | None:
        """Full parsed tree from the last chunk_file / _chunk_documents call."""
        return self._hierarchy_nodes

    def _embed_nodes(self, nodes: list[BaseNode]) -> list[BaseNode]:
        if self.embed_at == "leaves":
            return get_leaf_nodes(nodes)
        
        # if embed_at is an int; at a given level, return the nodes at that level
        return [
            n
            for n in nodes
            if int((n.metadata or {}).get("level", -1)) == self.embed_at
        ]

    def _build_chunks(self, nodes: list[BaseNode]) -> list[Chunk]:
        by_id = {n.node_id: n for n in nodes}
        chunks: list[Chunk] = []
        for node in self._embed_nodes(nodes):
            meta = ChunkMetadata.from_dict(node.metadata or {})
            context: ContextChunk | None = None
            parent_id = meta.extra.get("parent_id")
            if parent_id and parent_id in by_id:
                parent = by_id[parent_id]
                context = ContextChunk(
                    chunk_id=parent.node_id,
                    text=parent.get_content(),
                    metadata=ChunkMetadata.from_dict(parent.metadata or {}),
                )
            chunks.append(
                Chunk(
                    chunk_id=node.node_id,
                    text=node.get_content(),
                    metadata=meta,
                    context=context,
                )
            )
        return chunks
