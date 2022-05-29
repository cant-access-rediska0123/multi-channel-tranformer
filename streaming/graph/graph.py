from typing import List, Set, Generic, TypeVar, Optional, Iterable, Tuple

NodeInfo = TypeVar('NodeInfo')


class Graph(Generic[NodeInfo]):
    def __init__(self):
        self._forward_edges: List[Set[int]] = []
        self._backward_edges: List[Set[int]] = []
        self._node_info: List[Optional[NodeInfo]] = []
        self._nodes: Set[int] = set()

    def add_node(self) -> int:
        self._forward_edges.append(set())
        self._backward_edges.append(set())
        self._node_info.append(None)
        self._nodes.add(len(self._forward_edges) - 1)
        return len(self._forward_edges) - 1

    def add_node_info(self, node_id: int, node_info: NodeInfo):
        self._node_info[node_id] = node_info

    def add_edge(self, from_id: int, to_id: int):
        self._forward_edges[from_id].add(to_id)
        self._backward_edges[to_id].add(from_id)

    def remove_node(self, node_id: int):
        forward_edges = self._forward_edges[node_id].copy()
        backward_edges = self._backward_edges[node_id].copy()
        for b in forward_edges:
            if node_id != b:
                self.remove_edge(node_id, b)
        for b in backward_edges:
            if node_id != b:
                self.remove_edge(b, node_id)
        self._backward_edges[node_id].clear()
        self._forward_edges[node_id].clear()
        self._node_info[node_id] = None
        self._nodes.remove(node_id)

    def remove_edge(self, from_id: int, to_id: int):
        self._forward_edges[from_id].remove(to_id)
        self._backward_edges[to_id].remove(from_id)

    def merge_nodes(self, node_ids_to_merge: Iterable[int]) -> int:
        forward_edges = set.union(*[self._forward_edges[node_id] for node_id in node_ids_to_merge])
        backward_edges = set.union(*[self._backward_edges[node_id] for node_id in node_ids_to_merge])
        for node_id in node_ids_to_merge:
            self.remove_node(node_id)
        merged = self.add_node()
        for node_id in forward_edges:
            if node_id not in node_ids_to_merge:
                self.add_edge(merged, node_id)
        for node_id in backward_edges:
            if node_id not in node_ids_to_merge:
                self.add_edge(node_id, merged)
        if any(node_id in backward_edges | forward_edges for node_id in node_ids_to_merge):
            self.add_edge(merged, merged)
        return merged

    def get_node_info(self, node_id: int) -> Optional[NodeInfo]:
        return self._node_info[node_id]

    def get_forward_edges(self, node_id: int) -> Iterable[int]:
        return self._forward_edges[node_id].copy()

    def get_backward_edges(self, node_id: int) -> Iterable[int]:
        return self._backward_edges[node_id].copy()

    def __str__(self):
        str_repr = ''
        for i in self._nodes:
            str_repr += f'{i}: {self.get_node_info(i)}\n'
            for nxt in self._forward_edges[i]:
                str_repr += f'\t-> {nxt}\n'
        return str_repr

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterable[Tuple[int, NodeInfo]]:
        return iter([(node_id, self.get_node_info(node_id)) for node_id in self._nodes])
