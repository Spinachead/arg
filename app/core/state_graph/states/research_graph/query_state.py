from dataclasses import dataclass


@dataclass(kw_only=True)
class QueryState:
    """State class for managing research queries in the research graph."""
    query: str
