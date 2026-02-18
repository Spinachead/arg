from dataclasses import dataclass, field

@dataclass(kw_only=True)
class ResearcherState:
    """State of the researcher graph."""
    queries: list[str] = field(default_factory=list)
    context: str = field(default="")
