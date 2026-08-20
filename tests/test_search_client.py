from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.services.search_client import SearchClient


def _offline_settings() -> Settings:
    return Settings(_env_file=None, tavily_api_key=None)


def test_offline_search_returns_relevant_sources() -> None:
    client = SearchClient(settings=_offline_settings())

    results = client.search("multi-agent architecture research tasks", max_results=3)

    assert results
    assert len(results) <= 3
    assert all(r.snippet for r in results)
    assert all(r.metadata.get("provider") == "offline_corpus" for r in results)


def test_offline_search_falls_back_when_no_keyword_match() -> None:
    client = SearchClient(settings=_offline_settings())

    results = client.search("zzz_no_such_keyword_at_all_xyz", max_results=2)

    # Falls back to first-topic sources rather than returning nothing.
    assert results
