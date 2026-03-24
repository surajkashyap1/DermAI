from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import httpx


EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass
class PubMedFetchResult:
    dataset_id: str
    query: str
    count: int
    ids: list[str]
    articles: list[dict[str, Any]]


def _client() -> httpx.Client:
    return httpx.Client(base_url=EUTILS_BASE_URL, timeout=60.0)


def _extract_xml_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join(part.strip() for part in node.itertext() if part and part.strip())


def search_pubmed(query: str, retmax: int = 20) -> dict[str, Any]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
    }
    with _client() as client:
        response = client.get("/esearch.fcgi", params=params)
        response.raise_for_status()
    return response.json()["esearchresult"]


def fetch_pubmed_records(pubmed_ids: list[str]) -> list[dict[str, Any]]:
    if not pubmed_ids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pubmed_ids),
        "retmode": "xml",
    }
    with _client() as client:
        response = client.get("/efetch.fcgi", params=params)
        response.raise_for_status()
    root = ET.fromstring(response.text)
    records: list[dict[str, Any]] = []

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        article_node = medline.find("Article") if medline is not None else None
        journal = article_node.find("Journal") if article_node is not None else None
        journal_issue = journal.find("JournalIssue") if journal is not None else None
        pub_date = journal_issue.find("PubDate") if journal_issue is not None else None
        abstract_node = article_node.find("Abstract") if article_node is not None else None

        pmid = _extract_xml_text(medline.find("PMID")) if medline is not None else ""
        title = _extract_xml_text(article_node.find("ArticleTitle")) if article_node is not None else ""
        journal_title = _extract_xml_text(journal.find("Title")) if journal is not None else "PubMed"
        year = _extract_xml_text(pub_date.find("Year")) if pub_date is not None else ""
        medline_date = _extract_xml_text(pub_date.find("MedlineDate")) if pub_date is not None else ""
        pubdate = year or medline_date

        authors: list[dict[str, Any]] = []
        if article_node is not None:
            for author in article_node.findall(".//Author"):
                collective = _extract_xml_text(author.find("CollectiveName"))
                if collective:
                    authors.append({"name": collective})
                    continue
                last_name = _extract_xml_text(author.find("LastName"))
                fore_name = _extract_xml_text(author.find("ForeName"))
                name = " ".join(part for part in [fore_name, last_name] if part)
                if name:
                    authors.append({"name": name})

        abstract_sections: list[dict[str, str]] = []
        if abstract_node is not None:
            for abstract_text in abstract_node.findall("AbstractText"):
                label = abstract_text.attrib.get("Label") or "Abstract"
                text = _extract_xml_text(abstract_text)
                if text:
                    abstract_sections.append({"label": label, "text": text})

        records.append(
            {
                "uid": pmid,
                "title": title,
                "fulljournalname": journal_title,
                "pubdate": pubdate,
                "authors": authors,
                "abstract_sections": abstract_sections,
                "articleids": [{"idtype": "pubmed", "value": pmid}] if pmid else [],
            }
        )

    return records


def fetch_dataset(query: str, dataset_id: str, retmax: int = 20) -> PubMedFetchResult:
    search = search_pubmed(query=query, retmax=retmax)
    ids = search.get("idlist", [])
    articles = fetch_pubmed_records(ids)
    return PubMedFetchResult(
        dataset_id=dataset_id,
        query=query,
        count=int(search.get("count", len(ids))),
        ids=ids,
        articles=articles,
    )


def save_raw_dataset(output_path: Path, dataset: PubMedFetchResult) -> None:
    payload = {
        "dataset_id": dataset.dataset_id,
        "query": dataset.query,
        "count": dataset.count,
        "ids": dataset.ids,
        "articles": dataset.articles,
        "source": f"{EUTILS_BASE_URL}?{urlencode({'db': 'pubmed', 'term': dataset.query})}",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def normalize_article(article: dict[str, Any]) -> dict[str, Any]:
    pmid = article.get("uid") or article.get("articleids", [{}])[0].get("value")
    title = article.get("title") or article.get("sorttitle") or "Untitled PubMed record"
    source = article.get("fulljournalname") or article.get("source") or "PubMed"
    pubdate = article.get("pubdate") or article.get("epubdate") or ""
    year = 2026
    for token in str(pubdate).split():
        if token.isdigit() and len(token) == 4:
            year = int(token)
            break

    authors = []
    for author in article.get("authors", []):
        if isinstance(author, dict):
            name = author.get("name")
            if name:
                authors.append(name)
        elif isinstance(author, str):
            authors.append(author)

    sections = []
    abstract_sections = article.get("abstract_sections") or []
    if abstract_sections:
        sections.extend(
            {
                "title": item.get("label") or "Abstract",
                "text": item.get("text", "").strip(),
            }
            for item in abstract_sections
            if item.get("text")
        )
    else:
        abstract = article.get("abstract") or article.get("elocationid") or ""
        if isinstance(abstract, dict):
            abstract = ""
        sections = [{"title": "Abstract", "text": str(abstract).strip()}]
    return {
        "title": title,
        "source": source,
        "source_type": "pubmed_json",
        "authority_level": "external_reference",
        "audience": "clinician",
        "year": year,
        "href": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        "pmid": pmid,
        "authors": authors,
        "doc_type": "json",
        "topic_tags": [],
        "disease_tags": [],
        "sections": sections,
    }


def save_normalized_articles(output_dir: Path, dataset: PubMedFetchResult, topic_tags: list[str], disease_tags: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for article in dataset.articles:
        normalized = normalize_article(article)
        normalized["topic_tags"] = topic_tags
        normalized["disease_tags"] = disease_tags
        pmid = normalized.get("pmid") or "unknown"
        path = output_dir / f"pubmed-{pmid}.json"
        path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        written.append(path)
    return written
