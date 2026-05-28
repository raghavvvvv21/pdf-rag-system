import arxiv

search = arxiv.Search(
    query="large language models",
    max_results=2,
    sort_by=arxiv.SortCriterion.Relevance
)

results = search.results()

for i, result in enumerate(results):

    print(f"\n========== Result {i+1} ==========")

    print("Title:", result.title)

    print("Authors:", [a.name for a in result.authors])

    print("Published:", result.published)

    print("\nSummary:\n")

    print(result.summary[:500])