# Real mixed report example

This is one reviewed output from the local FinResearch AI report pipeline. The exact request was:

> Create a concise report comparing Apple and Microsoft FY2025 revenue and summarizing one supply-chain risk disclosed by Apple in its FY2025 Form 10-K.

The real FastAPI app generated a single ZIP through `POST /reports/export`. Its unchanged contents are the [Markdown report](report.md), the [attachment manifest](manifest.json), and the [revenue chart](attachments/chart-sql_revenue_comparison.png). Open the report and chart together; the Markdown names the evidence IDs, and the manifest records the chart's original attachment ID and path.

The report compares Apple revenue of 416,161 million USD with Microsoft revenue of 281,724 million USD for FY2025. These values came from the local SQLite `financial_metrics` rows in `data/financial_demo.db`. The supply-chain statement cites page 11, chunk 1 of the locally indexed Apple FY2025 Form 10-K, identified in the corpus as `nasdaq-aapl-2025-10K-251437791.pdf`. That chunk appears in the filing's Risk Factors section and describes Apple's large, complex global supply chain and the location of most supplier facilities outside the U.S. The chart shows the two revenue values; the filing evidence supports the qualitative statement.

Apple FY2025 ended September 27, 2025. Microsoft FY2025 ended June 30, 2025. Both revenue values are reported FY2025 figures, but the companies' fiscal years end on different dates. The chart compares reported annual revenue figures; the reporting windows are not identical.

For independent verification, see [Apple's FY2025 SEC Form 10-K](https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm) and [Microsoft's FY2025 Annual Report](https://www.microsoft.com/investor/reports/ar25/index.html). These primary sources independently corroborate the revenue amounts and fiscal-year end dates. The demo's SQL amounts originate from the project's local seeded SQLite database.

This run used the application's normal service composition inside FastAPI `TestClient`, with local Ollama models `qwen3:4b` and `qwen3:4b-instruct`. The active report model context was 8192 tokens. It used the existing local SQLite and Chroma data, with no dependency overrides.

To run the existing live smoke check on your own prepared local data, use:

```bash
.venv/bin/python -m scripts.smoke_report_live --scenario mixed --format both
```

That command makes fresh JSON and ZIP requests, so newly generated text, citations, chart details, and bytes may differ from this captured ZIP. It requires the local database, indexed filing, and Ollama models described in the root README. This local demo is one validated example, not an accuracy benchmark or evidence that arbitrary questions always produce correct answers.
