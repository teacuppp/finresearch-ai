import pytest

from app.agent.sql_executor import validate_read_only_sql
from app.analysis.financial_analyzer import AnalysisOperation
from app.analysis.retrieval import AnalysisRetrievalPlan
from app.analysis.sql_builder import AnalysisSQLBuildError, AnalysisSQLBuilder


def plan(entity_column="company", entities=("Apple",),
         years=(2024, 2025), metric="revenue_musd"):
    return AnalysisRetrievalPlan(
        entity_column=entity_column, entities=entities,
        fiscal_years=years, metric=metric,
    )


@pytest.mark.parametrize("operation", [
    AnalysisOperation.ABSOLUTE_CHANGE, AnalysisOperation.PERCENTAGE_CHANGE,
])
def test_change_queries_retrieve_two_raw_years_in_deterministic_order(operation):
    sql = AnalysisSQLBuilder().build(plan(years=(2025, 2024)), operation)
    assert sql == (
        "SELECT company, ticker, fiscal_year, revenue_musd\n"
        "FROM financial_metrics\n"
        "WHERE company = 'Apple' AND fiscal_year IN (2024, 2025)\n"
        "ORDER BY fiscal_year"
    )
    assert sql.count("SELECT") == 1
    for forbidden in ("JOIN", "UNION", "SELECT *", " AS old", " AS new", " - ", " / "):
        assert forbidden not in sql


def test_difference_queries_keep_entity_identity_and_order_by_identifier():
    sql = AnalysisSQLBuilder().build(
        plan("ticker", ("AAPL", "MSFT"), (2025,), "gross_margin"),
        AnalysisOperation.DIFFERENCE,
    )
    assert sql == (
        "SELECT company, ticker, fiscal_year, gross_margin\n"
        "FROM financial_metrics\n"
        "WHERE ticker IN ('AAPL', 'MSFT') AND fiscal_year = 2025\n"
        "ORDER BY ticker"
    )
    assert "ORDER BY gross_margin" not in sql


def test_ranking_retrieves_raw_rows_without_metric_sort():
    sql = AnalysisSQLBuilder().build(
        plan(entities=("Apple", "Microsoft"), years=(2025,)),
        AnalysisOperation.RANKING,
    )
    assert sql == (
        "SELECT company, ticker, fiscal_year, revenue_musd\n"
        "FROM financial_metrics\n"
        "WHERE company IN ('Apple', 'Microsoft') AND fiscal_year = 2025\n"
        "ORDER BY company"
    )
    assert "ORDER BY revenue_musd" not in sql


def test_all_entities_ranking_omits_entity_predicate():
    sql = AnalysisSQLBuilder().build(
        plan("ticker", (), (2025,), "operating_income_musd"),
        AnalysisOperation.RANKING,
    )
    assert sql == (
        "SELECT company, ticker, fiscal_year, operating_income_musd\n"
        "FROM financial_metrics\n"
        "WHERE fiscal_year = 2025\n"
        "ORDER BY ticker"
    )


def test_single_quote_in_entity_is_escaped_without_changing_input():
    source = plan(entities=("O'Reilly",))
    sql = AnalysisSQLBuilder().build(source, AnalysisOperation.ABSOLUTE_CHANGE)
    assert "company = 'O''Reilly'" in sql
    assert source.entities == ("O'Reilly",)


def test_existing_read_only_validator_is_used(monkeypatch):
    calls = []

    def spy(statement):
        calls.append(statement)
        return validate_read_only_sql(statement)

    monkeypatch.setattr("app.analysis.sql_builder.validate_read_only_sql", spy)
    sql = AnalysisSQLBuilder().build(plan(), AnalysisOperation.ABSOLUTE_CHANGE)
    assert calls == [sql]
    assert validate_read_only_sql(sql) == sql
    assert ";" not in sql


@pytest.mark.parametrize("operation,retrieval_plan", [
    (AnalysisOperation.RANKING, plan()),
    (AnalysisOperation.DIFFERENCE, plan()),
    (AnalysisOperation.ABSOLUTE_CHANGE, plan(years=(2025,))),
    ("ranking", plan(entities=(), years=(2025,))),
    (AnalysisOperation.RANKING, object()),
])
def test_invalid_operation_or_plan_is_rejected(operation, retrieval_plan):
    with pytest.raises(AnalysisSQLBuildError):
        AnalysisSQLBuilder().build(retrieval_plan, operation)


def test_forged_column_is_rejected_before_sql_construction():
    forged = AnalysisRetrievalPlan.model_construct(
        entity_column="id; DROP TABLE financial_metrics", entities=("Apple",),
        fiscal_years=(2024, 2025), metric="revenue_musd",
    )
    with pytest.raises(AnalysisSQLBuildError):
        AnalysisSQLBuilder().build(forged, AnalysisOperation.ABSOLUTE_CHANGE)
