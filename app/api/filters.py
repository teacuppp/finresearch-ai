from typing import Any


def build_metadata_filter(
    company: str | None = None,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    document_type: str | None = None,
) -> dict[str, Any] | None:
    conditions = []

    if company is not None:
        conditions.append(
            {
                "company": {
                    "$eq": company
                }
            }
        )

    if ticker is not None:
        conditions.append(
            {
                "ticker": {
                    "$eq": ticker
                }
            }
        )

    if fiscal_year is not None:
        conditions.append(
            {
                "fiscal_year": {
                    "$eq": fiscal_year
                }
            }
        )

    if document_type is not None:
        conditions.append(
            {
                "document_type": {
                    "$eq": document_type
                }
            }
        )

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {
        "$and": conditions
    }





# rag.py
#   ↓
# filters.py

# filters.py
#   ✗ 不依赖 FastAPI
#   ✗ 不依赖 AskRequest
#   ✗ 不依赖 RAGPipeline




# def build_metadata_filter(
#     request: AskRequest,
# ) -> dict | None:
#     conditions = []

#     if request.company is not None:
#         conditions.append(
#             {
#                 "company": {
#                     "$eq": request.company
#                 }
#             }
#         )

#     if request.ticker is not None:
#         conditions.append(
#             {
#                 "ticker": {
#                     "$eq": request.ticker
#                 }
#             }
#         )

#     if request.fiscal_year is not None:
#         conditions.append(
#             {
#                 "fiscal_year": {
#                     "$eq": request.fiscal_year
#                 }
#             }
#         )

#     if request.document_type is not None:
#         conditions.append(
#             {
#                 "document_type": {
#                     "$eq": request.document_type
#                 }
#             }
#         )

#     if not conditions:
#         return None

#     if len(conditions) == 1:
#         return conditions[0]

#     return {
#         "$and": conditions
#     }
