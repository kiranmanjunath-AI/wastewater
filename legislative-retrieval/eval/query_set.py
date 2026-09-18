"""
query_set.py
25 hand-authored query / expected-citation pairs for evaluating ITA retrieval quality.

Covers 7 topic areas:
  1. Employment income        (s.5-8)
  2. Capital gains            (s.38-55)
  3. Deductions – business    (s.20)
  4. Definitions              (s.248)
  5. Corporate tax            (s.125)
  6. International/non-resident (s.212-220)
  7. RRSP / pension           (s.146)

Format: list of dicts with keys:
  query               str   — natural-language question
  expected_citations  list  — one or more ITA citations that are correct answers
  topic               str   — topic area slug

Note: expected_citations are primary targets.  A result is counted correct if
any element of expected_citations appears in the top-k results.
"""

QUERY_SET = [
    # -----------------------------------------------------------------------
    # 1. Employment income (s.5-8)
    # -----------------------------------------------------------------------
    {
        "query": "What amounts must a taxpayer include in computing employment income?",
        "expected_citations": ["ITA s.5(1)", "ITA s.5"],
        "topic": "employment_income",
    },
    {
        "query": "Are automobile allowances received by an employee taxable?",
        "expected_citations": ["ITA s.6(1)(b)", "ITA s.6(1)"],
        "topic": "employment_income",
    },
    {
        "query": "Can an employee deduct the cost of a home office from employment income?",
        "expected_citations": ["ITA s.8(13)", "ITA s.8(1)(i)"],
        "topic": "employment_income",
    },
    {
        "query": "What employment expenses are deductible for commissioned salespeople?",
        "expected_citations": ["ITA s.8(1)(f)", "ITA s.8(1)"],
        "topic": "employment_income",
    },
    # -----------------------------------------------------------------------
    # 2. Capital gains (s.38-55)
    # -----------------------------------------------------------------------
    {
        "query": "What fraction of a taxable capital gain is included in income?",
        "expected_citations": ["ITA s.38(a)", "ITA s.38"],
        "topic": "capital_gains",
    },
    {
        "query": "How is the adjusted cost base of a capital property calculated?",
        "expected_citations": ["ITA s.53", "ITA s.53(1)", "ITA s.54"],
        "topic": "capital_gains",
    },
    {
        "query": "What is the principal residence exemption and how is it calculated?",
        "expected_citations": ["ITA s.40(2)(b)", "ITA s.40(2)"],
        "topic": "capital_gains",
    },
    {
        "query": "How are capital losses applied against capital gains?",
        "expected_citations": ["ITA s.3(b)", "ITA s.38(b)", "ITA s.111(1)(b)"],
        "topic": "capital_gains",
    },
    # -----------------------------------------------------------------------
    # 3. Deductions from business/property income (s.20)
    # -----------------------------------------------------------------------
    {
        "query": "Can interest on borrowed money used to earn income be deducted?",
        "expected_citations": ["ITA s.20(1)(c)", "ITA s.20(1)"],
        "topic": "deductions",
    },
    {
        "query": "How is capital cost allowance (CCA) deducted from business income?",
        "expected_citations": ["ITA s.20(1)(a)", "ITA s.20(1)"],
        "topic": "deductions",
    },
    {
        "query": "Can a taxpayer deduct premiums paid on a life insurance policy for business purposes?",
        "expected_citations": ["ITA s.20(1)(e.2)", "ITA s.20(1)"],
        "topic": "deductions",
    },
    {
        "query": "Are reserves for doubtful debts deductible in computing business income?",
        "expected_citations": ["ITA s.20(1)(l)", "ITA s.20(1)"],
        "topic": "deductions",
    },
    # -----------------------------------------------------------------------
    # 4. Definitions (s.248)
    # -----------------------------------------------------------------------
    {
        "query": "How does the Income Tax Act define 'business'?",
        "expected_citations": ["ITA s.248(1)", "ITA s.248"],
        "topic": "definitions",
    },
    {
        "query": "What is the definition of 'property' under the Income Tax Act?",
        "expected_citations": ["ITA s.248(1)", "ITA s.248"],
        "topic": "definitions",
    },
    {
        "query": "How is 'taxable Canadian property' defined?",
        "expected_citations": ["ITA s.248(1)", "ITA s.248"],
        "topic": "definitions",
    },
    # -----------------------------------------------------------------------
    # 5. Corporate tax (s.125)
    # -----------------------------------------------------------------------
    {
        "query": "What is the small business deduction for Canadian-controlled private corporations?",
        "expected_citations": ["ITA s.125(1)", "ITA s.125"],
        "topic": "corporate_tax",
    },
    {
        "query": "What is the business limit for the small business deduction?",
        "expected_citations": ["ITA s.125(2)", "ITA s.125(1)"],
        "topic": "corporate_tax",
    },
    {
        "query": "How does associated corporation status affect the small business deduction limit?",
        "expected_citations": ["ITA s.125(3)", "ITA s.125(2)", "ITA s.256"],
        "topic": "corporate_tax",
    },
    # -----------------------------------------------------------------------
    # 6. International / non-resident (s.212-220)
    # -----------------------------------------------------------------------
    {
        "query": "What withholding tax rate applies to dividends paid to non-residents of Canada?",
        "expected_citations": ["ITA s.212(2)", "ITA s.212(1)"],
        "topic": "international",
    },
    {
        "query": "Are royalties paid to non-residents subject to withholding tax?",
        "expected_citations": ["ITA s.212(1)(d)", "ITA s.212(1)"],
        "topic": "international",
    },
    {
        "query": "What are the Part XIII withholding tax obligations when paying interest to non-residents?",
        "expected_citations": ["ITA s.212(1)(b)", "ITA s.212(1)"],
        "topic": "international",
    },
    # -----------------------------------------------------------------------
    # 7. RRSP / pension (s.146)
    # -----------------------------------------------------------------------
    {
        "query": "What is the RRSP contribution deduction limit?",
        "expected_citations": ["ITA s.146(1)", "ITA s.146(5)"],
        "topic": "rrsp",
    },
    {
        "query": "When must RRSP funds be converted and what are the options?",
        "expected_citations": ["ITA s.146(2)", "ITA s.146(3)"],
        "topic": "rrsp",
    },
    {
        "query": "Are amounts withdrawn from an RRSP included in income?",
        "expected_citations": ["ITA s.146(8)", "ITA s.146"],
        "topic": "rrsp",
    },
    {
        "query": "What are the consequences of making an over-contribution to an RRSP?",
        "expected_citations": ["ITA s.204.1", "ITA s.146(1)"],
        "topic": "rrsp",
    },
]


if __name__ == "__main__":
    from collections import Counter

    topic_counts = Counter(q["topic"] for q in QUERY_SET)
    print(f"Total queries: {len(QUERY_SET)}")
    print("By topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic:30s}: {count}")
