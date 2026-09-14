# Prompt: Generate IT Ticket Triage Queries with Gold Labels

Copy everything below into another AI model to generate additional queries in the same format as the existing dataset.

---

## PROMPT START

You are generating synthetic IT service-desk queries for a research dataset that studies how reliably an LLM can infer structured fields from a short, informal IT request. The dataset is used to test both **accuracy** (does the model's inferred field match a human-agreed answer) and **stability** (does the model give the same answer every time at temperature=0).

### Output schema per query

Each query must be inferred into these fields:

- `request_type`: one of `access_grant`, `access_revoke`, `tool_install`, `password_reset`, `hardware_request`
- `resource`: free text naming what's being requested (e.g. "Claude Code", "shared drive", "VPN", "admin rights")
- `scope`: one of `individual`, `team`, `department`, `org_wide`
- `severity`: one of `low`, `medium`, `high`
- `requester_count_estimate`: one of `1`, `~5-10`, `~10-50`, `50+`

### Severity inference rule (must be applied consistently)

- **low**: individual scope, non-privileged resource
- **medium**: team scope, OR individual scope + a sensitive (but non-admin) resource
- **high**: department/org-wide scope, OR anything involving admin/root/privileged access, regardless of scope

### Resource types to draw from (use a mix)

`software_access`, `password_reset`, `vpn_access`, `shared_drive_access`, `admin_access`, `hardware_request` (also `tool_install` as a request_type paired with any resource)

### Two query categories — generate BOTH, roughly balanced

**1. Unambiguous queries**
The query must contain enough explicit information (a stated headcount, a named individual, an explicit "the whole department," an explicit privilege level like "root" or "read-only") that a human triager would assign exactly ONE correct value per field with high confidence.

**2. Ambiguous queries**
The query must be realistically underspecified in at least one dimension — e.g. it names a group ("the team," "the new hires," "some folks," "everyone affected") without a headcount, or it names an access level in vague terms ("elevated access," "some shared files") without clarifying privilege scope. These should read like real, casually-written help-desk tickets, not artificially confusing ones.

### Gold label format — IMPORTANT

- For **unambiguous** queries: each field gets a list containing exactly **one** value (the single correct answer).
- For **ambiguous** queries: each field gets a list of **all values a reasonable human triager could defensibly assign** given the information in the query — i.e. the genuine range of acceptable interpretations, not just "throw in extra values to be safe." Only widen a field's list if the query text genuinely fails to disambiguate that field. If a field IS clear even in an otherwise-ambiguous query (e.g. scope is unclear but severity is pinned to "high" because root access is explicitly mentioned), give that field a single-value list.
- Every query also needs a one-sentence `notes` field explaining *why* the gold label(s) were chosen — this is required for defensibility, not decoration.

### Output format

Return a JSON array, one object per query, in this exact shape:

```json
{
  "id": "u21",
  "ambiguous": false,
  "resource_type": "vpn_access",
  "query": "<the query text>",
  "gold_scope": ["<value(s)>"],
  "gold_severity": ["<value(s)>"],
  "gold_count_estimate": ["<value(s)>"],
  "notes": "<one sentence justifying the gold label(s)>"
}
```

Use sequential IDs: unambiguous queries prefixed `u`, ambiguous queries prefixed `a`, continuing numbering from wherever the existing dataset left off (specify the starting number when you use this prompt).

### Constraints

- Keep queries short and natural — the way a real employee would type a help-desk ticket, not a formally worded spec.
- Don't repeat scenarios already covered in the existing dataset (vary resource, phrasing, and department/context).
- Avoid making ambiguous queries ambiguous on EVERY field — real tickets are usually only unclear on one or two dimensions (headcount, or privilege level, or which access level exactly), not all three at once. Pin down whichever fields the text actually does make clear.
- Do not include any explanation, preamble, or text outside the JSON array — return only the JSON array of query objects.

### Number of queries to generate

Generate [N] queries total: aim for a roughly even split between unambiguous and ambiguous, and spread them across all six resource types.

## PROMPT END
