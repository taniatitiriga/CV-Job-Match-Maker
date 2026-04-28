---
name: ask-questions-if-underspecified
description: Ask the minimum set of clarifying questions needed to avoid wrong work.
license: Apache-2.0
compatibility: opencode
metadata:
  audience: developers
  workflow: opencode
---

# Ask Questions If Underspecified

Ask the minimum set of clarifying questions needed to avoid wrong work.

## When to Use

Use this skill when:
- The request has multiple plausible interpretations
- Success criteria, scope, constraints, or environment details are unclear
- Starting implementation without clarification risks doing the wrong work

## What It Does

- Asks 1–5 must-have questions in a scannable, answerable format (multiple choice + defaults)
- Pauses before acting until required answers are provided (unless the user approves proceeding on stated assumptions)
- Restates confirmed requirements before starting work