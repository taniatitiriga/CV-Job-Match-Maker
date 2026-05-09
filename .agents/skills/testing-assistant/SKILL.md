---
name: testing-assistant
description: Assist with writing, running, and maintaining tests for code
license: Apache-2.0
compatibility: opencode
metadata:
  audience: developers
  workflow: opencode
---

# Testing Assistant

This skill helps with writing, running, and maintaining tests for code to ensure quality and reliability.

## When to Use

Use this skill when:
- You need to write unit tests, integration tests, or end-to-end tests
- You want to improve test coverage for existing code
- You're debugging and need to create reproduction tests
- You want to set up or improve testing frameworks and practices
- You're refactoring code and need to ensure tests still pass

## What It Does

- Helps write test cases for various testing frameworks
- Suggests edge cases and scenarios to test
- Assists with test setup, teardown, and mocking
- Explains testing concepts and best practices
- Helps interpret test results and failures
- Assists with test maintenance and refactoring

## How to Use

1. Specify what code you want to test
2. Indicate the testing framework you're using (if any)
3. Describe what aspects or behaviors you want to test
4. The assistant will help write test cases
5. Review and run the generated tests
6. Iterate based on test results and feedback

## Best Practices

- Write tests that are independent and isolated
- Test both positive and negative cases
- Focus on behavior, not implementation details
- Keep tests fast and reliable
- Use descriptive test names that explain what's being tested
- Follow the testing pyramid (unit > integration > e2e)
- Maintain tests as you would production code