# The Testing Pyramid

The testing pyramid is a way of thinking about how many tests of each kind a healthy codebase should have. At the base sit unit tests: fast, numerous, each checking one function or class in isolation. In the middle sit integration tests, fewer in number, checking that two or more real pieces work together correctly, such as a repository talking to an actual database. At the top sit end-to-end tests, the fewest of all, exercising the whole system the way a user would.

The shape matters because of cost. A unit test runs in milliseconds and fails with a precise, local reason. An end-to-end test can take seconds, depends on more moving parts, and a failure often needs real investigation to trace back to its cause. A suite that inverts the pyramid, with mostly end-to-end tests and few unit tests, tends to be slow to run and painful to debug, even though every individual test still adds some confidence.

None of this means integration and end-to-end tests are optional. Unit tests cannot catch a wrong SQL join or a misconfigured route; only a test that touches the real pieces can. The pyramid is a guide for proportion, not a reason to skip the top.
