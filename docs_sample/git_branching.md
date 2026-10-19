# Git Branching Strategies

A branching strategy decides how work moves from an idea into the main line of a repository. In trunk-based development, everyone integrates into main at least once a day through short-lived branches, and unfinished features hide behind feature flags. The opposite habit, keeping a feature on a long-lived branch for weeks, lets the branch drift away from main and turns the eventual merge into a painful afternoon of conflicts.

There are two ways to bring a branch up to date. A merge adds a merge commit that ties the two histories together, preserving exactly what happened. A git rebase instead replays the branch's commits on top of the latest main, producing a straight, linear history, but it rewrites commit hashes. For that reason, commits that other people have already pulled should never be rebased.

A squash merge is a third option: it collapses all the commits of a branch into one before landing it on main, trading detailed history for a tidy log.
