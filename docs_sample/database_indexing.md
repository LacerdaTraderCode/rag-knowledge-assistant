# How Database Indexes Work

A database index is a separate data structure, usually a B-tree, that stores a sorted copy of one or more columns alongside a pointer back to the full row. Without an index, finding a row means scanning the whole table; with one, the database can jump almost directly to the matching rows, the same way a book's index lets a reader skip straight to a page instead of reading cover to cover.

Indexes speed up reads but slow down writes, because every insert or update has to keep the index's sorted structure in sync with the table. That tradeoff is why indexing every column is not a good default: an index only earns its cost if queries actually filter or sort by that column often enough to matter.

Composite indexes, built across more than one column, help queries that filter on several columns together, but only when the columns are listed in the index in the same order the query filters on them. An index on (user_id, created_at) speeds up a query filtering by both, but does little for a query that only filters by created_at.
