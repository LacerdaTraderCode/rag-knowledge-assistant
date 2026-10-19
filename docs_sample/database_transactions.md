# Database Transactions and Isolation

A transaction groups several statements into one unit of work that either completes entirely or leaves no trace. The guarantees are usually summarized as ACID: atomicity, consistency, isolation, and durability. Atomicity is what protects a bank transfer: if the system crashes after taking money out of one account but before putting it into the other, the database rolls the whole transfer back instead of leaving the money in limbo. Durability means that once a commit is acknowledged, the change survives a crash or a power loss.

Isolation decides how much concurrent transactions can see of each other's unfinished work. Weak levels allow anomalies such as dirty reads, where a transaction reads data another one has not committed yet, and phantom rows, where a repeated query suddenly returns extra matches. The strongest level, SERIALIZABLE, makes concurrent transactions behave as if they ran one after another, at the price of more conflicts and lower throughput.

When two transactions each hold a lock the other needs, the result is a deadlock. Databases detect it and abort one of them, so application code should be ready to retry, and transactions should stay short to keep the window for conflicts small.
