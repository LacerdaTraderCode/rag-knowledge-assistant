# Message Queues and Delivery Guarantees

A message queue lets one part of a system hand work to another without waiting for it. The producer drops a message on the queue and moves on, and consumers pick messages up at their own pace, which smooths out sudden traffic spikes that would otherwise overwhelm the downstream service.

Brokers offer different delivery guarantees. At-most-once delivery never duplicates a message but may lose one; at-least-once delivery never loses a message but may deliver it more than once; exactly-once delivery is expensive and is usually approximated by combining at-least-once delivery with deduplication. Because at-least-once is the common default, consumers should be idempotent: handling the same message twice must have the same effect as handling it once, which is often achieved by storing an idempotency key for every processed message.

Messages that keep failing should not block the queue forever. After a limited number of retries, a broker can move them to a dead-letter queue, where they wait for a person or a repair job to inspect them.
