# REST API Design Principles

A REST API organizes a system around resources, not actions. Each resource gets a URL, and the HTTP method on that URL says what to do: GET reads, POST creates, PATCH updates part of a resource, and DELETE removes it. Good REST APIs are also stateless: every request carries everything the server needs to handle it, so no request depends on a previous one being remembered in server memory.

Status codes matter more than most teams give them credit for. A 200 means success, a 201 means something was created, a 404 means the resource does not exist, and a 401 means the caller was not authenticated. Returning 200 for everything, with the real result buried in the response body, forces every client to parse the body just to know whether the call worked.

Versioning is worth planning before the first breaking change, not after. A version segment in the URL, like /v1/, or a version header are the two common approaches, and either is fine as long as it is decided early.
