# OAuth 2.0 in Plain Terms

OAuth 2.0 lets a person give an application limited access to their data on another service without ever handing the application their password. In the authorization code flow, the application sends the person to the provider, the person approves a list of scopes such as "read contacts", and the provider returns a short-lived authorization code that the application exchanges for an access token.

Mobile and single-page apps cannot keep a client secret safe, so they add PKCE. Before starting, the app generates a random code verifier and sends only a hash of it; when it later redeems the code, it must present the original verifier, which makes a stolen code useless to anyone else.

Access tokens expire quickly on purpose. When an API receives an expired token it answers 401, and the client can use its refresh token to obtain a new access token without sending the person through the login screen again. Refresh tokens are long-lived, so they deserve stricter storage than access tokens.
