# Password & Multi-Factor Authentication Standard (STD-SEC-002)

## Password Requirements
Passwords must be at least 14 characters and are checked against a breached-password
list at creation time. Rotation is no longer forced on a fixed schedule; instead,
passwords are rotated only on suspicion of compromise, in line with NIST 800-63B.

## Multi-Factor Authentication (MFA)
MFA is mandatory for all employees on every system that supports it. Approved
factors are: hardware security keys (FIDO2/WebAuthn), the corporate authenticator
app, and platform passkeys. SMS one-time codes are deprecated and only permitted
as a temporary fallback during device enrollment.

## Self-Service Password Reset
Employees can reset their own password through the identity portal after passing
an MFA challenge. If MFA devices are lost, a help-desk identity-verification flow
is required before a reset is issued.

## Account Lockout
After 10 failed sign-in attempts an account is locked for 15 minutes. Repeated
lockouts trigger an automatic alert to the Security Operations Center.

## Help-Desk Guidance
For "I forgot my password" tickets where the user still has a working MFA device,
direct them to the self-service portal — no manual reset is required.
