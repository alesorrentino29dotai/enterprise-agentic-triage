# Data Classification & Handling Standard (STD-DAT-005)

## Classification Tiers
- **Public** — may be shared freely (marketing material, published docs).
- **Internal** — default tier for day-to-day business data; not for external sharing.
- **Confidential** — customer PII, financials, source code. Encrypted at rest and
  in transit; access logged.
- **Restricted** — secrets, credentials, regulated data (PCI, health). Access is
  strictly need-to-know and requires Security approval.

## Storage Rules
Confidential and Restricted data may only live in approved, encrypted systems.
Copying such data to personal devices, personal cloud storage, or unmanaged
laptops is prohibited.

## Sharing Externally
Sharing Confidential data outside the company requires a signed NDA on file and a
data-sharing approval ticket. Restricted data may never be shared externally
without Legal and Security sign-off.

## Retention
Customer PII is retained only as long as there is a lawful business purpose and is
deleted on verified request within 30 days. Financial records follow the 7-year
statutory retention window.

## Incident Reporting
Any suspected exposure of Confidential or Restricted data must be reported to the
Security Operations Center within one hour of discovery.
