# Publication package verification — 2026-09-05

Base: public v1.0.4, commit 41968545a95b0b2da36f5e4d16e1b9fd8d66aade.

- Python 3.11.15, pinned requirements: all three main workflows plus econml, pygam, and plotly imported successfully.
- A two-fold, 200-row synthetic cross-fitted AIPW calculation completed with finite estimates/intervals and the expected known-effect neighborhood. No study data were used for this smoke check.
- Offline pytest: 20 passed; 8 database checks skipped by design.
- Optional database contracts: 8 passed against the prepared local research database using read-only connections.
- Empty PostgreSQL 16: source contracts and all 31 views built successfully in dependency order.
- Candidate release scan: no obvious private-key/token/home-path candidates. Generated hospital-level exports are ignored and excluded from the manifest.
- Main figures synchronized byte-for-byte to the accepted manuscript source files; no underlying plotted values were altered. Figure 1's internal typography remains a production-level review item.
- CT6 uncertainty/ESS corroborated against the saved May revision-support output. Hospital outcome/baseline counts matched Table 1 in an aggregate read-only check.

These checks validate the repair and package, not a full licensed-source statistical rerun. Original source staging and exact geography preparation remain prerequisites documented in database_setup.md and geographic_inputs.md. The new dependency snapshot is not represented as the original historical environment.
