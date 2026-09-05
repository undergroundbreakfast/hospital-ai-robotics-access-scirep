# Geographic inputs

`code/geospatial_access_lorenz.py` expects these local files (not committed):

| File | Required contents |
|---|---|
| `code/shapefiles/tl_2024_us_county.shp` | Census 2024 county boundaries, with the matching `.shx`, `.dbf`, and `.prj`; `STATEFP` is used for the contiguous-US filter. |
| `code/shapefiles/USA_BlockGroups_2020Pop.geojson` | Census block-group geometry joined to 2020 total population; numeric `POPULATION` (or `population`, which the script renames) and a valid CRS. |

Public starting points: Census TIGER/Line geography at https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html and 2020 Census population at https://data.census.gov/.

The population file is a prepared study input, not an official Census filename. To prepare a comparable input, preserve full block-group GEOIDs as text, join population to block-group geometry by GEOID, validate one row per block group and positive numeric population, retain the correct CRS, and save the joined data using the expected filename. The mapping code then constructs centroids and its circuity-adjusted travel proxy. It is not an OSRM-routing engine.

The exact historical geography source bundle and join recipe were not archived in v1.0.4. A newly assembled file must therefore be compared with the study's access-analysis geography and reported population (334.7 million) before its outputs are treated as reproductions. Matching a total alone is not proof of matching geometry. Do not rescale population or tune a join simply to force agreement with the published totals.

The original Figure 1 is retained as an archived publication artifact. A figure synchronized to the accepted local manuscript is provided; further typography changes for production should be applied to the author-approved figure source and then synchronized here. Altering map values or curves to improve visual agreement is not a formatting correction.
