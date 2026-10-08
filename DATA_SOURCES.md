# Data sources and acknowledgments

## Cassini ISS / NASA

The Cassini-derived `real` branch uses observations from the **Imaging Science
Subsystem (ISS)** of the **Cassini-Huygens mission**. Cassini ISS comprises
narrow-angle and wide-angle imaging cameras. The underlying spacecraft
observations are credited to **NASA / JPL-Caltech / Space Science Institute**.

Official source and archive references:

- [NASA: Cassini-Huygens mission](https://science.nasa.gov/mission/cassini/)
- [NASA Planetary Data System (PDS)](https://pds.nasa.gov/)
- [PDS Ring-Moon Systems Node: Cassini ISS data archive](https://pds-rings.seti.org/cassini/iss/)
- [PDS OPUS archive search](https://opus.pds-rings.seti.org/)
- [JPL image-use policy](https://www.jpl.nasa.gov/jpl-image-use-policy/)

These are the upstream mission and archive references, not a claim that NASA
produced or endorsed the ASTRA-SR benchmark. Retain any additional credit
specified for individual source products.

## ASTRA-SR processing

The ASTRA-SR authors curated and preprocessed the clean-source images, formed
the 512x512 / 256x256 resampled pairs, and synthesized spatially varying blur
and Gaussian noise under the frozen protocol. The release contains float32
FITS arrays, source-ID split assignments and per-sample protocol/PSF hashes.
It is a processed research dataset, not an unmodified mirror of NASA raw
products. Source names remain in `dataset_index.jsonl` and the historical
`source_manifest_strict_v2_x2_v1.jsonl`.

Suggested credit when using this release:

> ASTRA-SR dataset and processing: Xining Ge, Ziteng Cui and Shuhong Liu.
> Cassini ISS source observations: NASA / JPL-Caltech / Space Science Institute.
> Archive reference: NASA Planetary Data System, Ring-Moon Systems Node.

Also cite [ASTRA-SR, arXiv:2609.26731](https://arxiv.org/abs/2609.26731), link
the dataset and [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
and describe any further modifications you make.

## Other inputs and scope

The auxiliary `png` branch is a separate historical source collection; this
notice does **not** label every `png` sample as NASA/Cassini imagery. Its full
upstream source/license mapping is still pending maintainer verification.
The MASS atmospheric-observation CSV and the full PSF-bank population are
also not bundled here. Exact archive-product/download-version mappings are
not reconstructed merely from the processed FITS filenames.

The noncommercial project license applies only to the authors' licensable
contributions. Original NASA/Cassini and other third-party material retain
their applicable source terms; see [LICENSE_SCOPE.md](LICENSE_SCOPE.md).
