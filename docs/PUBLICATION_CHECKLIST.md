# Public-release checklist

- [ ] Confirm that `SC-MSF` is the final public repository and method name.
- [ ] Replace `SC-MSF Authors` in `LICENSE` with the agreed copyright holder.
- [ ] Add all manuscript authors and a valid `CITATION.cff`.
- [ ] Add the paper title, DOI or preprint URL, and BibTeX entry to `README.md`.
- [ ] Provide feature extraction scripts or an archival feature download.
- [ ] Verify final metrics against every manuscript table and figure.
- [x] Include the selected JAAD checkpoint with its checksum and source snapshot.
- [ ] Record the repository commit and archived-code DOI in the manuscript.
- [ ] Confirm all third-party notices and dataset terms.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `python -m compileall -q configs dataset SC_MSF tools utils`.
- [ ] Scan Git history for data, credentials, absolute private paths, and large files.
- [ ] Add a Data Availability Statement and Code Availability Statement to the paper.

Springer Nature requires original research articles to include a Data
Availability Statement. The statement should distinguish official PIE/JAAD
data, derived features, code, weights, and generated results, and should state
the access conditions for each item.
