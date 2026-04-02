---
title: "Resume"
layout: "page"
url: "/resume/"
summary: "Resume"
---
[📄 Download PDF](/resume.pdf)

---

## Summary

Physics PhD turned data scientist, with five years building and deploying
end-to-end ML systems that move the needle. I spent a decade in experimental
and computational physics before moving into industry. Since then I've designed
deep learning architectures, LLM pipelines, and automated ML workflows for
patent analytics and financial data — work that has directly driven hundreds of
thousands of dollars in revenue. I'm drawn to problems where the modeling is
genuinely hard, and I'll happily go as deep as the problem requires to get it
right.

---

## Work Experience

**Staff Data Scientist** — Moat Metrics, Spokane, WA *(Jun 2024 – Mar 2026)*

- Built an end-to-end patent valuation pipeline: trained a PyTorch MLP to
predict price-to-earnings from patent embeddings aggregated at the portfolio
level, with polynomial bias correction to calibrate output distributions
against actual PE values; deployed on AWS Batch with weekly automated
retraining and daily inference runs across thousands of entities using
Polars-based data retrieval from Snowflake
- Engineered LLM pipelines for large-scale patent text processing, including
few-shot and chained prompting, structured output extraction, and local LLM
inference via vLLM with KV caching to optimize throughput; deployed batch
workloads on GCP
- Created an ETL pipeline for a Snowflake database linking patents to
government contracts, including parsing and cleaning of malformed patent IDs
and contract numbers from heterogeneous sources, resulting in a $250,000
external contract
- Led a data wrangling and analysis exercise contracted by a high-profile
private equity firm, delivering results that generated $200,000 in revenue
above the base contract
- Contributed on a project constructing a knowledge graph on 5 million active
US patents through batch LLM parsing, local LLM calls, and Boolean
satisfiability (SAT) algorithms

**Research Data Scientist** — Aon PLC, Spokane, WA *(Sep 2021 – Jun 2024)*

- Developed an Adaptive Clustering System with a human-in-the-loop retraining
loop: user-corrected labels backpropagate through a jointly trained deep
autoencoder and classifier via masked loss, with convergence-based stopping
criteria that halts training when user overrides are satisfied rather than at a
fixed epoch count
- Built the supporting training infrastructure: S3-based checkpoint/resume
system supporting incremental retraining across sessions, and a coordinate
warping system that visually tightens cluster separation in the projected
embedding space
- Built an entity alignment system using Elasticsearch blocking and Conditional
Random Fields to probabilistically resolve patent entity identity across
heterogeneous data sources
- Implemented graph traversal algorithms to reconstruct patent assignment
histories across complex ownership chains

**Senior Research Associate (Postdoctoral)** — Duke University, Durham, NC *(Jun 2019 – May 2021)*

- Characterized trapped ions via microwave and optical transitions; optimized
PID laser stabilization and performed high-sensitivity interferometric
vibration analysis of the trapping apparatus
- Mentored 3 graduate students across physics and ECE departments

**Senior Research Associate (Postdoctoral)** — University of Oregon, Eugene, OR *(Jun 2018 – Jun 2019)*

- Conceived and modeled a quantum network based on diamond defect centers,
characterizing vibrational resonances and band structure via COMSOL
Multiphysics and building a scanning confocal microscope for defect measurement

**Research Assistant (PhD Candidate)** — University of Oregon, Eugene, OR *(Jun 2010 – Jun 2018)*

- Built Python simulation libraries for light-matter interactions and designed
automated data acquisition systems integrating hardware and software for
real-time experiment monitoring
- Directed 4 major projects end-to-end (theory → experiment → publication);
contributed to 17 peer-reviewed publications and 2 patents

---

## Education

**PhD in Physics** — University of Oregon, Eugene, OR *(June 2018)* · GPA: 3.88

**Bachelor of Science in Physics** — Washington State University, Pullman, WA *(May 2009)* · GPA: 3.5

---

## Skills

**Programming & Tools:** Python, SQL, COMSOL Multiphysics, PyTorch, Polars,
Numpy, Matplotlib, Scikit-Learn, Statsmodels, DuckDB, Postgres, Snowflake,
Elasticsearch, Git, Docker, Virtual Environments and UV, AWS (EC2, S3, Batch),
GCP, vLLM, Gitlab CI/CD Pipelines, Terraform (limited)

**Data Science & ML:** Deep Neural Networks, LLM Prompt Engineering (zero-shot,
few-shot, chaining), Vector Search and Embeddings, Approximate Nearest
Neighbors (ANN), Signal Processing, Time-series Analysis, Regression, KMeans
and HDBSCAN Clustering, KNN Classification, Conditional Random Fields,
Human-in-the-Loop, Active Learning, Joint Optimization, Metric Learning,
Manifold Learning, Bayesian Inference

**Soft Skills:** Scientific writing (17 peer-reviewed publications with over
2,000 citations), cross-disciplinary collaboration, mentorship, technical
communication

---

## Awards & Honors

- Marthe E. Smith Memorial Science Scholarship · 2012
- Weiser Sr Teaching Assistant Award · 2016
- The Optical Society Quantum Optical Science & Technology Technical Group Outstanding Poster Presentation Award · 2017
