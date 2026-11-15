# ORBITA — submission form, ready-to-copy English fields

## Project Name

ORBITA — Predictive Digital Twin with Chatbot Assistance

## Summary

ORBITA is a research prototype of a predictive digital twin with chatbot assistance for future astronaut health monitoring. It combines NASA research and Earth-based wearable evidence while keeping their different origins and sampling frequencies explicit. A temporal model learns from available historical measurements and forecasts each signal with a median and an uncertainty range. The dashboard compares earlier forecasts with real observations, lets users explore measurements and prediction horizons, and offers a research chatbot that explains the results. NASA Inspiration4 biomarkers, bed-rest cardiovascular studies and ISS RadLab radiation provide complementary evidence; continuous glucose and pulse demonstrations use terrestrial wearables. Our next step is to collect more frequent, paired astronaut measurements to improve and validate predictions. ORBITA supports research and interpretation; it is not clinically validated.

Words: 125. Characters: 941. Within the suggested 50–200 words and 1,500-character field limit.

## Project Demonstration

Upload ONE `ORBITA-demo-VOICE.mp4` from this folder to a public hosting service. Paste its complete HTTPS URL here. The video is 30 seconds and includes the SI disclosure label and synthetic-audio metadata. The local Windows file path is not the public link.

Public demo URL: TO BE FILLED AFTER UPLOAD.

## Project

Paste a separate public URL for the final project: for example, a public code repository with the README, source, setup instructions, architecture, data references, model/evaluation artifacts and demonstration link. A live dashboard is useful but is not the only acceptable project format. Do not paste localhost.

Public project URL: TO BE FILLED AFTER PUBLICATION.

## Project Details

ORBITA is a research prototype built around a predictive digital twin and a contextual research chatbot. It addresses the need to understand how health and environmental signals evolve, see how previous forecasts performed, and interpret what might happen next without hiding uncertainty.

The interface gives users a single, approachable view with measurement cards, searchable study-specific outputs, observed and predicted trajectories, a visible forecast origin, and shaded q10–q90 bands. Users can independently choose the displayed history and future horizon. The initial view shows two hours of history and four hours ahead. The model retains its full 24-hour input context. Forecasts at 1, 4, 8 and 24 hours are available; continuous heads produce 96 trained quarter-hour queries through 24 hours. Sparse cardiovascular studies retain their original visit-based prediction cadence.

Source-specific encoders represent related measurements, missing-data masks and observation timing. A shared causal temporal Transformer learns from historical sequences; source-specific heads produce q10, q50 and q90. Missing labels are excluded from the corresponding loss, allowing available observations to contribute without inventing missing measurements. Forecasting experiments compare persistence and learned alternatives; an ensemble and interval adjustment are selected using separate development and calibration data. Results are evaluated on held-out participants or later time periods. Performance varies by signal and horizon; the prototype does not claim universal superiority or clinically validated accuracy.

The historical dashed line is a retrospective replay: each one-hour-ahead prediction uses only the observations available at its own issue time, with a fixed checkpoint. Later real observations are shown for comparison and are not inputs to the current forecast. A nominal q10–q90 range is not guaranteed to cover 80% of outcomes in every variable or window; measured historical error and coverage are exposed.

NASA data have distinct roles. Inspiration4 OSD-575 and OSD-569 biomarkers support the earlier exploratory visit-based astronaut model. NASA LSDA cardiovascular bed-rest studies support the current cardiac and vascular heads. NASA RadLab DosTel1 supports a separate ISS environmental radiation head. It is not personal astronaut dosimetry. BIG IDEAs wearable data support the continuous glucose and pulse demonstration on Earth. These sources are independent studies, not synchronized observations of one astronaut.

The chatbot requires the visitor to connect their own Codex, Claude Code or OpenCode account through a local connector. No shared account or provider key is published. The connected agent acts as a research guide. It explains the selected forecast, quantiles, historical performance and data provenance. It does not simulate causal meal or exercise effects, prescribe treatment, or claim specialist-level clinical validation.

Implementation uses Python, PyTorch, NumPy, pandas, SciPy and native HTML/CSS/JavaScript, with reproducible model artifacts and evaluation. The presentation combines actual product images and forecast outputs with clearly identified conceptual network animations. AI-assisted code and content, and synthetic narration, are disclosed.

The intended impact is a transparent research workflow for future crew health support: inspect change, compare predictions with reality, and identify where more evidence is needed. Our next priority is more participants and more frequent, paired astronaut measurements and activity records, followed by prospective validation and specialist review.

## Use of Super Intelligence (SI)

Select the options that correspond to code assistance, generated text, assisted/generated programmatic visuals/video and synthetic speech. The exact option labels were not included in the pasted form; match them to the actual uses below. Do not select the no-SI option. Do not claim generative synthesis of the measured source data, image-model generation, voice cloning or a trained clinical LLM, since those are not what this project used.

OpenAI Codex was used for AI/SI-assisted software development, data-processing and evaluation code, architectural discussion, interface implementation, explanatory text, diagram generation and programmatic video composition. Microsoft Edge neural text-to-speech, accessed through edge-tts, generated the English narration. The selected voice is identified in the video metadata and the voice-edition manifests. Programmatically generated/assisted visuals include an explicit visible SI label. The sound effects and closing signature are original procedural audio generated by code; no third-party music or astronaut likeness was used. Source measurements were downloaded from the cited research repositories and were not synthesized by a generative model. Model forecasts are identified as predictions and compared with real observations. The conceptual neural network is illustrative, not an empirical embedding of patient data. The research chatbot uses the visitor’s own signed-in local agent; it is not a fine-tuned clinical language model and does not share the author’s account. Team members must review the final submission and ensure the disclosure reflects their actual additional contributions.

## NASA Data

Enter each name/URL pair separately in the form.

**Link Text:** NASA OSD-575 — Inspiration4 metabolic and biomarker research

**Link URL:** https://osdr.nasa.gov/bio/repo/data/studies/OSD-575

Use: Earlier exploratory astronaut biomarker model; sparse pre-/post-flight visits.

**Link Text:** NASA OSD-569 — Inspiration4 complete blood count

**Link URL:** https://osdr.nasa.gov/bio/repo/data/studies/OSD-569

Use: Earlier exploratory astronaut biomarker model; sparse visits.

**Link Text:** NASA LSDA — Bed Rest Standard Measures: Cardiovascular Function

**Link URL:** https://nlsp.nasa.gov/search/data/details/9af6f9ba-3730-54ab-a82a-1a32798667ae?dataType=EXPERIMENTS

Use: Experiment context for current ground-analogue cardiovascular heads.

**Link Text:** NASA LSDA — UTMB C1 echocardiography

**Link URL:** https://nlsp.nasa.gov/search/data/details/95186651-28d7-5c72-8f32-d40838766078?dataType=DATASETS

Use: Current visit-based cardiac/vascular research heads.

**Link Text:** NASA LSDA — UTMB C3 echocardiography

**Link URL:** https://nlsp.nasa.gov/search/data/details/c3bdfa7e-6661-5891-9ba0-0fc7c7a080d6?dataType=DATASETS

Use: Current visit-based cardiac head.

**Link Text:** NASA LSDA — CFT70 cardiovascular measurements

**Link URL:** https://nlsp.nasa.gov/search/data/details/59501092-7252-5848-b79c-650e954b51ed?dataType=DATASETS

Use: Current 2D/3D cardiac heads; overlapping people are not counted twice.

**Link Text:** NASA OSDR RadLab — ISS DosTel1 environmental radiation

**Link URL:** https://visualization.osdr.nasa.gov/radlab/gui/data-api/

Use: Current environmental radiation forecast head; not personal dosimetry.

## Space Agency Partner & Other Data

Enter these resources and any additional resources personally used by your team. Resources used only in earlier experiments are explicitly distinguished from the current demonstration.

**Link Text:** BIG IDEAs glycemic wearable dataset v1.1.3

**Link URL:** https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/

Use: Earth-based glucose, heart-rate and meal research; current continuous demonstration.

**Link Text:** PhysioNet Challenge 2019 v1.0.0

**Link URL:** https://physionet.org/content/challenge-2019/1.0.0/

Use: Earlier terrestrial forecasting benchmarks, not the current astronaut or wearable data.

**Link Text:** CGMacros v1.0.0

**Link URL:** https://physionet.org/content/cgmacros/1.0.0/

Use: Audited during data research; excluded from the current forecasting campaign because of unresolved interpolation concerns.

**Link Text:** Canadian Space Agency — monitoring radiation and astronaut health

**Link URL:** https://www.asc-csa.gc.ca/eng/news/articles/2026/2026-02-26-monitoring-radiation-astronaut-health.asp

Use: Background on the mission-health monitoring problem; not a paired training dataset.

**Link Text:** OpenAI Codex

**Link URL:** https://openai.com/codex/

Use: AI/SI-assisted development, analysis, text and programmatic visual/video production.

**Link Text:** edge-tts

**Link URL:** https://github.com/rany2/edge-tts

Use: Access to Microsoft Edge neural speech for synthetic narration.

**Link Text:** Python

**Link URL:** https://www.python.org/

Use: Implementation language.

**Link Text:** PyTorch

**Link URL:** https://pytorch.org/

Use: Neural forecasting models.

**Link Text:** NumPy

**Link URL:** https://numpy.org/

Use: Numerical processing.

**Link Text:** pandas

**Link URL:** https://pandas.pydata.org/

Use: Tabular processing.

**Link Text:** SciPy

**Link URL:** https://scipy.org/

Use: Scientific computations.

**Link Text:** scikit-learn

**Link URL:** https://scikit-learn.org/

Use: Earlier baseline and tree-model experiments.

**Link Text:** Matplotlib

**Link URL:** https://matplotlib.org/

Use: Research evaluation figures.

**Link Text:** Pillow

**Link URL:** https://python-pillow.org/

Use: Programmatic visual composition.

**Link Text:** FFmpeg

**Link URL:** https://ffmpeg.org/

Use: Video encoding, audio alignment and mixing.

**Link Text:** diagrams.net

**Link URL:** https://www.diagrams.net/

Use: Editable architecture diagram format.

**Link Text:** Segoe UI typeface

**Link URL:** https://learn.microsoft.com/en-us/typography/font-list/segoe-ui

Use: System typeface rendered into the visual presentation; font binaries are not redistributed.

## Submission actions

1. Check the public demo and project links in a private/incognito window without logging in.
2. Review the title, descriptions, NASA attribution, SI disclosure and original-work confirmation.
3. Save Progress saves the page; it does not submit for judging.
4. Click Submit for Judging and verify the confirmation.
5. The pasted form states a deadline of Sunday, November 15, 2026, at 11:59 PM in the Local Event's time zone (or the Team Owner's residence for the Universal Event).

## Hosting description to accompany the video

ORBITA is a research prototype of a predictive digital twin with chatbot assistance for future crew health monitoring. SI-generated presentation visuals and AI/SI-assisted software were created with OpenAI Codex. The English narration is synthetic, generated using Microsoft Edge neural text-to-speech through edge-tts; the selected voice is identified in the video metadata. Source measurements come from the cited NASA and terrestrial research datasets and were not synthesized. Network animations are conceptual. This is not a clinically validated monitoring or treatment system.

Team: ViMas Tech. Add the public project URL and dataset/resource links from this document to the hosting description.
