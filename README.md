# NLP_ET09_Named_Entity_Recognition
# NLP_ET09_NamedEntityRecognition

## Named Entity Recognition (NER) using spaCy

**Student Name:** Trisha Ramesh Barapatre
**Roll Number:** ET09
**Semester/Branch:** V Semester ETC
**Course:** Natural Language Processing (ET5M004)

---

## Problem Statement / Objective
To build an NLP-based system that can automatically identify and classify named entities (such as people, organizations, locations, dates, and monetary values) within a given text using a pre-trained industry-standard NLP pipeline.

## Introduction
Named Entity Recognition (NER) is a fundamental NLP task used to extract structured information from unstructured text. It is widely used in applications like resume parsing, chatbots, search engines, and information retrieval systems. Unlike text classification (which labels an entire sentence), NER performs **token-level classification**, identifying and labeling specific words or phrases within a sentence — making it a more granular and complex NLP task.

## NLP Technique / Method Used
- **spaCy** pre-trained NLP pipeline (`en_core_web_sm`) for Named Entity Recognition
- Entity extraction and classification (PERSON, ORG, GPE, DATE, MONEY, etc.)
- Visualization of extracted entities using `disp