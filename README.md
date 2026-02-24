# PaperIQ
***Overview***:

The Research Paper Analysis Application is designed to help students and researchers analyze academic documents efficiently. The system allows users to upload research papers and provides detailed insights including quality scores, domain classification, summaries, visualizations, and improvement suggestions.

**Application Flow**:

**1.Login Page**:

Provides email validation.

Allows role selection (Student or Researcher).

**2.Upload Page**:

Users can upload research documents.

Supported file formats:

PDF (.pdf)

Word Document (.docx)

Text File (.txt)

**3.Analysis Dashboard**:

After uploading the document, the system performs analysis and displays the following features:

**_Top Metrics Bar:_**

Displays key document statistics:

Number of pages

Total word count

Number of sections identified

Extracted keywords

Quality Scores(Composite,Language,Coherance,Reasoning,Sophistication,Readability).

Displays six quality scores(The client gets a quality benchmark without reading a single line).

Uses TextBlob for text tokenization and analysis.

**_Domain Classification and Keywords_:**

Classifies the research domain based on word frequency.Compares keywords with predefined domain datasets.Assigns the domain with the highest matching score.

Example domains:

Computer Science,Machine Learning'Data Science'Healthcare'Engineering'

_**Section Summaries Tab**:_

Extracts individual sections from the document.

_Text Processing and Cleaning_:

Cleans text by removing Extra spaces,Special characters,Symbols,Unnecessary punctuation

Uses a HuggingFace Transformer model for abstractive summarization.

Generates rewritten summaries instead of copying original sentences.

_**Visualization Tab**_:

Provides graphical representation of quality metrics:

Radar charts

Bar charts

These visualizations help users quickly understand document quality.

_**Issues Tab**_:

Identifies sentences longer than 30 words.

Highlights complex sentences that may affect readability.

_**Suggestions Tab**_:

Provides improvement suggestions:

Vocabulary enhancement recommendations

Suggestions to improve clarity and readability

**_Download Feature_** :

Users can download analysis results in the following formats:

PDF Report

JSON File

_**Output**_:

<img width="1919" height="1014" alt="Image" src="https://github.com/user-attachments/assets/8ece7fd3-49ed-48d0-8877-d81febc17da4" />

<img width="1916" height="1021" alt="Image" src="https://github.com/user-attachments/assets/8bebb187-d738-4cb0-9f3f-a587af6b8e43" />

<img width="1919" height="1016" alt="Image" src="https://github.com/user-attachments/assets/67867503-bbe4-40f4-838c-55475d3397bc" />

<img width="1916" height="1019" alt="Image" src="https://github.com/user-attachments/assets/c15228b9-6d5a-4927-a6f7-be7a7ea108ce" />

<img width="1919" height="1023" alt="Image" src="https://github.com/user-attachments/assets/a192cf5e-a2ac-4fae-b57a-c4380da12f61" />

<img width="1919" height="1019" alt="Image" src="https://github.com/user-attachments/assets/11318026-6faa-4e38-9114-e1e06429a7b4" />

<img width="1919" height="1029" alt="Image" src="https://github.com/user-attachments/assets/69b865bc-bc78-4cfd-aa55-94c7cb713c98" />

<img width="1919" height="1031" alt="Image" src="https://github.com/user-attachments/assets/2e0d9248-5689-4b26-956f-de8d6ceff153" />

<img width="1919" height="1024" alt="Image" src="https://github.com/user-attachments/assets/7ee23bcc-afc2-4f85-b5b5-371de843310f" />

<img width="1919" height="1018" alt="Image" src="https://github.com/user-attachments/assets/55e5d4c7-ea05-4d49-bfdd-f68509b26aa2" />

<img width="1917" height="1019" alt="Image" src="https://github.com/user-attachments/assets/13360ccd-9d32-4cfa-a2d0-65e215f33eb4" />
