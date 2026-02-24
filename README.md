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
